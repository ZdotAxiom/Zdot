"""
Ż Python Transformer (SES v2.3.1-aligned, patched full code)
============================================================

This file is a *drop-in runnable* .py version of the "Ż python transformer" PDF,
patched to (a) run without PDF-induced syntax issues, and (b) remove the key
theory contradiction around π sign by aligning with SES v2.3.1.

Core alignment decisions (primary sources):
- SES defines a danger score π↑(x)∈[0,1] where *larger means less definable / more anomalous*,
  and all control should be written in terms of π↑. (SES v2.3.1 §1.1) 
- SES defines Dε = {x | π↑(x) ≤ ε} and the operational vanishing horizon 𝒱ε = ∂Dε. 
- SES defines operational pressure magnitude via one-step improvability:
    ‖P(x)‖ := r(x) = max_{o∈𝒪(x)} (π↑(x) − π↑(o(x)))₊. 
- ZSIT paper reiterates π↑/σπ and the improvability-based pressure definition.  

What was contradicted in the PDF code:
- The PDF’s Z-CP uses “π(x) ≈ exp(-Δ)” based on hidden-state distance, which makes π *larger when
  things are stable/similar*, i.e., the opposite sign to SES π↑ (danger). 
- The PDF’s CollapseDetector mixes entropy terms with that π, so the trigger semantics drift. 

Patch summary:
1) Replace the exp(-Δ) π with SES-style π↑ (danger) as an ensemble of monotone anomaly metrics
   (output entropy, attention entropy, hidden variance collapse proxy, attention rank collapse proxy),
   normalized by rolling quantiles; also track σπ as disagreement.
2) Replace pressure trigger with SES operational pressure proxy using one-step improvability over a small
   set of admissible transformations O(x) implemented as truncation-based ZFilter.
3) Fix common runtime hazards:
   - guard against max_len overflow (positional embeddings)
   - detach Langevin steps to avoid graph blow-up during generation
   - make the script runnable standalone via a SimpleMockTokenizer demo

Dependencies:
  - torch
  - numpy
  - optional: matplotlib (demo plot)

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. Configuration
# =========================================================

@dataclass
class ZConfig:
    # Transformer
    vocab_size: int = 50257
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    max_len: int = 1024

    # SES-aligned control
    # π↑ : larger => more undefinable/anomalous (danger)
    pi_sensitivity: float = 1.0
    epsilon_definable: float = 0.70     # ε for Dε = {x | π↑(x) ≤ ε}
    pressure_threshold_init: float = 0.85  # τP in practice

    # Online quantile calibration (dev-set substitute)
    pi_calib_window: int = 256
    pi_q50: float = 0.50
    pi_q95: float = 0.95

    # Z-LFE (Langevin)
    langevin_steps: int = 5
    langevin_temperature: float = 0.1
    langevin_step_size: float = 0.01

    # Z-OTD (Sinkhorn-inspired)
    sinkhorn_iters: int = 5
    sinkhorn_epsilon: float = 0.05

    # Z-Imprint
    memory_slots: int = 1024
    imprint_coherence_threshold: float = 0.9

    # Collapse detection
    collapse_history_window: int = 64
    collapse_cooldown_steps: int = 3


# =========================================================
# 2. Math Kernel
# =========================================================

class ZMathKernel:
    @staticmethod
    def calc_entropy(probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
        val = probs * torch.log(probs + 1e-10)
        return -torch.sum(val, dim=dim)

    @staticmethod
    def attention_rank_score(attn: torch.Tensor) -> float:
        """
        attn: (B, H, T, T)
        Return a scalar rank-like score (higher => less collapsed).
        Uses nuclear_norm / frobenius_norm as a cheap proxy.
        """
        B, H, T, _ = attn.shape
        A = attn.reshape(B * H, T, T)
        scores: List[float] = []
        for i in range(A.shape[0]):
            Ai = A[i]
            # Use SVD if available; fallback to sym eigvals (rough).
            try:
                s = torch.linalg.svdvals(Ai)
            except Exception:
                s = torch.linalg.eigvalsh((Ai + Ai.transpose(-2, -1)) / 2.0).abs()
            num = s.sum()
            denom = torch.norm(s) + 1e-8
            scores.append(float((num / denom).item()))
        return float(np.mean(scores)) if scores else 0.0


# =========================================================
# 2.5 Online Rolling Quantile Calibrator (SES-style)
# =========================================================

class RollingQuantileCalibrator:
    """
    Keeps a rolling buffer and returns q50/q95; score() maps values to [0,1].
    """
    def __init__(self, window: int = 256, q50: float = 0.50, q95: float = 0.95):
        self.window = int(window)
        self.q50 = float(q50)
        self.q95 = float(q95)
        self.buf: deque[float] = deque(maxlen=self.window)

    def update(self, v: float) -> None:
        self.buf.append(float(v))

    def quantiles(self) -> Tuple[float, float]:
        if len(self.buf) < 8:
            return (0.0, 1.0)
        arr = np.array(self.buf, dtype=np.float64)
        return (float(np.quantile(arr, self.q50)), float(np.quantile(arr, self.q95)))

    def score(self, v: float) -> float:
        q50, q95 = self.quantiles()
        denom = (q95 - q50) + 1e-8
        s = (float(v) - q50) / denom
        return float(np.clip(s, 0.0, 1.0))


# =========================================================
# 3. Core Z Modules: Z-LFE, Z-OTD, Z-Imprint
# =========================================================

class ZLangevinBlock(nn.Module):
    """
    [Z-LFE] Langevin Functional Engine (Bloom refinement).
    """
    def __init__(self, config: ZConfig):
        super().__init__()
        self.config = config
        self.energy_proj = nn.Linear(config.dim, 1)
        self.norm = nn.LayerNorm(config.dim)

    def energy_function(self, x: torch.Tensor, context: Optional[torch.Tensor]) -> torch.Tensor:
        E_self = self.energy_proj(x).squeeze(-1)  # (B,L)
        if context is not None:
            ctx_mean = context.mean(dim=1, keepdim=True)  # (B,1,D)
            dist = torch.norm(x - ctx_mean, dim=-1)       # (B,L)
            E_total = E_self + dist
        else:
            E_total = E_self
        return E_total.sum()

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        active: bool = False,
    ) -> Tuple[torch.Tensor, float]:
        if not active:
            return x, 0.0

        step_size = float(self.config.langevin_step_size)
        temperature = float(self.config.langevin_temperature)

        last_energy: Optional[torch.Tensor] = None

        # Allow gradients even during generation-time no_grad sections
        with torch.enable_grad():
            x_curr = x.detach().clone().requires_grad_(True)
            for _ in range(int(self.config.langevin_steps)):
                energy = self.energy_function(x_curr, context)
                grads = torch.autograd.grad(energy, x_curr, retain_graph=False, create_graph=False)[0]
                noise = torch.randn_like(x_curr) * math.sqrt(2.0 * temperature)
                x_next = x_curr - step_size * grads + math.sqrt(step_size) * noise

                # Detach each step to prevent graph blow-up
                x_curr = x_next.detach().requires_grad_(True)
                last_energy = energy

        x_final = self.norm(x_curr.detach())
        delta_E = float(last_energy.item()) if last_energy is not None else 0.0
        return x_final, delta_E


class ZSinkhornHead(nn.Module):
    """
    [Z-OTD] Sinkhorn-inspired output head.
    """
    def __init__(self, config: ZConfig):
        super().__init__()
        self.config = config
        self.vocab_weight = nn.Parameter(torch.randn(config.vocab_size, config.dim))
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,L,D)
        x_norm = F.normalize(x, dim=-1)
        w_norm = F.normalize(self.vocab_weight, dim=-1)
        logits = F.linear(x_norm, w_norm)  # (B,L,V)

        iters = int(self.config.sinkhorn_iters)
        if iters > 0:
            logits = logits / max(float(self.config.sinkhorn_epsilon), 1e-6)
            for _ in range(iters):
                logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        return logits + self.bias


class ZImprintMemory(nn.Module):
    """
    [Logger] Z-Imprint memory.
    Stores pooled hidden states and a scalar energy score.
    """
    def __init__(self, config: ZConfig):
        super().__init__()
        self.config = config
        self.register_buffer("keys", torch.randn(config.memory_slots, config.dim))
        self.register_buffer("values", torch.zeros(config.memory_slots, 1))
        self.ptr: int = 0
        self.full: bool = False

    def write(self, x: torch.Tensor, energy_score: float) -> None:
        x_pooled = x.mean(dim=1)  # (B,D)
        B = int(x_pooled.size(0))
        for i in range(B):
            idx = (self.ptr + i) % int(self.config.memory_slots)
            self.keys[idx] = x_pooled[i].detach()
            self.values[idx, 0] = float(energy_score)
        self.ptr = (self.ptr + B) % int(self.config.memory_slots)
        if self.ptr < B:
            self.full = True

    def read(self, query: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.full and self.ptr == 0:
            return None
        q = query.mean(dim=1)  # (B,D)
        valid_keys = self.keys if self.full else self.keys[: self.ptr]
        attn = torch.matmul(q, valid_keys.T)  # (B,Slots)
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, valid_keys)  # (B,D)
        return context.unsqueeze(1)  # (B,1,D)


# =========================================================
# 4. Transformer Backbone
# =========================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ZConfig):
        super().__init__()
        self.c_attn = nn.Linear(config.dim, 3 * config.dim)
        self.c_proj = nn.Linear(config.dim, config.dim)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.n_head = int(config.heads)
        self.n_embd = int(config.dim)

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_len, config.max_len)).view(1, 1, config.max_len, config.max_len),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.c_proj(y))
        return y, att


class MLP(nn.Module):
    def __init__(self, config: ZConfig):
        super().__init__()
        hidden = int(config.dim * config.mlp_ratio)
        self.c_fc = nn.Linear(config.dim, hidden)
        self.c_proj = nn.Linear(hidden, config.dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.c_proj(self.act(self.c_fc(x))))


class ZBlock(nn.Module):
    def __init__(self, config: ZConfig, layer_id: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.dim)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.dim)
        self.mlp = MLP(config)
        self.lfe = ZLangevinBlock(config)
        self.layer_id = int(layer_id)

    def forward(
        self,
        x: torch.Tensor,
        bloom_active: bool = False,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        attn_out, attn_map = self.attn(self.ln1(x))
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        x_refined, energy_delta = self.lfe(x, context, active=bloom_active)
        return x_refined, attn_map, energy_delta


class ZDotTransformer(nn.Module):
    """
    Ż-aware Transformer: backbone + Z-OTD head + Z-Imprint memory.
    """
    def __init__(self, config: ZConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.position_embedding = nn.Embedding(config.max_len, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([ZBlock(config, i) for i in range(config.depth)])
        self.ln_f = nn.LayerNorm(config.dim)
        self.otd_head = ZSinkhornHead(config)
        self.imprint = ZImprintMemory(config)

    def forward(
        self,
        idx: torch.Tensor,
        bloom_mask: Optional[List[bool]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, T = idx.size()

        # Guard: avoid positional embedding overflow
        if T > int(self.config.max_len):
            idx = idx[:, -int(self.config.max_len):]
            B, T = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.dropout(x)

        context = self.imprint.read(x)

        attn_maps: List[torch.Tensor] = []
        energy_deltas: List[float] = []

        for i, block in enumerate(self.blocks):
            is_active = bool(
                bloom_mask is not None and len(bloom_mask) > 0 and bool(bloom_mask[0]) and (i >= int(self.config.depth) // 2)
            )
            x, att, e_delta = block(x, bloom_active=is_active, context=context)
            attn_maps.append(att)
            energy_deltas.append(float(e_delta))

        x = self.ln_f(x)
        logits = self.otd_head(x)
        return logits, {"attn_maps": attn_maps, "hidden_state": x, "energy_deltas": energy_deltas}


# =========================================================
# 5. Ż Filter / Rε (reachable ops set O(x))
# =========================================================

class ZFilter:
    """
    Simple, implementable Rε candidate family based on truncation.
    In SES terms, each 'mode' is one admissible o ∈ O(x).
    """
    def __init__(self, config: ZConfig):
        self.config = config

    def truncate(self, input_ids: torch.Tensor, keep: int) -> torch.Tensor:
        B, T = input_ids.shape
        keep = int(max(min(keep, T), 1))
        return input_ids[:, -keep:]

    def apply(self, input_ids: torch.Tensor, mode: str = "half") -> torch.Tensor:
        B, T = input_ids.shape
        if T <= 4:
            return input_ids
        if mode == "half":
            return self.truncate(input_ids, max(T // 2, 1))
        if mode == "quarter":
            return self.truncate(input_ids, max(T // 4, 1))
        if mode == "three_quarters":
            return self.truncate(input_ids, max((3 * T) // 4, 1))
        return input_ids  # identity fallback


# =========================================================
# 6. π↑ estimator (SES-like) + collapse detector (SES pressure proxy)
# =========================================================

class PiEstimator:
    """
    SES-like π↑ ensemble with σπ disagreement.
    Uses internal signals (since we don't have gzip/nll here).
    """
    def __init__(self, config: ZConfig):
        self.config = config
        self.cal_out_ent = RollingQuantileCalibrator(config.pi_calib_window, config.pi_q50, config.pi_q95)
        self.cal_attn_ent = RollingQuantileCalibrator(config.pi_calib_window, config.pi_q50, config.pi_q95)
        self.cal_hidden_invvar = RollingQuantileCalibrator(config.pi_calib_window, config.pi_q50, config.pi_q95)
        self.cal_rank_inv = RollingQuantileCalibrator(config.pi_calib_window, config.pi_q50, config.pi_q95)

        self.w = {"out_ent": 0.35, "attn_ent": 0.25, "hidden_invvar": 0.25, "rank_inv": 0.15}

    @torch.no_grad()
    def estimate(self, logits: torch.Tensor, attn_maps: List[torch.Tensor], hidden_state: torch.Tensor) -> Dict[str, float]:
        # Output entropy (last token)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        H_out = float(ZMathKernel.calc_entropy(probs).mean().item())
        H_out_norm = H_out / max(math.log(logits.size(-1) + 1e-8), 1e-6)

        # Attention entropy (last layer, last token over past)
        last_attn = attn_maps[-1]  # (B,H,T,T)
        attn_probs = last_attn[:, :, -1, :]  # (B,H,T)
        H_attn = float(ZMathKernel.calc_entropy(attn_probs, dim=-1).mean().item())
        H_attn_norm = H_attn / max(math.log(attn_probs.size(-1) + 1e-8), 1e-6)

        # Hidden variance collapse proxy: lower variance => more collapsed => higher anomaly => use inverse variance
        var = torch.var(hidden_state, dim=-1).mean(dim=-1)  # (B,)
        invvar = float((1.0 / (var + 1e-4)).mean().item())

        # Attention rank collapse proxy: lower rank => higher anomaly => use inverse rank-like score
        rank_score = ZMathKernel.attention_rank_score(last_attn)
        rank_inv = float(1.0 / (rank_score + 1e-6))

        # Update calibrators
        self.cal_out_ent.update(H_out_norm)
        self.cal_attn_ent.update(H_attn_norm)
        self.cal_hidden_invvar.update(invvar)
        self.cal_rank_inv.update(rank_inv)

        s_out = self.cal_out_ent.score(H_out_norm)
        s_attn = self.cal_attn_ent.score(H_attn_norm)
        s_invvar = self.cal_hidden_invvar.score(invvar)
        s_rank = self.cal_rank_inv.score(rank_inv)

        pi_up = (
            self.w["out_ent"] * s_out
            + self.w["attn_ent"] * s_attn
            + self.w["hidden_invvar"] * s_invvar
            + self.w["rank_inv"] * s_rank
        )
        sigma_pi = float(np.sqrt(np.var([s_out, s_attn, s_invvar, s_rank])))

        return {
            "pi_up": float(np.clip(pi_up, 0.0, 1.0)),
            "sigma_pi": float(np.clip(sigma_pi, 0.0, 1.0)),
            "H_out_norm": float(np.clip(H_out_norm, 0.0, 1.0)),
            "H_attn_norm": float(np.clip(H_attn_norm, 0.0, 1.0)),
            "invvar": invvar,
            "rank_inv": rank_inv,
        }


class CollapseDetector:
    """
    Collapse detector with:
      - π↑, σπ
      - operational pressure proxy ‖P(x)‖ ≈ r(x) (one-step improvability)
    """
    def __init__(self, config: ZConfig):
        self.config = config
        self.window: deque[float] = deque(maxlen=int(config.collapse_history_window))
        self.threshold = float(config.pressure_threshold_init)
        self.cooldown = 0
        self.last_pressure: float = 0.0

    def step_cooldown(self) -> None:
        if self.cooldown > 0:
            self.cooldown -= 1

    def arm_cooldown(self) -> None:
        self.cooldown = int(self.config.collapse_cooldown_steps)

    def monitor(self, pi_up: float, sigma_pi: float, pressure_norm: float) -> bool:
        pi_up = float(pi_up)
        sigma_pi = float(sigma_pi)
        pressure_norm = float(pressure_norm)

        score = 0.55 * pressure_norm + 0.35 * pi_up + 0.10 * sigma_pi
        self.window.append(score)
        self.last_pressure = pressure_norm

        if self.cooldown > 0 or len(self.window) < 12:
            return False

        avg = float(np.mean(self.window))
        std = float(np.std(self.window) + 1e-8)
        is_tail = score > avg + 2.5 * std
        is_high = (pressure_norm > self.threshold) and (pi_up > float(self.config.epsilon_definable))
        return bool(is_tail and is_high)

    def get_status(self) -> str:
        return f"||P||={self.last_pressure:.3f} (Th={self.threshold:.3f})"


# =========================================================
# 7. Z-CP: Central Processor (π–P–R loop)
# =========================================================

class ZCentralProcessor:
    """
    Input -> Pattern (π↑) -> Pressure (‖P‖=r(x)) -> (Collapse?) -> Rε (Filter) -> Bloom -> Output
    """
    def __init__(self, model: ZDotTransformer, config: ZConfig, tokenizer: Any):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.pi_est = PiEstimator(config)
        self.detector = CollapseDetector(config)
        self.filter = ZFilter(config)

        self.pi_history: List[float] = []
        self.pressure_history: List[float] = []

    @torch.no_grad()
    def _forward_and_pi(self, input_ids: torch.Tensor, bloom: bool) -> Tuple[torch.Tensor, Dict[str, Any], Dict[str, float]]:
        logits, info = self.model(input_ids, bloom_mask=[bool(bloom)])
        pi_info = self.pi_est.estimate(logits, info["attn_maps"], info["hidden_state"])
        return logits, info, pi_info

    @torch.no_grad()
    def _operational_pressure(
        self,
        input_ids: torch.Tensor,
        pi_current: float,
        ops: Optional[List[str]] = None,
    ) -> Tuple[float, str]:
        """
        SES operational pressure proxy:
          r(x)=max_{o∈O(x)} (π↑(x) − π↑(o(x)))_+.
        Here O(x) is a small truncation set for speed.
        """
        if ops is None:
            ops = ["identity", "half", "quarter", "three_quarters"]

        best_drop = 0.0
        best_op = "identity"

        for op in ops:
            cand = input_ids if op == "identity" else self.filter.apply(input_ids, mode=op)
            _, _, pi_info = self._forward_and_pi(cand, bloom=False)
            pi_cand = float(pi_info["pi_up"])
            drop = max(float(pi_current) - pi_cand, 0.0)
            if drop > best_drop:
                best_drop = drop
                best_op = op

        return float(best_drop), best_op

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        verbose: bool = True,
    ) -> str:
        if verbose:
            print(f"--- Ż-CP Initiated: {prompt!r} ---")

        self.model.eval()
        device = next(self.model.parameters()).device

        input_ids = torch.tensor([self.tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated: List[int] = []
        collapse_events = 0

        self.pi_history.clear()
        self.pressure_history.clear()

        for step in range(int(max_new_tokens)):
            # Guard: avoid positional embedding overflow
            if input_ids.size(1) > int(self.config.max_len):
                input_ids = input_ids[:, -int(self.config.max_len):]

            # Pattern phase (no Bloom)
            logits, info, pi_info = self._forward_and_pi(input_ids, bloom=False)
            pi_up = float(pi_info["pi_up"])
            sigma_pi = float(pi_info["sigma_pi"])
            self.pi_history.append(pi_up)

            # For speed: only search ops near the frontier (π↑ close to / above ε)
            if pi_up < (float(self.config.epsilon_definable) * 0.85):
                p_norm, best_op = 0.0, "identity"
            else:
                p_norm, best_op = self._operational_pressure(input_ids, pi_up, ops=["identity", "half", "quarter"])
            self.pressure_history.append(p_norm)

            # Collapse decision
            is_collapse = self.detector.monitor(pi_up=pi_up, sigma_pi=sigma_pi, pressure_norm=p_norm)

            if is_collapse:
                collapse_events += 1
                self.detector.arm_cooldown()

                if verbose:
                    print(f" [!] Ż-Collapse Detected @ step {step}. {self.detector.get_status()} π↑={pi_up:.3f} σπ={sigma_pi:.3f}")
                    print(f" [⟂] Rε chosen: {best_op!r} (best immediate π↓={p_norm:.3f})")
                    print(" [⟳] Bloom phase (Z-LFE) triggered.")

                # Apply Rε
                if best_op != "identity":
                    input_ids = self.filter.apply(input_ids, mode=best_op)

                # Bloom pass
                logits, info, pi_info_bloom = self._forward_and_pi(input_ids, bloom=True)
                pi_after = float(pi_info_bloom["pi_up"])
                sigma_after = float(pi_info_bloom["sigma_pi"])
                self.pi_history.append(pi_after)

                deltas = info["energy_deltas"]
                avg_delta = float(np.mean(deltas)) if deltas else 0.0

                if verbose:
                    print(f" [*] Bloom Reconstruction: energy Δ={avg_delta:.4f}")
                    print(f" [*] π↑(after Bloom)={pi_after:.3f} σπ={sigma_after:.3f}")

                # Imprint
                self.model.imprint.write(info["hidden_state"], avg_delta)

            # Sample next token
            next_token_logits = logits[:, -1, :] / max(float(temperature), 1e-6)
            probs = F.softmax(next_token_logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, idx_next), dim=1)
            token_id = int(idx_next.item())
            generated.append(token_id)

            if hasattr(self.tokenizer, "eos_token_id") and token_id == int(self.tokenizer.eos_token_id):
                break

            self.detector.step_cooldown()

        decoded = self.tokenizer.decode(generated)
        if verbose:
            print(f"--- Generation Complete. Collapse events: {collapse_events} ---")
        return decoded


# =========================================================
# 8. Minimal runnable demo tokenizer
# =========================================================

class SimpleMockTokenizer:
    """
    Minimal tokenizer so the script can run standalone.
    For real use, swap with a real tokenizer (e.g., tiktoken / HF tokenizer).
    """
    def __init__(self):
        self.vocab_size = 128
        self.eos_token_id = 0

    def encode(self, text: str) -> List[int]:
        ids = [(ord(c) % self.vocab_size) for c in text]
        return ids if ids else [self.eos_token_id]

    def decode(self, ids: List[int]) -> str:
        # printable-ish
        return "".join([chr((i % 95) + 32) for i in ids if i != self.eos_token_id])


# =========================================================
# 9. Optional: Collapse curve utility (demo only)
# =========================================================

def z_collapse_curve(
    c: np.ndarray,
    A0: float = 1.0,
    w: float = 0.6,
    tau1: float = 6.0,
    sigma1: float = 3.0,
    tau2: float = 12.0,
    sigma2: float = 1.0,
) -> np.ndarray:
    s1 = 1.0 / (1.0 + np.exp((c - tau1) / sigma1))
    s2 = 1.0 / (1.0 + np.exp((c - tau2) / sigma2))
    return A0 * ((1.0 - w) * s1 + w * s2)


# =========================================================
# 10. Demo / test_run
# =========================================================

def test_run() -> None:
    conf = ZConfig(
        vocab_size=128,
        dim=32,
        depth=2,
        heads=2,
        max_len=128,
        pressure_threshold_init=0.25,  # lower for demo visibility
        epsilon_definable=0.55,
        pi_calib_window=64,
    )

    print("Initializing Ż python transformer (SES-aligned) ...")
    tokenizer = SimpleMockTokenizer()
    model = ZDotTransformer(conf)
    cp = ZCentralProcessor(model, conf, tokenizer)

    prompt = "Z-theory is the architecture of "
    print(f"\nPrompt: {prompt}")
    out = cp.generate(prompt, max_new_tokens=6, temperature=1.0, verbose=True)
    print(f"\nOutput: {prompt + out}")

    # Optional plot (safe to skip if matplotlib missing)
    try:
        import matplotlib.pyplot as plt
        c = np.linspace(0, 25, 400)
        A_baseline = z_collapse_curve(c, w=0.7, tau1=6.0, sigma1=2.5, tau2=10.0, sigma2=0.7)
        A_z = z_collapse_curve(c, w=0.7, tau1=8.0, sigma1=3.5, tau2=15.0, sigma2=1.5)
        plt.figure(figsize=(7, 4.5))
        plt.plot(c, A_baseline, label="Baseline (collapse)")
        plt.plot(c, A_z, label="Ż-structured (mitigated)")
        plt.title("Ż Collapse Curve (demo)")
        plt.xlabel("Complexity (rel. units)")
        plt.ylabel("Accuracy (0–1)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[Collapse Curve] Skipping plot: {e}")


if __name__ == "__main__":
    test_run()
