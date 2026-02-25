#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
z_dot_cp_overlay_hf_SES_v2_3_1_strict.py
=======================================

"Production-use" Ż controller overlay that **strictly follows SES v2.3.1 operational definitions**
while keeping a pretrained HF Transformer (host LLM) unchanged.

What "strict" means here
------------------------
π↑(x) is computed as an ensemble of monotone metrics calibrated by *quantiles* (q50, q95)
on a calibration corpus, as in SES v2.3.1:

  s_m(x) = clip( (κ_m(x)-q50_m)/(q95_m-q50_m+1e-8), 0, 1 )
  π↑(x)  = Σ_m w_m s_m(x)
  σπ(x)  = sqrt( Var_m[s_m(x)] )

Operational pressure is defined by one-step improvability over admissible operations O(x):

  r(x)   = max_{o∈O(x)} ( π↑(x) - π↑(o(x)) )_+
  ||P||  := r(x)

Reconstruction/projection is an implementable Rε approximated via discrete search:

  Rε(x) = argmin_{y∈Fε(x)} [ π↑(y) + λ c(x,y) ]
  where Fε(x) are reachable candidates with π↑(y) ≤ ε.

Operational control laws (SES v2.3.1):
  - If π↑(x) > ε  AND  ||P|| > τP: apply Rε (reconstruction-effective regime)
  - If π↑(x) > τπ AND  ||P|| ≤ τP: invoke μ-redesign and/or reset (collapse-precursor)

This script implements μ-redesign as a "mode switch" (temperature/top-p down + constraint),
and Bloom as a short self-consistency sampling then selecting the continuation that yields
the lowest π↑.

Dependencies
------------
  - torch
  - transformers
  - numpy

Usage
-----
  python z_dot_cp_overlay_hf_SES_v2_3_1_strict.py \
    --model gpt2 \
    --prompt "Żとは？" \
    --max_new_tokens 200 \
    --calib_file calib.txt \
    --verbose

Calibration file format
-----------------------
A plain text file where each line is a calibration example text.
(Blank lines are ignored.)

If --calib_file is omitted, a small built-in calibration set is used (works, but for real
production you should pass your own dev corpus for stable q50/q95).

"""

from __future__ import annotations

import argparse
import gzip
import math
import os
import random
import difflib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as e:
    raise SystemExit(
        "ERROR: 'transformers' is required.\n"
        "Install: pip install transformers\n\n"
        f"Original import error: {e}"
    )


# -------------------------
# Config
# -------------------------

@dataclass
class SESConfig:
    # π / definability
    epsilon_definable: float = 0.70  # ε in Dε={x|π↑≤ε}
    tau_pi: float = 0.85             # τπ for collapse-precursor regime

    # pressure / control
    tau_P: float = 0.18              # τP
    lambda_cost: float = 0.15        # λ in objective π↑ + λ c(x,y)
    alpha_edit: float = 1.0          # α for c(x,y)
    beta_len: float = 0.25           # β for c(x,y)

    # quantile calibration
    q50: float = 0.50
    q95: float = 0.95

    # metric weights (sum to 1)
    w_gz: float = 0.34
    w_nll: float = 0.33
    w_ent: float = 0.33

    # generation defaults
    temperature: float = 0.9
    top_p: float = 0.95

    # μ-redesign (safe mode)
    safe_temperature: float = 0.55
    safe_top_p: float = 0.85

    # bloom
    bloom_samples: int = 3
    bloom_tokens: int = 48
    bloom_temperature: float = 0.85
    bloom_top_p: float = 0.95

    # operational search settings
    pressure_gate: float = 0.85      # only search O(x) if π↑ ≥ ε*gate
    kmax: int = 1                    # reachability depth (we implement Kmax=1 for speed)

    # operations O(x)
    constraint_text: str = "\n\n[Ż-Rε] Keep assumptions explicit. Use minimal contradictions."
    decompose_text: str = "\n\n[Ż-Rε] Decompose the task into 3–5 sub-steps before answering."
    normalize_whitespace: bool = True
    tail_fracs: Tuple[float, ...] = (1.0, 0.75, 0.50, 0.25)

    # logging / reproducibility
    seed: int = 0


DEFAULT_CALIB_TEXTS: List[str] = [
    "Explain the difference between correlation and causation.",
    "Summarize the plot of a well-known fairy tale in 3 bullet points.",
    "Compute 17*23 and show your steps.",
    "Write a short email asking to reschedule a meeting.",
    "Translate 'Good morning' into Japanese and French.",
    "List pros and cons of remote work.",
    "Define entropy in information theory, briefly.",
]


# -------------------------
# SES π metrics
# -------------------------

def gzip_ratio(text: str) -> float:
    b = text.encode("utf-8", errors="ignore")
    if len(b) == 0:
        return 0.0
    z = gzip.compress(b)
    return float(len(z)) / float(len(b))  # higher => less compressible => more complex


def token_empirical_entropy(token_ids: List[int]) -> float:
    """
    Empirical token entropy of the sequence itself (not the model distribution).
    Normalized by log(K) where K is #unique tokens.
    """
    if not token_ids:
        return 0.0
    vals, counts = np.unique(np.array(token_ids, dtype=np.int64), return_counts=True)
    p = counts.astype(np.float64) / float(np.sum(counts))
    H = float(-(p * np.log(p + 1e-12)).sum())
    K = max(int(len(vals)), 2)
    return float(np.clip(H / math.log(K + 1e-12), 0.0, 1.0))


def normalized_next_token_entropy(logits_last: torch.Tensor) -> float:
    """
    Normalized entropy of the model's next-token distribution (cheap during generation).
    """
    probs = F.softmax(logits_last, dim=-1)
    H = float(-(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item())
    V = logits_last.size(-1)
    return float(np.clip(H / max(math.log(V + 1e-12), 1e-6), 0.0, 1.0))


# -------------------------
# Quantile calibration (dev-set)
# -------------------------

class QuantileCalib:
    def __init__(self, q50: float, q95: float):
        self.q50 = float(q50)
        self.q95 = float(q95)
        self.q50_v: float = 0.0
        self.q95_v: float = 1.0

    def fit(self, values: List[float]) -> None:
        if len(values) < 8:
            self.q50_v, self.q95_v = 0.0, 1.0
            return
        arr = np.array(values, dtype=np.float64)
        self.q50_v = float(np.quantile(arr, self.q50))
        self.q95_v = float(np.quantile(arr, self.q95))

    def score01(self, v: float) -> float:
        denom = (self.q95_v - self.q50_v) + 1e-8
        s = (float(v) - self.q50_v) / denom
        return float(np.clip(s, 0.0, 1.0))


class PiSES:
    """
    SES v2.3.1 scalar π↑ (danger score) with quantile-calibrated ensemble:
      - gz ratio κ_gz
      - mean NLL κ_nll (from teacher forcing / incremental update)
      - entropy κ_ent (we use next-token distribution entropy during generation)
    """
    def __init__(self, cfg: SESConfig):
        self.cfg = cfg
        self.cal_gz = QuantileCalib(cfg.q50, cfg.q95)
        self.cal_nll = QuantileCalib(cfg.q50, cfg.q95)
        self.cal_ent = QuantileCalib(cfg.q50, cfg.q95)

        # running NLL (incremental during generation)
        self.nll_sum = 0.0
        self.nll_count = 0

    def reset_nll(self) -> None:
        self.nll_sum = 0.0
        self.nll_count = 0

    def update_nll_from_step(self, logits_last: torch.Tensor, next_token_id: int) -> None:
        # logits_last: (1,V)
        logp = F.log_softmax(logits_last, dim=-1)[0, int(next_token_id)].item()
        nll = float(-logp)
        self.nll_sum += nll
        self.nll_count += 1

    def mean_nll(self) -> float:
        if self.nll_count <= 0:
            return 0.0
        return float(self.nll_sum / float(self.nll_count))

    def calibrate(
        self,
        model: Any,
        tokenizer: Any,
        calib_texts: List[str],
        device: torch.device,
        max_len: int = 512
    ) -> None:
        """
        Fit q50/q95 for each κ_m on a dev corpus.
        We compute:
          κ_gz(text)
          κ_nll(text) via teacher forcing
          κ_ent(text) via average next-token distribution entropy (teacher forcing)
        """
        gz_vals: List[float] = []
        nll_vals: List[float] = []
        ent_vals: List[float] = []

        model.eval()
        for text in calib_texts:
            t = (text or "").strip()
            if not t:
                continue

            # κ_gz
            gz_vals.append(gzip_ratio(t))

            # tokenize
            enc = tokenizer(t, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            if input_ids.shape[1] < 2:
                nll_vals.append(0.0)
                ent_vals.append(0.0)
                continue

            # truncate for calibration speed
            if input_ids.shape[1] > max_len:
                input_ids = input_ids[:, -max_len:]

            # teacher forcing
            with torch.no_grad():
                out = model(input_ids=input_ids, use_cache=False, return_dict=True)
                logits = out.logits  # (1,T,V)

            # next-token NLL + entropy
            # predict token t_i from prefix up to i-1 => use logits[:, i-1]
            V = logits.size(-1)
            nll_sum = 0.0
            ent_sum = 0.0
            cnt = 0
            for i in range(1, input_ids.shape[1]):
                logits_last = logits[:, i-1, :]  # (1,V)
                tok = int(input_ids[0, i].item())
                logp = F.log_softmax(logits_last, dim=-1)[0, tok].item()
                nll_sum += float(-logp)
                ent_sum += normalized_next_token_entropy(logits_last)
                cnt += 1
            nll_vals.append(nll_sum / max(cnt, 1))
            ent_vals.append(ent_sum / max(cnt, 1))

        self.cal_gz.fit(gz_vals)
        self.cal_nll.fit(nll_vals)
        self.cal_ent.fit(ent_vals)

    def compute(self, gz: float, nll: float, ent: float) -> Dict[str, float]:
        s_gz = self.cal_gz.score01(gz)
        s_nll = self.cal_nll.score01(nll)
        s_ent = self.cal_ent.score01(ent)

        # π↑ scalar danger score
        pi_up = (
            self.cfg.w_gz * s_gz
            + self.cfg.w_nll * s_nll
            + self.cfg.w_ent * s_ent
        )
        sigma_pi = float(np.sqrt(np.var([s_gz, s_nll, s_ent])))

        return {
            "pi_up": float(np.clip(pi_up, 0.0, 1.0)),
            "sigma_pi": float(np.clip(sigma_pi, 0.0, 1.0)),
            "s_gz": float(s_gz),
            "s_nll": float(s_nll),
            "s_ent": float(s_ent),
        }


# -------------------------
# Cost c(x,y) and operations O(x)
# -------------------------

def distortion_cost(cfg: SESConfig, x: str, y: str) -> float:
    """
    c(x,y)=α edit(x,y)+β |len(x)-len(y)|.
    We approximate edit with difflib ratio (0..1): edit≈1-ratio
    """
    x = x or ""
    y = y or ""
    ratio = difflib.SequenceMatcher(a=x, b=y).ratio()
    edit = 1.0 - float(ratio)
    len_pen = abs(len(x) - len(y)) / max(len(x), 1)
    return float(cfg.alpha_edit * edit + cfg.beta_len * len_pen)


def normalize_text(t: str) -> str:
    # conservative normalization
    return " ".join((t or "").split())


def tail_keep_text(tokenizer: Any, text: str, frac: float, max_tokens: int) -> str:
    enc = tokenizer(text, add_special_tokens=False)
    ids = enc["input_ids"]
    if not ids:
        return text
    keep = max(1, int(len(ids) * float(frac)))
    ids2 = ids[-keep:]
    if len(ids2) > max_tokens:
        ids2 = ids2[-max_tokens:]
    return tokenizer.decode(ids2, skip_special_tokens=True)


# -------------------------
# Controller (SES control laws)
# -------------------------

class ZDotSESController:
    def __init__(self, model: Any, tokenizer: Any, cfg: SESConfig, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.pi = PiSES(cfg)

        self.events: List[Dict[str, float]] = []
        self.pi_hist: List[float] = []
        self.p_hist: List[float] = []

        # soft mode state
        self.safe_mode = False

    def _encode(self, text: str) -> torch.Tensor:
        return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(self.device)

    def _decode(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids[0].tolist(), skip_special_tokens=True)

    @torch.no_grad()
    def _pi_for_text(self, text: str, logits_last: Optional[torch.Tensor], ent_override: Optional[float] = None) -> Dict[str, float]:
        """
        Compute π↑ for current text using:
          gz(text)
          nll = running mean NLL (already tracked during generation)
          ent = next-token entropy (from logits_last) or override
        """
        gz = gzip_ratio(text)
        nll = self.pi.mean_nll()
        if ent_override is not None:
            ent = float(ent_override)
        elif logits_last is not None:
            ent = normalized_next_token_entropy(logits_last)
        else:
            ent = 0.0
        return self.pi.compute(gz=gz, nll=nll, ent=ent)

    @torch.no_grad()
    def _search_ops(self, x_text: str, pi_x: float, max_pos_tokens: int) -> Tuple[float, str, str, float]:
        """
        Compute operational pressure:
          r(x)=max_o (π(x)-π(o(x)))_+.
        Also return the best candidate y and objective score for Rε.
        """
        if pi_x < (self.cfg.epsilon_definable * self.cfg.pressure_gate):
            # skip search when safely below boundary
            return 0.0, "identity", x_text, pi_x + self.cfg.lambda_cost * 0.0

        # Build 1-step candidates (Reach_{Kmax} with Kmax=1).
        cand: List[Tuple[str, str]] = [("identity", x_text)]

        t0 = x_text
        if self.cfg.normalize_whitespace:
            cand.append(("normalize", normalize_text(t0)))

        # Tail keeps (token-based projection)
        for frac in self.cfg.tail_fracs:
            cand.append((f"tail_keep_{frac:.2f}", tail_keep_text(self.tokenizer, t0, frac, max_pos_tokens)))

        # constraint insertion / decomposition
        cand.append(("add_constraint", t0 + self.cfg.constraint_text))
        cand.append(("add_decompose", t0 + self.cfg.decompose_text))

        # Evaluate π↑ for each candidate
        best_drop = 0.0
        best_op = "identity"
        best_y = x_text

        # Rε selection (approx): minimize π↑(y)+λ c(x,y) subject to π↑(y)≤ε; fallback to best objective.
        best_obj = 1e9
        best_obj_y = x_text
        best_obj_pi = pi_x
        best_obj_op = "identity"
        found_feasible = False

        for op, y in cand:
            y = (y or "").strip()
            pi_info = self.pi.compute(gz=gzip_ratio(y), nll=self.pi.mean_nll(), ent=0.0)
            # We set ent=0.0 here because this op-search is text-level; in practice you can
            # also run a cheap forward to estimate ent. Keeping it deterministic + cheap.
            pi_y = float(pi_info["pi_up"])

            drop = max(pi_x - pi_y, 0.0)
            if drop > best_drop:
                best_drop, best_op, best_y = drop, op, y

            obj = pi_y + self.cfg.lambda_cost * distortion_cost(self.cfg, x_text, y)
            if (pi_y <= self.cfg.epsilon_definable) and (obj < best_obj):
                best_obj = obj
                best_obj_y = y
                best_obj_pi = pi_y
                best_obj_op = op
                found_feasible = True
            elif not found_feasible and obj < best_obj:
                best_obj = obj
                best_obj_y = y
                best_obj_pi = pi_y
                best_obj_op = op

        # operational pressure norm ||P|| := r(x)
        p_norm = float(best_drop)

        return p_norm, best_obj_op, best_obj_y, float(best_obj)

    def _sample_next(self, logits_last: torch.Tensor, temperature: float, top_p: float) -> int:
        logits = logits_last / max(float(temperature), 1e-6)
        probs = F.softmax(logits, dim=-1)

        if 0.0 < float(top_p) < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cum = torch.cumsum(sorted_probs, dim=-1)
            mask = cum > float(top_p)
            mask[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(mask, 0.0)
            sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)
            idx = torch.multinomial(sorted_probs, num_samples=1)
            token = int(sorted_idx.gather(-1, idx).item())
            return token

        return int(torch.multinomial(probs, num_samples=1).item())

    @torch.no_grad()
    def _bloom(self, base_text: str, max_pos_tokens: int, verbose: bool) -> str:
        """
        Bloom: sample K continuations; pick the one with minimal π↑ afterwards.
        """
        K = max(1, int(self.cfg.bloom_samples))
        best_pi = 1e9
        best_text = ""

        base_ids = self._encode(base_text)
        # enforce max pos
        if base_ids.shape[1] > max_pos_tokens:
            base_ids = base_ids[:, -max_pos_tokens:]

        gen_kwargs = dict(
            do_sample=True,
            temperature=float(self.cfg.bloom_temperature),
            top_p=float(self.cfg.bloom_top_p),
            max_new_tokens=int(self.cfg.bloom_tokens),
            num_return_sequences=1,
            use_cache=True,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
            eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
        )

        for i in range(K):
            out_ids = self.model.generate(input_ids=base_ids, **gen_kwargs)
            cont_ids = out_ids[:, base_ids.shape[1]:]
            cont_text = self.tokenizer.decode(cont_ids[0].tolist(), skip_special_tokens=True)

            # score π↑ after bloom
            y = (base_text + cont_text).strip()
            pi_y = float(self.pi.compute(gz=gzip_ratio(y), nll=self.pi.mean_nll(), ent=0.0)["pi_up"])
            if verbose:
                print(f"  [Bloom {i+1}/{K}] π↑={pi_y:.3f}")
            if pi_y < best_pi:
                best_pi = pi_y
                best_text = cont_text

        if verbose:
            print(f"  [Bloom] selected π↑={best_pi:.3f}")
        return best_text

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int, verbose: bool = True) -> str:
        self.model.eval()
        self.safe_mode = False
        self.events.clear()
        self.pi_hist.clear()
        self.p_hist.clear()

        # reset incremental NLL
        self.pi.reset_nll()

        # model max pos
        try:
            max_pos = int(getattr(self.model.config, "max_position_embeddings", 2048))
        except Exception:
            max_pos = 2048

        # start
        text = (prompt or "").strip()
        input_ids = self._encode(text)
        if input_ids.shape[1] > max_pos:
            input_ids = input_ids[:, -max_pos:]
            text = self._decode(input_ids)

        past = None
        gen_tokens: List[int] = []

        for t in range(int(max_new_tokens)):
            # forward (cached)
            step_in = input_ids if past is None else input_ids[:, -1:]
            out = self.model(
                input_ids=step_in,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = getattr(out, "past_key_values", None)
            logits_last = out.logits[:, -1, :]  # (1,V)

            # compute ent for this step
            ent = normalized_next_token_entropy(logits_last)

            # compute π↑ for current text with running NLL + gz + ent(step)
            pi_info = self._pi_for_text(text, logits_last=logits_last, ent_override=ent)
            pi_x = float(pi_info["pi_up"])
            sigma_pi = float(pi_info["sigma_pi"])
            self.pi_hist.append(pi_x)

            # operational pressure search (text-level; ||P|| := r(x))
            p_norm, op_R, y_text, obj = self._search_ops(text, pi_x, max_pos_tokens=max_pos)
            self.p_hist.append(p_norm)

            # SES control laws
            if (pi_x > self.cfg.epsilon_definable) and (p_norm > self.cfg.tau_P):
                # reconstruction-effective: apply Rε
                if verbose:
                    print(f"[Ż][Rε] step={t} π↑={pi_x:.3f} σπ={sigma_pi:.3f} ||P||={p_norm:.3f} op={op_R} obj={obj:.3f}")

                text = y_text.strip()
                input_ids = self._encode(text)
                if input_ids.shape[1] > max_pos:
                    input_ids = input_ids[:, -max_pos:]
                    text = self._decode(input_ids)
                past = None

                # Bloom (optional, but "production" keeps it)
                bloom_add = self._bloom(text, max_pos_tokens=max_pos, verbose=verbose)
                if bloom_add:
                    text = (text + bloom_add).strip()
                    input_ids = self._encode(text)
                    if input_ids.shape[1] > max_pos:
                        input_ids = input_ids[:, -max_pos:]
                        text = self._decode(input_ids)
                    past = None

                # reset running NLL after restructure (idempotence-like stabilization)
                self.pi.reset_nll()

                self.events.append({"step": float(t), "pi": pi_x, "sigma": sigma_pi, "p": p_norm})
                continue

            if (pi_x > self.cfg.tau_pi) and (p_norm <= self.cfg.tau_P):
                # collapse-precursor: μ-redesign / reset
                if verbose:
                    print(f"[Ż][μ] step={t} π↑={pi_x:.3f} σπ={sigma_pi:.3f} ||P||={p_norm:.3f} -> safe_mode on")
                self.safe_mode = True

            # sample next token
            temp = self.cfg.safe_temperature if self.safe_mode else self.cfg.temperature
            top_p = self.cfg.safe_top_p if self.safe_mode else self.cfg.top_p
            tok = self._sample_next(logits_last, temperature=temp, top_p=top_p)

            # update incremental NLL using chosen token
            self.pi.update_nll_from_step(logits_last, tok)

            gen_tokens.append(tok)

            # append
            next_id = torch.tensor([[tok]], dtype=torch.long, device=self.device)
            input_ids = torch.cat([input_ids, next_id], dim=1)

            # update text occasionally (decode can be expensive; here we do it every step for correctness)
            # For speed, you can switch to periodic decode.
            text = self._decode(input_ids)

            # keep within max_pos
            if input_ids.shape[1] > max_pos:
                input_ids = input_ids[:, -max_pos:]
                text = self._decode(input_ids)
                past = None

            eos = getattr(self.tokenizer, "eos_token_id", None)
            if eos is not None and int(tok) == int(eos):
                break

        # Return only the continuation (like typical generate)
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True)


# -------------------------
# Helpers / CLI
# -------------------------

def pick_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def read_calib_file(path: str) -> List[str]:
    texts: List[str] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            t = line.strip()
            if t:
                texts.append(t)
    return texts


def set_seed(seed: int) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt2", help="HF model id or local path")
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--device", type=str, default="auto", help="auto/cpu/cuda/mps")
    ap.add_argument("--verbose", action="store_true")

    # calibration
    ap.add_argument("--calib_file", type=str, default="", help="path to calibration text file (one text per line)")
    ap.add_argument("--calib_max_len", type=int, default=512)

    # SES knobs
    ap.add_argument("--epsilon", type=float, default=0.70)
    ap.add_argument("--tauP", type=float, default=0.18)
    ap.add_argument("--taupi", type=float, default=0.85)
    ap.add_argument("--lambda_cost", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)

    # bloom knobs
    ap.add_argument("--bloom_samples", type=int, default=3)
    ap.add_argument("--bloom_tokens", type=int, default=48)

    args = ap.parse_args()

    cfg = SESConfig(
        epsilon_definable=float(args.epsilon),
        tau_P=float(args.tauP),
        tau_pi=float(args.taupi),
        lambda_cost=float(args.lambda_cost),
        bloom_samples=int(args.bloom_samples),
        bloom_tokens=int(args.bloom_tokens),
        seed=int(args.seed),
    )

    set_seed(cfg.seed)
    device = pick_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    ctrl = ZDotSESController(model=model, tokenizer=tokenizer, cfg=cfg, device=device)

    # calibration
    calib_texts = DEFAULT_CALIB_TEXTS
    if args.calib_file:
        calib_texts = read_calib_file(args.calib_file)
    ctrl.pi.calibrate(model=model, tokenizer=tokenizer, calib_texts=calib_texts, device=device, max_len=int(args.calib_max_len))

    out = ctrl.generate(args.prompt, max_new_tokens=int(args.max_new_tokens), verbose=bool(args.verbose))

    print("\n=== OUTPUT ===")
    print(args.prompt + out)


if __name__ == "__main__":
    main()
