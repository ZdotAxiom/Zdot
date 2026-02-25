# z_collapse_lab.py
# ============================================================
# Ż Collapse Lab (SES v2.4.2 — Single-File “All-in-One” Edition, patched)
# Dependencies:
#   - required: numpy, pandas, matplotlib
#   - optional: torch, scipy, transformers
#
# Patch notes vs v2.4.1 (high-level):
#   - Runtime config resolution via resolve_runtime_cfg (no SESConfig mutation; optional NLL auto-disable)
#   - two_phase_logistic uses the *increasing* logistic (typical for "rate increases with complexity")
#   - Headless-safe matplotlib backend selection (uses Agg if DISPLAY missing on non-Windows)
#   - Adds optional CLI flags --no_text for run_text/run_lm to keep logs lightweight
#   - compute_collapse_curves includes 'precursor_rate' alias (keeps 'collapse_only_rate' for compat)
# ============================================================

from __future__ import annotations

import argparse
import copy
import dataclasses
import gzip
import hashlib
import io
import json
import math
import os
import random
import re
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---- matplotlib backend (headless-safe) ----
# Must be decided *before* importing pyplot.
try:
    import matplotlib  # type: ignore
    _HEADLESS = (os.name != "nt") and (os.environ.get("DISPLAY", "") == "") and (os.environ.get("WAYLAND_DISPLAY", "") == "")
    if _HEADLESS and os.environ.get("MPLBACKEND", "") == "":
        matplotlib.use("Agg")  # type: ignore
except Exception:
    # If matplotlib import fails here, we'll fail later when pyplot is imported anyway.
    pass

import matplotlib.pyplot as plt  # noqa: E402

# Optional SciPy
try:
    from scipy.optimize import curve_fit  # type: ignore
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

# Optional Torch
try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ============================================================
# 0. Configuration (SES v2.4.2 operational knobs)
# ============================================================

@dataclass
class SESConfig:
    # --- π↑ calibration ---
    use_gz: bool = True
    use_ent: bool = True
    use_nll: bool = False  # requires a local causal LM + tokenizer

    # Metric weights (will be normalized over enabled metrics)
    w_gz: float = 0.34
    w_ent: float = 0.33
    w_nll: float = 0.33

    # Percentiles computed on calibration corpus C_dev:
    # stored as dict metric -> {q05,q50,q95}
    calib: Dict[str, Dict[str, float]] = dataclasses.field(default_factory=dict)

    # --- Entropy normalization knob ---
    ent_H_max: float = 8.0

    # --- gzip overhead correction ---
    gz_overhead_correct: bool = True
    gz_overhead_min_raw_bytes: int = 64  # disable overhead correction for shorter texts

    # --- D_ε threshold ---
    epsilon: float = 0.35  # D_ε := {x | π↑ <= ε}

    # --- Control thresholds ---
    tau_pi: float = 0.65   # collapse-precursor if π↑ > τπ and improvability <= τP
    tau_P: float = 0.05    # τP compares to 1-step improvability r(x) (not a true norm)

    # --- R_ε search ---
    K_max: int = 3
    beam_width: int = 6
    lambda_cost: float = 0.20  # λ in π(y) + λ c(x,y)

    # R_epsilon dedup + caps
    repsilon_dedup: bool = True
    repsilon_dedup_across_depths: bool = True
    repsilon_max_unique_per_depth: int = 2000
    repsilon_obj_improve_eps: float = 1e-9
    repsilon_keep_best_only: bool = True

    # Distortion cost coefficients
    alpha_edit: float = 1.0
    beta_len: float = 0.002

    # Optional fast distortion approximation
    distortion_fast: bool = False
    distortion_fast_len_only: bool = False  # if True: ignore similarity term entirely

    # --- Distortion similarity speed knob ---
    edit_cost_sample_window: int = 1200
    edit_cost_long_threshold: int = 2500

    # --- Complexity proxy (for collapse curve x-axis) ---
    complexity_mix_gz: float = 0.5
    complexity_mix_ent: float = 0.5

    # --- Logging / runtime ---
    seed: int = 0
    max_steps: int = 999999
    track_text: bool = True

    # --- LM generation (optional) ---
    temperature: float = 1.0
    max_new_tokens: int = 64

    # --- Safety: avoid exploding states ---
    max_state_chars: int = 20000

    # --- NLL compute cap (optional) ---
    nll_max_tokens: int = 1024

    # --- Cache size (per run / per prompt) ---
    pi_cache_max_items: int = 16384

    # --- Hash hardening ---
    hash_digest_bytes: int = 32

    # --- Tokenizer fallback safety (hard caps) ---
    tokenizer_fallback_char_cap: int = 4096


def warn(msg: str) -> None:
    print(msg, file=sys.stderr)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# ============================================================
# 0.4 Device helpers
# ============================================================

def normalize_device(requested: str) -> str:
    req = str(requested or "cpu").strip().lower()
    if not TORCH_AVAILABLE:
        return "cpu"
    if req.startswith("cuda"):
        if torch.cuda.is_available():
            return req
        return "cpu"
    if req in ("mps",):
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # type: ignore
                return "mps"
        except Exception:
            pass
        return "cpu"
    return "cpu" if req == "" else req


# ============================================================
# 0.5 Helpers: context length + truncation
# ============================================================

def _finite(x: Any, default: float = 0.0) -> float:
    try:
        xf = float(x)
    except Exception:
        return float(default)
    return xf if np.isfinite(xf) else float(default)


def _safe_truncate(text: str, max_chars: int) -> str:
    max_chars = int(max(1, max_chars))
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _get_context_limit(model: Any, tokenizer: Any, default: int = 1024) -> int:
    cand: List[int] = []

    tmax = getattr(tokenizer, "model_max_length", None)
    try:
        if tmax is not None:
            cand.append(int(tmax))
    except Exception:
        pass

    cfg = getattr(model, "config", None)
    if cfg is not None:
        for k in ["n_positions", "max_position_embeddings", "seq_length", "context_length"]:
            v = getattr(cfg, k, None)
            try:
                if v is not None:
                    cand.append(int(v))
            except Exception:
                pass

    good = [c for c in cand if isinstance(c, int) and 8 <= c <= 100000]
    if not good:
        return int(default)
    return int(min(good))


def _truncate_ids_and_mask(
    input_ids: Any,
    attention_mask: Any,
    max_len: int,
) -> Tuple[Any, Any]:
    if not TORCH_AVAILABLE:
        return input_ids, attention_mask

    max_len = int(max(2, max_len))
    T = int(input_ids.size(1))
    if T <= max_len:
        return input_ids, attention_mask
    return input_ids[:, -max_len:], attention_mask[:, -max_len:]


def _tokenize_trunc(
    tokenizer: Any,
    text: str,
    max_len: int,
    fallback_char_cap: int = 4096,
) -> Dict[str, Any]:
    max_len = int(max(2, max_len))
    fallback_char_cap = int(max(256, fallback_char_cap))

    try:
        return tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    except (TypeError, ValueError, RuntimeError):
        safe_text = _safe_truncate(text, min(fallback_char_cap, max(256, 8 * max_len)))
        try:
            return tokenizer(safe_text, return_tensors="pt", truncation=True, max_length=max_len)
        except Exception:
            safe_text2 = _safe_truncate(safe_text, min(fallback_char_cap, 2048))
            return tokenizer(safe_text2, return_tensors="pt")


def _ensure_model_device(model: Any, device: str) -> None:
    if not TORCH_AVAILABLE:
        return
    try:
        dev = torch.device(device)
    except Exception:
        return

    try:
        p = next(model.parameters())
        if p.device != dev:
            model.to(dev)
    except StopIteration:
        try:
            model.to(dev)
        except Exception:
            pass
    except Exception:
        try:
            model.to(dev)
        except Exception:
            pass


def _transformers_available() -> bool:
    try:
        import transformers  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def runtime_status() -> Dict[str, bool]:
    """Quick availability snapshot for optional deps."""
    return {
        "torch": bool(TORCH_AVAILABLE),
        "transformers": bool(_transformers_available()),
        "scipy": bool(SCIPY_AVAILABLE),
    }


def resolve_runtime_cfg(cfg: SESConfig, require_nll: bool = False, verbose: bool = False) -> SESConfig:
    """    Return a runtime-resolved copy of cfg.

    - If cfg.use_nll=True but torch/transformers are unavailable:
        * require_nll=False -> disables NLL with a warning
        * require_nll=True  -> raises RuntimeError
    - Validates calibration presence for enabled metrics (after any runtime disabling).
    """
    cfg2: SESConfig = copy.deepcopy(cfg)

    disabled_reason: Optional[str] = None
    if bool(cfg2.use_nll):
        if not TORCH_AVAILABLE:
            disabled_reason = "torch not available"
        elif not _transformers_available():
            disabled_reason = "transformers not available"

        if disabled_reason is not None:
            if require_nll:
                raise RuntimeError(
                    f"use_nll=True was requested but {disabled_reason}. "
                    "Install the missing dependency (torch/transformers), "
                    "or rerun with use_nll disabled."
                )
            warn(f"[warn] use_nll=True but {disabled_reason}; disabling NLL metric for this run.")
            cfg2.use_nll = False

    # Validate that calib covers enabled metrics (after runtime resolution).
    need: List[str] = []
    if cfg2.use_gz:
        need.append("gz")
    if cfg2.use_ent:
        need.append("ent")
    if cfg2.use_nll:
        need.append("nll")

    missing = [m for m in need if m not in (cfg2.calib or {})]
    if missing:
        raise ValueError(
            f"Config inconsistency: enabled metrics {need} but calib missing {missing}. "
            "Re-run 'calibrate' with matching flags (e.g., include --use_nll if needed), "
            "or disable the missing metric(s)."
        )

    # Tag as resolved to avoid repeated deepcopies in hot paths.
    try:
        setattr(cfg2, "_runtime_resolved", True)
        setattr(cfg2, "_runtime_nll_disabled_reason", disabled_reason)
    except Exception:
        pass

    if verbose:
        ms = _enabled_metrics(cfg2)
        extra = f" (nll disabled: {disabled_reason})" if disabled_reason else ""
        warn(f"[runtime] enabled metrics: {', '.join(ms) if ms else 'NONE'}{extra}")

    return cfg2

# ============================================================
# 1. Low-level metrics (gzip, entropy, optional NLL)
# ============================================================

def _gzip_len(raw: bytes) -> int:
    with io.BytesIO() as buf:
        with gzip.GzipFile(fileobj=buf, mode="wb") as f:
            f.write(raw)
        return len(buf.getvalue())


_GZIP_EMPTY_OVERHEAD = _gzip_len(b"")


def gzip_ratio(text: str, cfg: SESConfig) -> float:
    if not text:
        return 0.0
    raw = text.encode("utf-8", errors="ignore")
    denom = max(len(raw), 1)
    comp_len = _gzip_len(raw)

    overhead_correct = bool(cfg.gz_overhead_correct) and (len(raw) >= int(cfg.gz_overhead_min_raw_bytes))
    if overhead_correct:
        comp_len = max(0, comp_len - _GZIP_EMPTY_OVERHEAD)

    return _finite(float(comp_len) / float(denom), 0.0)


def char_entropy_bits(text: str) -> float:
    if not text:
        return 0.0
    c = Counter(text)
    n = sum(c.values())
    if n <= 0:
        return 0.0
    probs = [v / n for v in c.values()]
    return _finite(float(-sum(p * math.log2(p + 1e-12) for p in probs)), 0.0)


def norm_char_entropy(text: str, H_max: float = 8.0) -> float:
    H_max = float(max(H_max, 1e-9))
    return _finite(char_entropy_bits(text) / H_max, 0.0)


def _model_logits(output: Any) -> Any:
    return output.logits if hasattr(output, "logits") else output[0]


def compute_nll_per_token(
    text: str,
    model: Any,
    tokenizer: Any,
    device: str = "cpu",
    max_tokens: int = 1024,
    max_chars_fallback: int = 20000,
    tokenizer_fallback_char_cap: int = 4096,
) -> float:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is not available but NLL was requested.")
    if not text:
        return 0.0

    device = normalize_device(device)
    cap = int(max(2, max_tokens))

    try:
        ctx_limit = _get_context_limit(model, tokenizer, default=cap)
        cap = int(min(cap, max(2, ctx_limit)))
    except Exception:
        pass

    model.eval()
    _ensure_model_device(model, device)

    with torch.no_grad():
        try:
            enc = _tokenize_trunc(
                tokenizer,
                text,
                max_len=cap,
                fallback_char_cap=int(max(256, tokenizer_fallback_char_cap)),
            )
        except Exception:
            safe_text = _safe_truncate(text, int(max(256, max_chars_fallback)))
            safe_text = _safe_truncate(safe_text, int(max(256, tokenizer_fallback_char_cap)))
            enc = tokenizer(safe_text, return_tensors="pt")

        ids = enc["input_ids"].to(device)
        mask = enc.get("attention_mask", torch.ones_like(ids)).to(device)

        ids, mask = _truncate_ids_and_mask(ids, mask, cap)

        out = model(input_ids=ids, attention_mask=mask)
        logits = _model_logits(out)

        if int(logits.size(1)) < 2:
            return 0.0

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        nll_tok = -log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

        valid = shift_mask.to(dtype=nll_tok.dtype)
        denom = valid.sum().clamp_min(1.0)
        return _finite(float((nll_tok * valid).sum().item() / denom.item()), 0.0)


# ============================================================
# 2. Calibration and π↑ / σπ (SES v2.4.2)
# ============================================================

def percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    return _finite(float(np.percentile(np.array(vals, dtype=np.float64), q)), 0.0)


def calibrate_metrics(
    texts: Sequence[str],
    cfg: SESConfig,
    model: Any = None,
    tokenizer: Any = None,
    device: str = "cpu",
) -> Dict[str, Dict[str, float]]:
    device = normalize_device(device)

    vals: Dict[str, List[float]] = {}
    if cfg.use_gz:
        vals["gz"] = [gzip_ratio(t, cfg) for t in texts]
    if cfg.use_ent:
        vals["ent"] = [norm_char_entropy(t, H_max=cfg.ent_H_max) for t in texts]
    if cfg.use_nll:
        if model is None or tokenizer is None:
            raise ValueError("use_nll=True requires model and tokenizer for calibration.")
        _ensure_model_device(model, device)
        vals["nll"] = [
            compute_nll_per_token(
                t, model, tokenizer,
                device=device,
                max_tokens=int(cfg.nll_max_tokens),
                max_chars_fallback=int(cfg.max_state_chars),
                tokenizer_fallback_char_cap=int(cfg.tokenizer_fallback_char_cap),
            )
            for t in texts
        ]

    calib: Dict[str, Dict[str, float]] = {}
    for m, v in vals.items():
        q05 = percentile(v, 5.0)
        q50 = percentile(v, 50.0)
        q95 = percentile(v, 95.0)

        if q50 <= q05:
            q50 = q05 + 1e-6
        if q95 <= q50:
            q95 = q50 + 1e-6

        calib[m] = {"q05": q05, "q50": q50, "q95": q95}
    return calib


def _enabled_metrics(cfg: SESConfig) -> List[str]:
    ms: List[str] = []
    if cfg.use_gz:
        ms.append("gz")
    if cfg.use_ent:
        ms.append("ent")
    if cfg.use_nll:
        ms.append("nll")
    return ms


def _weights(cfg: SESConfig) -> Dict[str, float]:
    w: Dict[str, float] = {}
    if cfg.use_gz:
        w["gz"] = float(cfg.w_gz)
    if cfg.use_ent:
        w["ent"] = float(cfg.w_ent)
    if cfg.use_nll:
        w["nll"] = float(cfg.w_nll)

    ms = _enabled_metrics(cfg)
    if not ms:
        return {}

    s = sum(w.get(m, 0.0) for m in ms)
    if s <= 0:
        return {m: 1.0 / len(ms) for m in ms}
    return {m: w.get(m, 0.0) / s for m in ms}


def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, float(x))))


def s_m(kappa: float, q05: float, q50: float, q95: float) -> float:
    q05 = float(q05)
    q50 = float(q50)
    q95 = float(q95)

    if q50 <= q05:
        q50 = q05 + 1e-6
    if q95 <= q50:
        q95 = q50 + 1e-6

    k = float(kappa)

    if k <= q05:
        return 0.0
    if k <= q50:
        return clip01(0.5 * (k - q05) / ((q50 - q05) + 1e-12))
    if k <= q95:
        return clip01(0.5 + 0.5 * (k - q50) / ((q95 - q50) + 1e-12))
    return 1.0


def _hash_key(text: str, digest_bytes: int = 32) -> str:
    b = text.encode("utf-8", errors="ignore")
    d = int(digest_bytes)
    d = max(16, min(64, d))
    return hashlib.blake2b(b, digest_size=d).hexdigest()


class PiCache:
    def __init__(self, max_items: int = 16384):
        self.max_items = int(max(64, max_items))
        self._d: "OrderedDict[str, Tuple[float, float, Dict[str, float], Dict[str, float]]]" = OrderedDict()

    def get(self, key: str) -> Optional[Tuple[float, float, Dict[str, float], Dict[str, float]]]:
        v = self._d.get(key)
        if v is None:
            return None
        self._d.move_to_end(key, last=True)
        pi, sigma, ks, sm = v
        return float(pi), float(sigma), dict(ks), dict(sm)

    def put(self, key: str, value: Tuple[float, float, Dict[str, float], Dict[str, float]]) -> None:
        self._d[key] = (float(value[0]), float(value[1]), dict(value[2]), dict(value[3]))
        self._d.move_to_end(key, last=True)
        while len(self._d) > self.max_items:
            self._d.popitem(last=False)


def compute_pi_up(
    text: str,
    cfg: SESConfig,
    model: Any = None,
    tokenizer: Any = None,
    device: str = "cpu",
    cache: Optional[PiCache] = None,
) -> Tuple[float, float, Dict[str, float], Dict[str, float]]:
    cfg2 = cfg if bool(getattr(cfg, "_runtime_resolved", False)) else resolve_runtime_cfg(cfg, require_nll=False, verbose=False)

    if not cfg2.calib:
        raise ValueError("cfg.calib is empty. Run calibration first or load calib JSON.")

    device = normalize_device(device)

    key = _hash_key(text, digest_bytes=int(cfg2.hash_digest_bytes))
    if cache is not None:
        hit = cache.get(key)
        if hit is not None:
            return hit

    ks: Dict[str, float] = {}
    if cfg2.use_gz:
        ks["gz"] = gzip_ratio(text, cfg2)
    if cfg2.use_ent:
        ks["ent"] = norm_char_entropy(text, H_max=cfg2.ent_H_max)
    if cfg2.use_nll:
        if model is None or tokenizer is None:
            raise ValueError("use_nll=True requires model and tokenizer.")
        _ensure_model_device(model, device)
        ks["nll"] = compute_nll_per_token(
            text, model, tokenizer,
            device=device,
            max_tokens=int(cfg2.nll_max_tokens),
            max_chars_fallback=int(cfg2.max_state_chars),
            tokenizer_fallback_char_cap=int(cfg2.tokenizer_fallback_char_cap),
        )

    for k in list(ks.keys()):
        ks[k] = _finite(ks[k], 0.0)

    sm: Dict[str, float] = {}
    for m, kappa in ks.items():
        q = cfg2.calib.get(m)
        if q is None:
            raise ValueError(f"Missing calibration for metric '{m}'.")
        q05 = float(q.get("q05", q.get("q50", 0.0)))
        q50 = float(q.get("q50", 0.0))
        q95 = float(q.get("q95", q50 + 1e-6))
        sm[m] = s_m(float(kappa), q05, q50, q95)

    w = _weights(cfg2)
    pi = 0.0
    for m, val in sm.items():
        pi += w.get(m, 0.0) * float(val)
    pi = clip01(_finite(pi, 0.0))

    svals = list(sm.values())
    sigma = 0.0 if len(svals) <= 1 else float(np.sqrt(np.var(np.array(svals, dtype=np.float64))))
    sigma = _finite(sigma, 0.0)

    out = (float(pi), float(sigma), dict(ks), dict(sm))
    if cache is not None:
        cache.put(key, out)
    return out


# ============================================================
# 3. Operations 𝒪(x): admissible transformations
# ============================================================

def _safe_apply_op(f: Callable[[str], str], x: str) -> str:
    try:
        y = f(x)
        if y is None:
            return x
        return str(y)
    except Exception:
        return x


def op_strip_whitespace(text: str) -> str:
    return re.sub(r"[ 	]+", " ", re.sub(r"\n{3,}", "\n\n", text)).strip()


def op_keep_last_fraction(text: str, frac: float = 0.5) -> str:
    if not text:
        return text
    frac = float(min(max(frac, 0.01), 1.0))
    n = max(1, int(len(text) * frac))
    return text[-n:]


def op_extract_first_sentences(text: str, n_sent: int = 3) -> str:
    if not text:
        return text
    sents = re.split(r"(?<=[\.\?\!。！？])\s+", text.strip())
    sents = [s for s in sents if s]
    return " ".join(sents[: max(1, int(n_sent))]).strip()


def op_bullets(text: str, max_lines: int = 12) -> str:
    if not text:
        return text
    parts = re.split(r"[\n]+", text.strip())
    if len(parts) <= 1:
        parts = re.split(r"(?<=[\.\?\!。！？])\s+", text.strip())
    parts = [p.strip() for p in parts if p.strip()]
    parts = parts[: max(1, int(max_lines))]
    return "\n".join([f"- {p}" for p in parts])


def op_sectionize(text: str, max_sections: int = 3) -> str:
    if not text:
        return text
    clean = text.strip()
    L = len(clean)
    if L < 200:
        return clean
    max_sections = max(1, int(max_sections))
    cuts = [int((i + 1) * L / max_sections) for i in range(max_sections - 1)]
    segs = []
    prev = 0
    for c in cuts:
        segs.append(clean[prev:c].strip())
        prev = c
    segs.append(clean[prev:].strip())
    out = []
    for i, s in enumerate(segs, 1):
        if s:
            out.append(f"[Section {i}]\n{s}")
    return "\n\n".join(out)


def op_add_constraints_template(text: str) -> str:
    clean = text.strip()
    template = (
        "CONSTRAINTS:\n"
        "1) Keep claims checkable.\n"
        "2) Use clear definitions for variables.\n"
        "3) Avoid introducing new entities without definition.\n"
        "4) Prefer short steps; if unsure, state uncertainty.\n\n"
        "STATE:\n"
    )
    if clean.startswith("CONSTRAINTS:"):
        return clean
    return template + clean


def build_default_operations() -> Dict[str, Callable[[str], str]]:
    return {
        "strip": op_strip_whitespace,
        "keep_last_half": lambda t: op_keep_last_fraction(t, 0.5),
        "first_sents_3": lambda t: op_extract_first_sentences(t, 3),
        "bullets": op_bullets,
        "sectionize": op_sectionize,
        "constraints": op_add_constraints_template,
    }


# ============================================================
# 4. Distortion cost c(x,y) and improvability r(x)
# ============================================================

def _sample_text_for_similarity(t: str, window: int) -> str:
    t = t or ""
    window = int(max(32, window))
    if len(t) <= 2 * window:
        return t
    return t[:window] + "\n...\n" + t[-window:]


def edit_cost_ratio(a: str, b: str, cfg: SESConfig) -> float:
    if a == b:
        return 0.0

    if cfg.distortion_fast:
        if cfg.distortion_fast_len_only:
            la, lb = len(a), len(b)
            denom = max(1, max(la, lb))
            return _finite(abs(la - lb) / denom, 1.0)
        aa = _sample_text_for_similarity(a, cfg.edit_cost_sample_window)
        bb = _sample_text_for_similarity(b, cfg.edit_cost_sample_window)
        r = SequenceMatcher(None, aa, bb).ratio()
        return _finite(float(1.0 - r), 1.0)

    la, lb = len(a), len(b)
    if max(la, lb) >= int(cfg.edit_cost_long_threshold):
        aa = _sample_text_for_similarity(a, cfg.edit_cost_sample_window)
        bb = _sample_text_for_similarity(b, cfg.edit_cost_sample_window)
        r = SequenceMatcher(None, aa, bb).ratio()
        return _finite(float(1.0 - r), 1.0)

    r = SequenceMatcher(None, a, b).ratio()
    return _finite(float(1.0 - r), 1.0)


def distortion_cost(x: str, y: str, cfg: SESConfig) -> float:
    if x == y:
        return 0.0
    ec = edit_cost_ratio(x, y, cfg)
    lc = abs(len(x) - len(y)) * float(cfg.beta_len)
    return _finite(float(cfg.alpha_edit) * float(ec) + float(lc), 0.0)


def pressure_norm_r(
    x: str,
    cfg: SESConfig,
    ops: Dict[str, Callable[[str], str]],
    model: Any = None,
    tokenizer: Any = None,
    device: str = "cpu",
    cache: Optional[PiCache] = None,
) -> Tuple[float, str, float, Dict[str, float]]:
    pi_x, _, _, _ = compute_pi_up(x, cfg, model=model, tokenizer=tokenizer, device=device, cache=cache)

    if not ops:
        return 0.0, "none", float(pi_x), {}

    best_name: str = "none"
    best_pi: float = float(pi_x)
    best_impr = 0.0
    imprs: Dict[str, float] = {}

    any_changed = False

    for name, f in ops.items():
        y = _safe_truncate(_safe_apply_op(f, x), cfg.max_state_chars)
        if y == x:
            imprs[name] = 0.0
            continue

        any_changed = True
        pi_y, _, _, _ = compute_pi_up(y, cfg, model=model, tokenizer=tokenizer, device=device, cache=cache)
        pi_y = float(pi_y)

        impr = max(0.0, float(pi_x) - pi_y)
        imprs[name] = float(impr)

        if pi_y < best_pi:
            best_pi = pi_y
            best_name = name

        if impr > best_impr:
            best_impr = float(impr)

    if not any_changed:
        return 0.0, "none", float(pi_x), imprs

    return float(best_impr), str(best_name), float(best_pi), imprs


# ============================================================
# 5. R_ε(x): reachable projection via bounded operation composition
# ============================================================

@dataclass
class Candidate:
    text: str
    pi: float
    obj: float
    seq: List[str]


def R_epsilon(
    x: str,
    cfg: SESConfig,
    ops: Dict[str, Callable[[str], str]],
    model: Any = None,
    tokenizer: Any = None,
    device: str = "cpu",
    cache: Optional[PiCache] = None,
) -> Optional[Candidate]:
    if not ops:
        return None

    pi_x, _, _, _ = compute_pi_up(x, cfg, model=model, tokenizer=tokenizer, device=device, cache=cache)
    start = Candidate(text=x, pi=float(pi_x), obj=float(pi_x), seq=[])
    beam: List[Candidate] = [start]
    best_feasible: Optional[Candidate] = None

    visited_global: Dict[str, float] = {}
    if cfg.repsilon_dedup and cfg.repsilon_dedup_across_depths:
        visited_global[_hash_key(x, digest_bytes=int(cfg.hash_digest_bytes))] = float(start.obj)

    for _depth in range(1, int(cfg.K_max) + 1):
        new_beam: List[Candidate] = []
        best_obj_by_key: Dict[str, float] = {}
        best_cand_by_key: Dict[str, Candidate] = {}

        for cand in beam:
            for name, f in ops.items():
                y_raw = _safe_apply_op(f, cand.text)
                y = _safe_truncate(y_raw, cfg.max_state_chars)

                if y == cand.text:
                    continue

                ky = _hash_key(y, digest_bytes=int(cfg.hash_digest_bytes))

                if cfg.repsilon_dedup and cfg.repsilon_dedup_across_depths and (not cfg.repsilon_keep_best_only):
                    if ky in visited_global:
                        continue

                pi_y, _, _, _ = compute_pi_up(y, cfg, model=model, tokenizer=tokenizer, device=device, cache=cache)
                c_xy = distortion_cost(x, y, cfg)
                obj = float(pi_y) + float(cfg.lambda_cost) * float(c_xy)

                if cfg.repsilon_dedup:
                    if cfg.repsilon_dedup_across_depths and cfg.repsilon_keep_best_only:
                        prev_global = visited_global.get(ky)
                        if prev_global is not None and obj >= prev_global - float(cfg.repsilon_obj_improve_eps):
                            continue

                    if cfg.repsilon_keep_best_only:
                        prev = best_obj_by_key.get(ky)
                        if prev is not None and obj >= prev - float(cfg.repsilon_obj_improve_eps):
                            continue
                        best_obj_by_key[ky] = float(obj)
                        best_cand_by_key[ky] = Candidate(text=y, pi=float(pi_y), obj=float(obj), seq=cand.seq + [name])
                    else:
                        if ky in best_obj_by_key:
                            continue
                        best_obj_by_key[ky] = float(obj)
                        best_cand_by_key[ky] = Candidate(text=y, pi=float(pi_y), obj=float(obj), seq=cand.seq + [name])

                    if cfg.repsilon_dedup_across_depths:
                        if cfg.repsilon_keep_best_only:
                            prev_global = visited_global.get(ky)
                            if prev_global is None or obj < prev_global - float(cfg.repsilon_obj_improve_eps):
                                visited_global[ky] = float(obj)
                        else:
                            visited_global[ky] = float(obj)

                    if len(best_obj_by_key) > int(cfg.repsilon_max_unique_per_depth):
                        items = sorted(best_obj_by_key.items(), key=lambda kv: kv[1])[: int(cfg.repsilon_max_unique_per_depth)]
                        keep = set(k for k, _ in items)
                        best_obj_by_key = {k: best_obj_by_key[k] for k in keep}
                        best_cand_by_key = {k: best_cand_by_key[k] for k in keep}
                else:
                    new_beam.append(Candidate(text=y, pi=float(pi_y), obj=float(obj), seq=cand.seq + [name]))

        if cfg.repsilon_dedup:
            new_beam = list(best_cand_by_key.values())

        if not new_beam:
            break

        new_beam.sort(key=lambda z: (z.obj, z.pi, len(z.text)))
        beam = new_beam[: max(1, int(cfg.beam_width))]

        for cand in beam:
            if cand.pi <= float(cfg.epsilon):
                if best_feasible is None or cand.obj < best_feasible.obj:
                    best_feasible = cand

    return best_feasible


# ============================================================
# 6. Complexity proxy (for collapse curve x-axis)
# ============================================================

def hybrid_complexity(text: str, cfg: SESConfig) -> float:
    gz = gzip_ratio(text, cfg)
    ent = norm_char_entropy(text, H_max=cfg.ent_H_max)
    return _finite(float(cfg.complexity_mix_gz) * float(gz) + float(cfg.complexity_mix_ent) * float(ent), 0.0)


# ============================================================
# 7. Ż-CP step + runners
# ============================================================

def _reset_with_constraints(x: str, frac: float = 0.7) -> str:
    tail = op_keep_last_fraction(op_strip_whitespace(x), frac)
    return op_add_constraints_template(tail)


def zcp_step_control(
    x: str,
    cfg: SESConfig,
    ops: Dict[str, Callable[[str], str]],
    model: Any = None,
    tokenizer: Any = None,
    device: str = "cpu",
    cache: Optional[PiCache] = None,
) -> Tuple[str, Dict[str, Any]]:
    cache = cache or PiCache(max_items=cfg.pi_cache_max_items)

    pi, sigma, ks, sm = compute_pi_up(x, cfg, model=model, tokenizer=tokenizer, device=device, cache=cache)
    r_x, best_op, best_pi_after, imprs = pressure_norm_r(
        x, cfg, ops, model=model, tokenizer=tokenizer, device=device, cache=cache
    )

    action = "none"
    x_next = x
    cand: Optional[Candidate] = None

    collapse_pred = bool((float(pi) > float(cfg.tau_pi)) and (float(r_x) <= float(cfg.tau_P)))

    if float(pi) > float(cfg.epsilon):
        cand = R_epsilon(x, cfg, ops, model=model, tokenizer=tokenizer, device=device, cache=cache)
        if cand is not None:
            x_next = cand.text
            action = f"R_epsilon(seq={','.join(cand.seq)})"
        else:
            if best_op in ops and best_op != "none":
                x_next = _safe_apply_op(ops[best_op], x)
                action = f"fallback_one_step({best_op})"
            else:
                if collapse_pred or (float(r_x) <= float(cfg.tau_P)):
                    x_next = _reset_with_constraints(x, frac=0.7)
                    action = "mu_redesign+reset"
                else:
                    x_next = x
                    action = "none"
    else:
        if collapse_pred:
            x_next = _reset_with_constraints(x, frac=0.7)
            action = "mu_redesign+reset"

    x_next = _safe_truncate(x_next, cfg.max_state_chars)

    reset_event = ("reset" in str(action).lower())
    collapse_or_reset = bool(collapse_pred or reset_event)

    info = {
        "pi_up": float(pi),
        "sigma_pi": float(sigma),
        "r_x": float(r_x),
        "best_op": str(best_op),
        "best_pi_after": float(best_pi_after),
        "action": str(action),
        "kappa": dict(ks),
        "s_m": dict(sm),
        "improvements": dict(imprs),
        "R_seq": (cand.seq if cand else []),
        "R_obj": (cand.obj if cand else None),
        "collapse_pred": bool(collapse_pred),
        "reset_event": bool(reset_event),
        "collapse_or_reset": bool(collapse_or_reset),
    }
    return x_next, info


def run_text_state_series(
    states: Sequence[str],
    cfg: SESConfig,
    ops: Optional[Dict[str, Callable[[str], str]]] = None,
    label: str = "state_series",
    model: Any = None,
    tokenizer: Any = None,
    device: str = "cpu",
    apply_control: bool = False,
) -> pd.DataFrame:
    if ops is None:
        ops = build_default_operations()

    rows: List[Dict[str, Any]] = []
    interventions = 0
    max_n = min(len(states), int(cfg.max_steps))
    x_roll = ""

    cache = PiCache(max_items=cfg.pi_cache_max_items)

    for t in range(max_n):
        x = states[t]
        if apply_control and t > 0:
            x = x_roll

        x = _safe_truncate(x, cfg.max_state_chars)
        x_next, info = zcp_step_control(x, cfg, ops, model=model, tokenizer=tokenizer, device=device, cache=cache)

        intervene = (info["action"] != "none")
        if intervene:
            interventions += 1

        rows.append({
            "condition": str(label),
            "step": int(t),
            "text": x if cfg.track_text else "",
            "text_len": int(len(x)),
            "complexity": float(hybrid_complexity(x, cfg)),
            "pi_up": float(info["pi_up"]),
            "sigma_pi": float(info["sigma_pi"]),
            "r_x": float(info["r_x"]),
            "action": str(info["action"]),
            "interventions": int(interventions),
            "intervene": int(intervene),
            "intervened": int(intervene),
            "intervention_type": (str(info["action"]).upper() if intervene else "NONE"),
            "collapse_pred": int(bool(info["collapse_pred"])),
            "reset_event": int(bool(info["reset_event"])),
            "collapse_or_reset": int(bool(info["collapse_or_reset"])),
            "epsilon": float(cfg.epsilon),
            "tau_pi": float(cfg.tau_pi),
            "tau_P": float(cfg.tau_P),
        })

        x_roll = x_next

    return pd.DataFrame(rows)


def _sample_next_token(logits_1v: Any, temperature: float) -> int:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch required for sampling.")
    logits = logits_1v / max(float(temperature), 1e-6)
    probs = F.softmax(logits, dim=-1)
    nxt = torch.multinomial(probs, num_samples=1)
    return int(nxt.item())


def run_lm_generation_with_zcp(
    prompts: Sequence[str],
    model: Any,
    tokenizer: Any,
    cfg: SESConfig,
    device: str = "cpu",
    condition: str = "lm",
    ops: Optional[Dict[str, Callable[[str], str]]] = None,
    apply_zcp: bool = True,
) -> pd.DataFrame:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for LM generation runner.")
    if ops is None:
        ops = build_default_operations()

    device = normalize_device(device)
    set_seed(cfg.seed)

    model.eval()
    _ensure_model_device(model, device)

    ctx_limit = _get_context_limit(model, tokenizer, default=1024)

    all_rows: List[Dict[str, Any]] = []

    cache = PiCache(max_items=cfg.pi_cache_max_items)

    # NOTE: cfg.track_text / --no_text must only affect *logging* of raw text.
    # We still decode the context to compute SES metrics and apply interventions.
    store_text = bool(cfg.track_text)

    for pid, prompt in enumerate(prompts):
        interventions = 0

        enc0 = _tokenize_trunc(
            tokenizer, prompt, max_len=ctx_limit,
            fallback_char_cap=int(cfg.tokenizer_fallback_char_cap),
        )
        input_ids = enc0["input_ids"].to(device)
        attention_mask = enc0.get("attention_mask", torch.ones_like(input_ids)).to(device)
        input_ids, attention_mask = _truncate_ids_and_mask(input_ids, attention_mask, ctx_limit)

        for step in range(int(cfg.max_new_tokens)):
            input_ids, attention_mask = _truncate_ids_and_mask(input_ids, attention_mask, ctx_limit)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = _model_logits(out)
                last_logits = logits[0, -1, :]

            decoded_pre = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
            decoded_pre = _safe_truncate(decoded_pre, cfg.max_state_chars)

            pi_pre, sigma_pre, _, _ = compute_pi_up(
                decoded_pre,
                cfg,
                model=model if cfg.use_nll else None,
                tokenizer=tokenizer if cfg.use_nll else None,
                device=device,
                cache=cache,
            )
            r_pre, best_op_pre, best_pi_after_pre, _ = pressure_norm_r(
                decoded_pre,
                cfg,
                ops,
                model=model if cfg.use_nll else None,
                tokenizer=tokenizer if cfg.use_nll else None,
                device=device,
                cache=cache,
            )

            collapse_pred_pre = bool((float(pi_pre) > float(cfg.tau_pi)) and (float(r_pre) <= float(cfg.tau_P)))

            action = "none"
            decoded_post = decoded_pre
            sampled_token_id: Optional[int] = None

            if apply_zcp:
                if float(pi_pre) > float(cfg.epsilon):
                    cand = R_epsilon(
                        decoded_pre,
                        cfg,
                        ops,
                        model=model if cfg.use_nll else None,
                        tokenizer=tokenizer if cfg.use_nll else None,
                        device=device,
                        cache=cache,
                    )
                    if cand is not None:
                        decoded_post = _safe_truncate(cand.text, cfg.max_state_chars)
                        action = f"R_epsilon(seq={','.join(cand.seq)})"
                        interventions += 1
                    else:
                        if collapse_pred_pre or (float(r_pre) <= float(cfg.tau_P)):
                            decoded_post = _reset_with_constraints(decoded_pre, frac=0.7)
                            decoded_post = _safe_truncate(decoded_post, cfg.max_state_chars)
                            action = "mu_redesign+reset"
                            interventions += 1

                if action != "none":
                    enc_new = _tokenize_trunc(
                        tokenizer, decoded_post, max_len=ctx_limit,
                        fallback_char_cap=int(cfg.tokenizer_fallback_char_cap),
                    )
                    input_ids = enc_new["input_ids"].to(device)
                    attention_mask = enc_new.get("attention_mask", torch.ones_like(input_ids)).to(device)
                    input_ids, attention_mask = _truncate_ids_and_mask(input_ids, attention_mask, ctx_limit)

            if action == "none":
                sampled_token_id = _sample_next_token(last_logits, cfg.temperature)

                next_id_t = torch.tensor([[sampled_token_id]], device=device, dtype=torch.long)
                input_ids = torch.cat([input_ids, next_id_t], dim=1)

                one = torch.ones((attention_mask.size(0), 1), device=device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, one], dim=1)

                input_ids, attention_mask = _truncate_ids_and_mask(input_ids, attention_mask, ctx_limit)

                decoded_post = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)
                decoded_post = _safe_truncate(decoded_post, cfg.max_state_chars)

            pi_post, sigma_post, _, _ = compute_pi_up(
                decoded_post,
                cfg,
                model=model if cfg.use_nll else None,
                tokenizer=tokenizer if cfg.use_nll else None,
                device=device,
                cache=cache,
            )
            r_post, best_op_post, best_pi_after_post, _ = pressure_norm_r(
                decoded_post,
                cfg,
                ops,
                model=model if cfg.use_nll else None,
                tokenizer=tokenizer if cfg.use_nll else None,
                device=device,
                cache=cache,
            )

            reset_event = ("reset" in str(action).lower())
            collapse_or_reset = bool(collapse_pred_pre or reset_event)

            all_rows.append({
                "condition": str(condition),
                "prompt_id": int(pid),
                "prompt": prompt if store_text else "",
                "step": int(step),
                "text": decoded_post if store_text else "",
                "text_len": int(len(decoded_post)),
                "complexity": float(hybrid_complexity(decoded_post, cfg)),
                "pi_up": float(pi_post),
                "sigma_pi": float(sigma_post),
                "r_x": float(r_post),
                "best_op": str(best_op_post),
                "best_pi_after": float(best_pi_after_post),
                "action": str(action),
                "interventions": int(interventions),
                "intervene": int(action != "none"),
                "intervened": int(action != "none"),
                "intervention_type": (str(action).upper() if action != "none" else "NONE"),
                "sampled_token_id": sampled_token_id,
                "epsilon": float(cfg.epsilon),
                "tau_pi": float(cfg.tau_pi),
                "tau_P": float(cfg.tau_P),
                "collapse_pred": int(bool(collapse_pred_pre)),
                "reset_event": int(bool(reset_event)),
                "collapse_or_reset": int(bool(collapse_or_reset)),
                "pi_up_pre": float(pi_pre),
                "sigma_pi_pre": float(sigma_pre),
                "r_x_pre": float(r_pre),
                "best_op_pre": str(best_op_pre),
                "best_pi_after_pre": float(best_pi_after_pre),
                "ctx_limit": int(ctx_limit),
                "device": str(device),
            })

            if action == "none":
                eos_id = getattr(tokenizer, "eos_token_id", None)
                if eos_id is not None and sampled_token_id is not None and int(sampled_token_id) == int(eos_id):
                    break

    return pd.DataFrame(all_rows)


# ============================================================
# 8. Collapse curve computation + plotting + optional fit
# ============================================================

def _safe_int01(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return int(default)
        if isinstance(x, bool):
            return int(x)
        xf = float(x)
        if not np.isfinite(xf):
            return int(default)
        return int(1 if xf >= 0.5 else 0)
    except Exception:
        return int(default)


def compute_collapse_curves(
    df: pd.DataFrame,
    bin_count: int = 20,
    y_metric: str = "pi_up",
    cfg: Optional[SESConfig] = None,
    collapse_rule: Optional[Callable[[pd.Series], bool]] = None,
) -> pd.DataFrame:
    if len(df) == 0:
        raise ValueError("Empty dataframe.")
    d = df.copy()
    if "condition" not in d.columns:
        d["condition"] = "default"

    if "r_x" not in d.columns and "P_norm" in d.columns:
        d["r_x"] = d["P_norm"]

    if cfg is not None:
        tau_pi = float(cfg.tau_pi)
        tau_P = float(cfg.tau_P)
    else:
        tau_pi = float(d["tau_pi"].iloc[0]) if "tau_pi" in d.columns else 0.65
        tau_P = float(d["tau_P"].iloc[0]) if "tau_P" in d.columns else 0.05

    if y_metric not in d.columns:
        raise ValueError(f"y_metric='{y_metric}' not in dataframe columns: {list(d.columns)}")

    if "collapse_pred" not in d.columns:
        d["collapse_pred"] = d.apply(
            lambda r: int((float(r.get("pi_up", 0.0)) > tau_pi) and (float(r.get("r_x", 1.0)) <= tau_P)),
            axis=1
        )
    else:
        d["collapse_pred"] = pd.to_numeric(d["collapse_pred"], errors="coerce").fillna(0).astype(int)

    if "reset_event" not in d.columns:
        d["reset_event"] = d["action"].astype(str).str.lower().str.contains("reset").astype(int)
    else:
        d["reset_event"] = pd.to_numeric(d["reset_event"], errors="coerce").fillna(0).astype(int)

    if collapse_rule is None:
        def collapse_rule_row(row: pd.Series) -> bool:
            if "collapse_or_reset" in row.index:
                return bool(_safe_int01(row.get("collapse_or_reset", 0)))
            a = (float(row.get("pi_up", 0.0)) > tau_pi) and (float(row.get("r_x", 1.0)) <= tau_P)
            b = "reset" in str(row.get("action", "")).lower()
            return bool(a or b)
        collapse_rule = collapse_rule_row

    d["collapse"] = d.apply(collapse_rule, axis=1).astype(int)
    d["intervene"] = (d["action"].astype(str) != "none").astype(int)

    if "complexity" not in d.columns:
        raise ValueError("Dataframe must contain 'complexity' column.")
    min_c = float(d["complexity"].min())
    max_c = float(d["complexity"].max())
    if not np.isfinite(min_c) or not np.isfinite(max_c):
        raise ValueError("Non-finite complexity values found.")
    if max_c <= min_c:
        max_c = min_c + 1e-9

    bin_count = int(max(2, bin_count))
    bins = np.linspace(min_c, max_c + 1e-12, bin_count + 1)

    idx = np.digitize(d["complexity"].astype(float).values, bins, right=False) - 1
    idx = np.clip(idx, 0, bin_count - 1)
    d["complexity_bin"] = idx

    groups = []
    for cond, g in d.groupby("condition"):
        agg = (
            g.groupby("complexity_bin")
            .agg(
                y_mean=(y_metric, "mean"),
                y_std=(y_metric, "std"),
                collapse_rate=("collapse", "mean"),
                # Backward-compatible name:
                collapse_only_rate=("collapse_pred", "mean"),
                reset_rate=("reset_event", "mean"),
                intervention_rate=("intervene", "mean"),
                count=("collapse", "size"),
            )
            .reset_index()
        )

        # Clearer alias (kept alongside compat):
        agg["precursor_rate"] = agg["collapse_only_rate"]

        centers = []
        for b in agg["complexity_bin"].tolist():
            b_i = int(b)
            centers.append(0.5 * (bins[b_i] + bins[b_i + 1]))
        agg["complexity_center"] = centers
        agg["condition"] = cond
        groups.append(agg)

    curve_df = pd.concat(groups, ignore_index=True)
    curve_df = curve_df[
        ["condition", "complexity_bin", "complexity_center",
         "y_mean", "y_std",
         "collapse_rate", "collapse_only_rate", "precursor_rate", "reset_rate",
         "intervention_rate", "count"]
    ].sort_values(["condition", "complexity_bin"])
    return curve_df


def plot_collapse_curves(
    curve_df: pd.DataFrame,
    title: str = "Ż Collapse Curves",
    y_label: str = "y",
    save_path: Optional[str] = None,
):
    if len(curve_df) == 0:
        print("[plot_collapse_curves] No data.")
        return

    plt.figure(figsize=(7.8, 5.2))
    for cond, g in curve_df.groupby("condition"):
        x = g["complexity_center"].values
        y = g["y_mean"].values
        yerr = g["y_std"].fillna(0.0).values
        plt.plot(x, y, marker="o", label=f"{cond}")
        plt.fill_between(x, y - yerr, y + yerr, alpha=0.2)

    plt.xlabel("Complexity (hybrid)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    else:
        plt.show()
    plt.close()


def plot_rates(
    curve_df: pd.DataFrame,
    title: str = "Collapse / Intervention Rates",
    save_path: Optional[str] = None,
):
    if len(curve_df) == 0:
        print("[plot_rates] No data.")
        return

    plt.figure(figsize=(7.8, 5.2))
    for cond, g in curve_df.groupby("condition"):
        x = g["complexity_center"].values
        cr = g["collapse_rate"].values
        ir = g["intervention_rate"].values
        plt.plot(x, cr, marker="o", label=f"{cond}: collapse_rate")
        plt.plot(x, ir, marker="x", label=f"{cond}: intervention_rate")

    plt.xlabel("Complexity (hybrid)")
    plt.ylabel("Rate")
    plt.title(title)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=160)
    else:
        plt.show()
    plt.close()


# Increasing logistic (typical for "rate rises with complexity").
# If you want the decreasing form, flip the sign inside exp().
def two_phase_logistic(c, A0, w, tau1, sigma1, tau2, sigma2):
    s1 = 1.0 / (1.0 + np.exp(-(c - tau1) / max(sigma1, 1e-9)))
    s2 = 1.0 / (1.0 + np.exp(-(c - tau2) / max(sigma2, 1e-9)))
    return A0 * ((1.0 - w) * s1 + w * s2)


def fit_two_phase_logistic(
    curve_df: pd.DataFrame,
    condition: str,
    y_col: str = "y_mean",
) -> Optional[Dict[str, float]]:
    if not SCIPY_AVAILABLE:
        print("[fit_two_phase_logistic] SciPy not available.")
        return None

    sub = curve_df[curve_df["condition"] == condition].dropna(subset=["complexity_center", y_col])
    if len(sub) < 6:
        print(f"[fit_two_phase_logistic] Not enough points for '{condition}'.")
        return None

    x = sub["complexity_center"].astype(float).values
    y = sub[y_col].astype(float).values

    A0_init = float(np.max(y))
    w_init = 0.5
    tau1_init = float(np.percentile(x, 25))
    tau2_init = float(np.percentile(x, 75))
    sigma1_init = max(1e-6, (x.max() - x.min()) / 6.0)
    sigma2_init = max(1e-6, sigma1_init / 2.0)
    p0 = [A0_init, w_init, tau1_init, sigma1_init, tau2_init, sigma2_init]

    try:
        popt, _ = curve_fit(two_phase_logistic, x, y, p0=p0, maxfev=10000)
    except Exception as e:
        print(f"[fit_two_phase_logistic] Fit failed: {e}")
        return None

    A0, w, tau1, sigma1, tau2, sigma2 = popt
    return {
        "A0": float(A0),
        "w": float(w),
        "tau1": float(tau1),
        "sigma1": float(sigma1),
        "tau2": float(tau2),
        "sigma2": float(sigma2),
    }


def compute_z_index(
    curve_df: pd.DataFrame,
    cond_a: str,
    cond_b: str,
    y_col: str = "y_mean",
) -> Optional[float]:
    a = curve_df[curve_df["condition"] == cond_a][["complexity_bin", "complexity_center", y_col]].dropna()
    b = curve_df[curve_df["condition"] == cond_b][["complexity_bin", "complexity_center", y_col]].dropna()
    m = pd.merge(a, b, on="complexity_bin", suffixes=("_a", "_b")).dropna()
    if len(m) < 3:
        print("[compute_z_index] Not enough overlapping bins.")
        return None

    x = 0.5 * (m["complexity_center_a"].astype(float).values + m["complexity_center_b"].astype(float).values)
    ya = m[f"{y_col}_a"].astype(float).values
    yb = m[f"{y_col}_b"].astype(float).values

    order = np.argsort(x)
    x = x[order]
    diff = (ya - yb)[order]

    if len(x) != len(np.unique(x)):
        df_tmp = pd.DataFrame({"x": x, "diff": diff}).groupby("x", as_index=False).mean()
        x = df_tmp["x"].values
        diff = df_tmp["diff"].values

    return float(np.trapz(diff, x))


# ============================================================
# 9. ∂D_α (collapse horizon) — optional estimator
# ============================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (0.0, 1.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2 * n)) / denom
    radius = (z * math.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)) / denom
    return (max(0.0, center - radius), min(1.0, center + radius))


def estimate_collapse_horizon(
    task_family: Sequence[Any],
    solve_fn: Callable[[Any], bool],
    alpha: float = 0.1,
    ci_z: float = 1.96,
) -> Dict[str, Any]:
    results = []
    horizon = None

    for n, tasks in enumerate(task_family):
        trials = list(tasks) if isinstance(tasks, (list, tuple)) else [tasks]
        k = sum(1 for t in trials if solve_fn(t))
        N = len(trials)
        S = k / max(N, 1)
        lo, hi = wilson_ci(k, N, z=ci_z)
        results.append({"n": n, "N": N, "k": k, "S": S, "S_lo": lo, "S_hi": hi})
        if horizon is None and S < alpha:
            horizon = n

    return {"alpha": alpha, "horizon": horizon, "series": results}


# ============================================================
# 10. IO helpers (calib JSON, text blocks)
# ============================================================

def load_text_blocks(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    return blocks


def save_json(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def cfg_to_json(cfg: SESConfig) -> Dict[str, Any]:
    return dataclasses.asdict(cfg)


def _validate_cfg_consistency(cfg: SESConfig) -> None:
    """Validate that cfg.calib contains entries for all metrics enabled in cfg."""
    need: List[str] = []
    if cfg.use_gz:
        need.append("gz")
    if cfg.use_ent:
        need.append("ent")
    if cfg.use_nll:
        need.append("nll")

    missing = [m for m in need if m not in (cfg.calib or {})]
    if missing:
        raise ValueError(
            f"Config inconsistency: enabled metrics {need} but calib missing {missing}. "
            "Re-run calibrate with matching flags (e.g., --use_nll) or disable missing metrics."
        )



def cfg_from_json(d: Dict[str, Any]) -> SESConfig:
    cfg = SESConfig()
    for k, v in d.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if not isinstance(cfg.calib, dict):
        cfg.calib = {}

    # Do NOT mutate based on availability here; validate will account for availability.
    _validate_cfg_consistency(cfg)
    return cfg


# ============================================================
# 11. CLI commands
# ============================================================

def _ensure_tokenizer_padding(tokenizer: Any, model: Any = None) -> None:
    if getattr(tokenizer, "pad_token_id", None) is not None:
        return

    eos = getattr(tokenizer, "eos_token", None)
    if eos is not None:
        try:
            tokenizer.pad_token = eos
            return
        except Exception:
            pass

    try:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        if model is not None and hasattr(model, "resize_token_embeddings"):
            try:
                model.resize_token_embeddings(len(tokenizer))
            except Exception:
                pass
    except Exception:
        pass


def _maybe_load_hf_lm_for_nll(
    model_name: str,
    device: str,
    tokenizer_pad_to_eos: bool = True,
):
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for HF LM support.")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError("transformers is required for HF LM support") from e

    device = normalize_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    _ensure_model_device(model, device)
    model.eval()

    if tokenizer_pad_to_eos:
        _ensure_tokenizer_padding(tokenizer, model=model)

    return model, tokenizer, device


def cmd_calibrate(args: argparse.Namespace) -> None:
    cfg = SESConfig()
    cfg.use_nll = bool(args.use_nll)
    if cfg.use_nll and (not TORCH_AVAILABLE or not _transformers_available()):
        raise RuntimeError("use_nll=True requires both torch and transformers. Install them or rerun without --use_nll.")
    cfg.use_gz = True
    cfg.use_ent = True
    cfg.nll_max_tokens = int(args.nll_max_tokens)
    cfg.tokenizer_fallback_char_cap = int(getattr(args, "tokenizer_fallback_char_cap", cfg.tokenizer_fallback_char_cap))

    texts = load_text_blocks(args.calib_txt)
    if len(texts) < 5:
        raise ValueError("Calibration corpus is too small. Provide more text blocks.")

    model = None
    tokenizer = None
    device = normalize_device(args.device)
    if str(args.device).lower().startswith("cuda") and device == "cpu":
        warn("[warn] CUDA requested but unavailable; falling back to cpu.")

    if cfg.use_nll:
        if not TORCH_AVAILABLE:
            raise RuntimeError("use_nll=True but torch is not available.")
        model, tokenizer, device = _maybe_load_hf_lm_for_nll(args.model_name, device=device)

    cfg.calib = calibrate_metrics(texts, cfg, model=model, tokenizer=tokenizer, device=device)
    save_json(cfg_to_json(cfg), args.out)
    print(f"[calibrate] saved calibration to {args.out}")
    print(json.dumps(cfg.calib, indent=2, ensure_ascii=False))


def cmd_run_text(args: argparse.Namespace) -> None:
    cfg = cfg_from_json(load_json(args.calib))
    cfg.max_steps = int(args.max_steps)
    cfg.track_text = (not bool(getattr(args, "no_text", False)))

    if getattr(args, "nll_max_tokens", None) is not None:
        try:
            cfg.nll_max_tokens = int(args.nll_max_tokens)
        except Exception:
            pass

    states = load_text_blocks(args.inputs)
    if len(states) == 0:
        raise ValueError("No state blocks found.")

    ops = build_default_operations()
    model = None
    tokenizer = None
    device = normalize_device(getattr(args, "device", "cpu"))
    if str(getattr(args, "device", "")).lower().startswith("cuda") and device == "cpu":
        warn("[warn] CUDA requested but unavailable; falling back to cpu.")
    require_nll = bool(getattr(args, "require_nll", False))
    cfg = resolve_runtime_cfg(cfg, require_nll=require_nll, verbose=True)
    if cfg.use_nll:
        model_name = getattr(args, "model_name", "gpt2")
        model, tokenizer, device = _maybe_load_hf_lm_for_nll(model_name, device=device)

    df = run_text_state_series(
        states,
        cfg,
        ops=ops,
        label=args.condition,
        model=model,
        tokenizer=tokenizer,
        device=device,
        apply_control=bool(getattr(args, "apply_control", False)),
    )
    df.to_csv(args.out, index=False)
    print(f"[run_text] wrote log to {args.out}")


def cmd_run_lm(args: argparse.Namespace) -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch is required for LM runner.")

    cfg = cfg_from_json(load_json(args.calib))
    cfg.max_new_tokens = int(args.max_new_tokens)
    cfg.temperature = float(args.temperature)
    cfg.nll_max_tokens = int(args.nll_max_tokens)
    cfg.track_text = (not bool(getattr(args, "no_text", False)))

    # run_lm requires transformers for generation
    if not _transformers_available():
        raise RuntimeError("run_lm requires transformers. Install it or use run_text instead.")

    require_nll = bool(getattr(args, "require_nll", False))
    cfg = resolve_runtime_cfg(cfg, require_nll=require_nll, verbose=True)

    prompts = load_text_blocks(args.prompts)
    if len(prompts) == 0:
        raise ValueError("No prompts found.")

    device = normalize_device(args.device)
    if str(args.device).lower().startswith("cuda") and device == "cpu":
        warn("[warn] CUDA requested but unavailable; falling back to cpu.")

    model, tokenizer, device = _maybe_load_hf_lm_for_nll(args.model_name, device=device)

    ops = build_default_operations()
    df = run_lm_generation_with_zcp(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        device=device,
        condition=args.condition,
        ops=ops,
        apply_zcp=bool(args.apply_zcp),
    )
    df.to_csv(args.out, index=False)
    print(f"[run_lm] wrote log to {args.out}")


def cmd_analyze(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.log)

    cfg = None
    if all(c in df.columns for c in ["tau_pi", "tau_P", "epsilon"]):
        cfg = SESConfig()
        cfg.tau_pi = float(pd.to_numeric(df["tau_pi"], errors="coerce").iloc[0])
        cfg.tau_P = float(pd.to_numeric(df["tau_P"], errors="coerce").iloc[0])
        cfg.epsilon = float(pd.to_numeric(df["epsilon"], errors="coerce").iloc[0])

    curve_df = compute_collapse_curves(
        df,
        bin_count=int(args.bins),
        y_metric=args.y_metric,
        cfg=cfg,
        collapse_rule=None,
    )

    curve_path = args.out_prefix + "curve.csv"
    curve_df.to_csv(curve_path, index=False)
    print(f"[analyze] wrote curve to {curve_path}")

    no_plot = bool(getattr(args, "no_plot", False))
    save_plots = bool(getattr(args, "save_plots", False))

    # Safety: if running headless, always save plots (show can hang/fail).
    headless = (os.name != "nt") and (os.environ.get("DISPLAY", "") == "") and (os.environ.get("WAYLAND_DISPLAY", "") == "")

    if not no_plot:
        if save_plots or headless:
            p1 = args.out_prefix + "collapse_curves.png"
            p2 = args.out_prefix + "rates.png"
            plot_collapse_curves(curve_df, title="Ż Collapse Curves", y_label=args.y_metric, save_path=p1)
            plot_rates(curve_df, title="Collapse / Intervention Rates", save_path=p2)
            print(f"[analyze] saved plots: {p1}, {p2}")
        else:
            plot_collapse_curves(curve_df, title="Ż Collapse Curves", y_label=args.y_metric, save_path=None)
            plot_rates(curve_df, title="Collapse / Intervention Rates", save_path=None)

    if args.fit and SCIPY_AVAILABLE:
        for cond in sorted(curve_df["condition"].unique().tolist()):
            params = fit_two_phase_logistic(curve_df, cond, y_col="y_mean")
            print(f"[fit] {cond}: {params}")

    if args.z_index and args.cond_a and args.cond_b:
        z = compute_z_index(curve_df, args.cond_a, args.cond_b, y_col="y_mean")
        print(f"[Ż-Index] ∫(A-B) dc = {z}")


def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ż Collapse Lab (SES v2.4.2 single-file, patched)")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("calibrate", help="Calibrate q05/q50/q95 on C_dev")
    c.add_argument("--calib_txt", required=True, help="Text file with blocks separated by blank lines")
    c.add_argument("--out", required=True, help="Output JSON path (stores SESConfig incl. calib)")
    c.add_argument("--use_nll", action="store_true", help="Include NLL metric (requires model)")
    c.add_argument("--model_name", default="gpt2", help="HF model name for NLL calibration")
    c.add_argument("--device", default="cpu", help="cpu or cuda or mps")
    c.add_argument("--nll_max_tokens", default=1024, help="Max tokens for NLL evaluation")
    c.add_argument("--tokenizer_fallback_char_cap", default=4096, help="Hard cap for tokenizer fallback chars")

    r1 = sub.add_parser("run_text", help="Log SES metrics over provided state blocks (no LM)")
    r1.add_argument("--inputs", required=True, help="Text file with state blocks separated by blank lines")
    r1.add_argument("--calib", required=True, help="Calibration JSON from 'calibrate'")
    r1.add_argument("--out", required=True, help="CSV log output")
    r1.add_argument("--condition", default="state_series", help="Condition label")
    r1.add_argument("--max_steps", default=999999, help="Max states to process")
    r1.add_argument("--model_name", default="gpt2", help="HF causal LM name (used only if calib.use_nll=True)")
    r1.add_argument("--device", default="cpu", help="cpu or cuda or mps (used only if calib.use_nll=True)")
    r1.add_argument("--nll_max_tokens", default=None, help="Override max tokens for NLL evaluation")
    r1.add_argument("--apply_control", action="store_true", help="Chain x <- x_next each step (control rollout)")
    r1.add_argument("--no_text", action="store_true", help="Do not store raw text in CSV (lighter logs)")
    r1.add_argument("--require_nll", action="store_true", help="Error if NLL was requested in calib but is unavailable at runtime")

    r2 = sub.add_parser("run_lm", help="Run LM generation and log SES metrics (optional Ż-CP interventions)")
    r2.add_argument("--prompts", required=True, help="Text file with prompt blocks separated by blank lines")
    r2.add_argument("--calib", required=True, help="Calibration JSON from 'calibrate'")
    r2.add_argument("--out", required=True, help="CSV log output")
    r2.add_argument("--condition", default="lm", help="Condition label")
    r2.add_argument("--model_name", "--model", dest="model_name", default="gpt2", help="HF causal LM name")
    r2.add_argument("--device", default="cpu", help="cpu or cuda or mps")
    r2.add_argument("--apply_zcp", action="store_true", help="Enable SES control interventions during generation")
    r2.add_argument("--max_new_tokens", default=64, help="Max new tokens")
    r2.add_argument("--temperature", default=1.0, help="Sampling temperature")
    r2.add_argument("--nll_max_tokens", default=1024, help="Max tokens for NLL evaluation")
    r2.add_argument("--no_text", action="store_true", help="Do not store raw text in CSV (lighter logs)")
    r2.add_argument("--require_nll", action="store_true", help="Error if NLL was requested in calib but is unavailable at runtime")

    a = sub.add_parser("analyze", help="Analyze a CSV log: collapse curves, rates, optional fit/Z-index")
    a.add_argument("--log", required=True, help="CSV log produced by run_text/run_lm")
    a.add_argument("--out_prefix", default="out_", help="Prefix for outputs (e.g., out_)")
    a.add_argument("--bins", default=20, help="Bin count for curves")
    a.add_argument("--y_metric", default="pi_up", help="Metric to plot on y-axis (pi_up, r_x, etc.)")
    a.add_argument("--fit", action="store_true", help="Fit 2-phase logistic if SciPy available")
    a.add_argument("--z_index", action="store_true", help="Compute Ż-Index between two conditions")
    a.add_argument("--cond_a", default=None, help="Condition A for Ż-Index")
    a.add_argument("--cond_b", default=None, help="Condition B for Ż-Index")
    a.add_argument("--no_plot", action="store_true", help="Disable plotting (headless/CI safe)")
    a.add_argument("--save_plots", action="store_true", help="Save plots to files instead of showing")

    return p


def main() -> None:
    parser = build_cli()
    args = parser.parse_args()

    if not getattr(args, "cmd", None):
        parser.print_help()
        raise SystemExit(2)

    if args.cmd == "calibrate":
        cmd_calibrate(args)
    elif args.cmd == "run_text":
        cmd_run_text(args)
    elif args.cmd == "run_lm":
        cmd_run_lm(args)
    elif args.cmd == "analyze":
        cmd_analyze(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
