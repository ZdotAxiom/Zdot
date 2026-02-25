import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Core SES utilities (complexity + reset + device normalize)
# You can swap to a different module if needed, as long as it exports:
#   - SESConfig
#   - hybrid_complexity(text, cfg)
#   - _reset_with_constraints(text, frac)
#   - _safe_truncate(text, max_chars)
#   - normalize_device(device_str)
import z_collapse_lab_SES_v2_4_2_vsave_compat as ses


# -----------------------------
# Reproducibility helpers
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_blocks(path: str):
    txt = Path(path).read_text(encoding="utf-8")
    blocks = [b.strip() for b in txt.split("\n\n") if b.strip()]
    return blocks


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def norm_q50_q95(x: float, q50: float, q95: float) -> float:
    den = (q95 - q50)
    if abs(den) < 1e-12:
        return 0.0
    return clamp01((x - q50) / den)


# -----------------------------
# Sampling
# -----------------------------

def sample_top_p(logits, temperature: float, top_p: float):
    """Nucleus sampling from a single-step logits vector."""
    logits = logits / max(float(temperature), 1e-6)
    probs = torch.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        mask = (cumsum <= top_p)
        mask[..., 0] = True
        filt = sorted_probs * mask
        filt = filt / filt.sum()
        pick = torch.multinomial(filt, 1)
        return int(sorted_idx[pick].item())

    pick = torch.multinomial(probs, 1)
    return int(pick.item())


# -----------------------------
# Repetition score (v2-style)
# -----------------------------

def repetition_score(text: str, n: int = 3, tail_chars: int = 2000) -> float:
    """A simple, deterministic repetition metric.

    We compute n-gram repetition on the tail of the decoded text.
    Score in [0,1], higher means more repetition.

    This intentionally stays lightweight (CPU-friendly).
    """
    if n <= 1:
        return 0.0

    tail = text[-int(tail_chars):] if tail_chars > 0 else text
    if len(tail) < n:
        return 0.0

    # Character-level n-grams (works even without tokenization alignment)
    ngrams = [tail[i:i + n] for i in range(0, len(tail) - n + 1)]
    total = len(ngrams)
    if total <= 1:
        return 0.0

    uniq = len(set(ngrams))
    rep = 1.0 - (uniq / total)
    return float(clamp01(rep))


def resolve_ctx_limit(tokenizer, model, default: int = 1024) -> int:
    candidates = []
    for attr in ("n_positions", "max_position_embeddings"):
        val = getattr(getattr(model, "config", None), attr, None)
        if isinstance(val, int) and val > 0:
            candidates.append(val)
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int) and 0 < tok_max < 1_000_000:
        candidates.append(tok_max)
    return int(min(candidates)) if candidates else int(default)


def main():
    ap = argparse.ArgumentParser()

    # experiment id (paper-fixed)
    ap.add_argument("--exp_id", type=str, default="vSAVE-1.1.1")

    # run identity
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["baseline", "controlled"], required=True)
    ap.add_argument("--pillar", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    # model
    ap.add_argument("--model_id", type=str, default="gpt2")
    ap.add_argument("--device", type=str, default="cpu")

    # decoding
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top_p", type=float, default=0.95)

    # paper-fixed collapse rule (PI)
    ap.add_argument("--epsilon_pi", type=float, default=0.85)
    ap.add_argument("--k_consecutive", type=int, default=3)

    # v2-style repetition rule (REP)
    ap.add_argument("--rep_ngram", type=int, default=3)
    ap.add_argument("--rep_threshold", type=float, default=0.20)
    ap.add_argument("--rep_tail_chars", type=int, default=2000)

    # too-short rule (paper-fixed)
    # NOTE: based on generated tokens (prompt length excluded).
    ap.add_argument("--min_tokens", type=int, default=64,
                    help="alias for --min_generated_tokens (kept for v2 compatibility)")

    # paper-fixed intervention budget (controlled only)
    ap.add_argument("--max_interventions", type=int, default=5)
    ap.add_argument("--reset_frac", type=float, default=0.70)

    args = ap.parse_args()

    min_tokens = int(args.min_tokens)
    min_generated_tokens = int(args.min_tokens)

    set_seed(args.seed)

    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    if "complexity" not in calib:
        raise RuntimeError("calib.json must contain calib['complexity'] with q50/q95 (independent calibration corpus)")
    q50 = float(calib["complexity"]["q50"])
    q95 = float(calib["complexity"]["q95"])

    prompts = load_blocks(args.prompts)
    if not prompts:
        raise RuntimeError("prompts file has no blocks (split by blank lines)")
    prompt_id = int(args.seed % len(prompts))
    prompt = prompts[prompt_id]

    device = ses.normalize_device(args.device)

    tok = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id).to(device)
    model.eval()

    # pad token to avoid warnings / generation errors
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tok))

    ctx_limit = resolve_ctx_limit(tok, model, default=1024)
    eos_id = tok.eos_token_id

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=ctx_limit)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    start_len = int(input_ids.shape[1])
    prompt_len_tokens = int(start_len)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    timestamp_start = datetime.now().astimezone().isoformat(timespec="seconds")

    meta = {
        "record_type": "meta",
        "exp_id": args.exp_id,
        "mode": args.mode,
        "pillar": args.pillar,
        "seed": int(args.seed),
        "model_id": args.model_id,
        "device": args.device,
        "decoder": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_new_tokens": int(args.max_new_tokens),
        },
        "timestamp_start": timestamp_start,
        "prompt_id": int(prompt_id),
        "prompt_len_tokens": int(prompt_len_tokens),
        "ctx_limit": int(ctx_limit),

        # π definition (1-page fixed)
        "pi_source": "complexity",
        "q50": q50,
        "q95": q95,
        "epsilon_pi": float(args.epsilon_pi),
        "k_consecutive": int(args.k_consecutive),

        # REP + SHORT rules
        "rep_ngram": int(args.rep_ngram),
        "rep_threshold": float(args.rep_threshold),
        "rep_tail_chars": int(args.rep_tail_chars),
        "min_tokens": int(min_tokens),
        "min_generated_tokens": int(min_generated_tokens),
        "token_idx_schema": "token_idx_ctx + token_idx_global (global monotonic)",
        "too_short_schema": "generated_tokens = token_idx_global - prompt_len_tokens",
        "too_short_eval": "final_only",
        "collapse_flag_spec": "collapse_flag = 1 if (pi_flag OR rep_flag) else 0",
    }

    rows = [meta]

    cfg = ses.SESConfig()

    interventions = 0
    H_pre = None

    collapse_streak = 0

    collapse_run = 0
    collapse_reason = "NONE"

    extra_tokens_due_to_intervention = 0.0
    global_generated_tokens = 0

    def decode(ids):
        return tok.decode(ids[0].tolist(), skip_special_tokens=True)

    for step in range(int(args.max_new_tokens)):
        text = ses._safe_truncate(decode(input_ids), 4000)
        token_idx_ctx = int(input_ids.shape[1])
        token_idx_ctx_pre = int(token_idx_ctx)
        token_idx_global = int(prompt_len_tokens + global_generated_tokens)
        generated_tokens = int(token_idx_global - prompt_len_tokens)

        # π = complexity_norm
        pi_raw = float(ses.hybrid_complexity(text, cfg))
        pi_norm = float(norm_q50_q95(pi_raw, q50, q95))

        # repetition auxiliary
        rep = float(repetition_score(text, n=int(args.rep_ngram), tail_chars=int(args.rep_tail_chars)))

        pi_flag = 1 if (pi_norm > float(args.epsilon_pi)) else 0
        rep_flag = 1 if (rep > float(args.rep_threshold)) else 0
        too_short_flag = 1 if (generated_tokens < int(min_tokens)) else 0

        # unified step-level collapse flag (mode-parity)
        collapse_flag = 1 if (pi_flag or rep_flag) else 0
        collapse_streak = (collapse_streak + 1) if collapse_flag else 0

        intervened = 0
        intervention_type = "NONE"
        post_text = None
        token_idx_ctx_post = None
        pi_raw_post = None
        pi_norm_post = None
        rep_post = None
        collapse_flag_post = None

        # baseline collapse rule: sustained collapse_flag
        if args.mode == "baseline" and collapse_streak >= int(args.k_consecutive):
            collapse_run = 1
            if False and too_short_flag:
                collapse_reason = "TOO_SHORT"
            elif pi_flag:
                collapse_reason = "PI"
            elif rep_flag:
                collapse_reason = "REP"
            else:
                collapse_reason = "NONE"

        # controlled intervention rule: sustained collapse_flag triggers deterministic intervention
        if args.mode == "controlled" and collapse_streak >= int(args.k_consecutive):
            if interventions < int(args.max_interventions):
                new_text = ses._reset_with_constraints(text, frac=float(args.reset_frac))
                new_text = ses._safe_truncate(new_text, 4000)

                enc2 = tok(new_text, return_tensors="pt", truncation=True, max_length=ctx_limit)
                input_ids = enc2["input_ids"].to(device)
                attn = enc2.get("attention_mask", torch.ones_like(input_ids)).to(device)

                interventions += 1
                intervened = 1
                intervention_type = "RESET_CONSTRAINTS"
                if H_pre is None:
                    H_pre = int(token_idx_global)

                token_idx_ctx_post = int(input_ids.shape[1])
                post_text = ses._safe_truncate(decode(input_ids), 4000)
                pi_raw_post = float(ses.hybrid_complexity(post_text, cfg))
                pi_norm_post = float(norm_q50_q95(pi_raw_post, q50, q95))
                rep_post = float(repetition_score(post_text, n=int(args.rep_ngram), tail_chars=int(args.rep_tail_chars)))
                pi_flag_post = 1 if (pi_norm_post > float(args.epsilon_pi)) else 0
                rep_flag_post = 1 if (rep_post > float(args.rep_threshold)) else 0
                collapse_flag_post = 1 if (pi_flag_post or rep_flag_post) else 0

                extra_tokens_due_to_intervention += max(0, token_idx_ctx_post - token_idx_ctx_pre)

                # reset streak after intervention
                collapse_streak = 0
            else:
                collapse_run = 1
                collapse_reason = "BUDGET_EXCEEDED"

        rows.append({
            "record_type": "step",
            "step_idx": int(step),
            "token_idx_ctx": int(token_idx_ctx),
            "token_idx_ctx_pre": int(token_idx_ctx_pre),
            "token_idx_ctx_post": int(token_idx_ctx_post) if token_idx_ctx_post is not None else None,
            "token_idx_global": int(token_idx_global),
            "generated_tokens": int(generated_tokens),
            "token_idx_pre": int(token_idx_ctx_pre),
            "token_idx_post": int(token_idx_ctx_post) if token_idx_ctx_post is not None else None,

            "pi_raw": float(pi_raw),
            "pi_norm": float(pi_norm),
            "pi_flag": int(pi_flag),
            "pi_raw_post": float(pi_raw_post) if pi_raw_post is not None else None,
            "pi_norm_post": float(pi_norm_post) if pi_norm_post is not None else None,

            "rep_score": float(rep),
            "rep_flag": int(rep_flag),
            "rep_score_post": float(rep_post) if rep_post is not None else None,
            "too_short_flag": int(too_short_flag),

            "collapse_flag": int(collapse_flag),
            "collapse_flag_post": int(collapse_flag_post) if collapse_flag_post is not None else None,

            "intervened": int(intervened),
            "intervention_type": str(intervention_type),

            "text_preview": text[:160].replace("\n", " "),
            "text_preview_post": post_text[:160].replace("\n", " ") if post_text is not None else None,
        })

        if collapse_run == 1:
            break

        # --- normal decode ---
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[0, -1, :]

        nxt = sample_top_p(logits, args.temperature, args.top_p)
        nxt_t = torch.tensor([[nxt]], device=device, dtype=torch.long)

        input_ids = torch.cat([input_ids, nxt_t], dim=1)
        attn = torch.cat([attn, torch.ones((attn.size(0), 1), device=device, dtype=attn.dtype)], dim=1)
        global_generated_tokens += 1

        if input_ids.shape[1] > ctx_limit:
            input_ids = input_ids[:, -ctx_limit:]
            attn = attn[:, -ctx_limit:]

        # EOS handling
        if eos_id is not None and int(nxt) == int(eos_id):
            break

    t1 = time.time()

    final_total_len = int(prompt_len_tokens + global_generated_tokens)
    timestamp_end = datetime.now().astimezone().isoformat(timespec="seconds")

    final = {
        "record_type": "final",
        "timestamp_end": timestamp_end,
        "H_pre": H_pre if (args.mode == "controlled") else None,
        "n_tokens_total": int(final_total_len),
        "n_interventions": int(interventions) if (args.mode == "controlled") else 0,

        "collapse_run": int(collapse_run),
        "collapse_reason": str(collapse_reason),

        "runtime_sec": float(t1 - t0),
        "extra_tokens_due_to_intervention": float(extra_tokens_due_to_intervention) if (args.mode == "controlled") else 0.0,
    }
    rows.append(final)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[run_vsave_with_seed_v4] wrote: {out_path} "
        f"steps={len(rows)-2} collapse_run={collapse_run} "
        f"reason={collapse_reason} n_interventions={final['n_interventions']}"
    )


if __name__ == "__main__":
    main()
