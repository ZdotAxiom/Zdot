import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

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


# -----------------------------
# Repetition score (v2-style)
# -----------------------------

def repetition_score(text: str, n: int = 3, tail_chars: int = 2000) -> float:
    """Simple deterministic repetition metric on tail chars (char n-grams)."""
    if n <= 1:
        return 0.0
    tail = text[-int(tail_chars):] if tail_chars > 0 else text
    if len(tail) < n:
        return 0.0
    ngrams = [tail[i:i + n] for i in range(0, len(tail) - n + 1)]
    total = len(ngrams)
    if total <= 1:
        return 0.0
    uniq = len(set(ngrams))
    rep = 1.0 - (uniq / total)
    return float(clamp01(rep))


# -----------------------------
# Token-level n-gram blocking
# -----------------------------

def banned_tokens_ngram(input_ids_1d, n: int):
    """
    Standard token-level n-gram blocking:
      If the last (n-1) tokens have appeared before, ban any token that would
      recreate an n-gram already seen.
    """
    if n <= 1:
        return set()
    ids = list(map(int, input_ids_1d))
    if len(ids) < (n - 1) + 1:
        return set()

    # build map: prefix(n-1) -> set(next_token)
    prefix_to_next = {}
    for i in range(len(ids) - n + 1):
        prefix = tuple(ids[i:i + n - 1])
        nxt = ids[i + n - 1]
        if prefix not in prefix_to_next:
            prefix_to_next[prefix] = set()
        prefix_to_next[prefix].add(nxt)

    cur_prefix = tuple(ids[-(n - 1):])
    return prefix_to_next.get(cur_prefix, set())


# -----------------------------
# Sampling (top_p + optional top_k + optional ngram block)
# -----------------------------

def sample_next_token(
    logits_1d,
    temperature: float,
    top_p: float,
    top_k: int = 0,
    banned_token_ids=None
) -> int:
    """
    Nucleus sampling with optional top-k and optional banned tokens.
    logits_1d: shape [vocab]
    """
    logits = logits_1d.clone()

    # ban tokens (ngram block)
    if banned_token_ids:
        banned = list(banned_token_ids)
        if len(banned) > 0:
            logits[banned] = -1e10

    # temperature
    logits = logits / max(float(temperature), 1e-6)

    # top-k filter
    if int(top_k) and int(top_k) > 0:
        k = int(top_k)
        if k < logits.numel():
            v, idx = torch.topk(logits, k)
            mask = torch.full_like(logits, fill_value=True, dtype=torch.bool)
            mask[idx] = False
            logits[mask] = -1e10

    # top-p (nucleus)
    probs = torch.softmax(logits, dim=-1)
    if float(top_p) < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        keep = (cumsum <= float(top_p))
        keep[..., 0] = True
        filt = sorted_probs * keep
        denom = filt.sum()
        if float(denom.item()) <= 0:
            # fallback: argmax
            return int(torch.argmax(probs).item())
        filt = filt / denom
        pick = torch.multinomial(filt, 1)
        return int(sorted_idx[pick].item())

    pick = torch.multinomial(probs, 1)
    return int(pick.item())


def main():
    ap = argparse.ArgumentParser()

    # experiment id
    ap.add_argument("--exp_id", type=str, default="vNEXT-1.2")

    # run identity
    ap.add_argument("--seed", type=int, required=True)

    # NOTE: vNEXT adds baseline variants to crush NeurIPS W1 (strawman baseline)
    ap.add_argument(
        "--mode",
        choices=[
            "baseline",
            "controlled",
            "baseline_temp09",
            "baseline_topk50",
            "baseline_ngram3",
        ],
        required=True
    )

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

    # new (baseline-strengthening knobs)
    ap.add_argument("--top_k", type=int, default=0, help="0 disables top-k; e.g., 50 for top-k baseline")
    ap.add_argument("--ngram_block", type=int, default=0, help="0 disables; e.g., 3 for 3-gram blocking baseline")

    # collapse rule (PI)
    ap.add_argument("--epsilon_pi", type=float, default=0.85)
    ap.add_argument("--k_consecutive", type=int, default=3)

    # repetition rule (REP)
    ap.add_argument("--rep_ngram", type=int, default=3)
    ap.add_argument("--rep_threshold", type=float, default=0.20)
    ap.add_argument("--rep_tail_chars", type=int, default=2000)

    # too-short rule
    ap.add_argument("--min_tokens", type=int, default=64)

    # controlled intervention budget
    ap.add_argument("--max_interventions", type=int, default=5)
    ap.add_argument("--reset_frac", type=float, default=0.70)

    # budget reporting (keep legacy collapse fields but expose budget_exhausted)
    ap.add_argument(
        "--treat_budget_as_collapse",
        type=int,
        default=1,
        help="1 = legacy: BUDGET_EXCEEDED counts as collapse_run. 0 = separate budget_exhausted flag."
    )

    args = ap.parse_args()

    # -----------------------------
    # mode normalization + variants
    # -----------------------------
    mode_original = str(args.mode)
    mode_core = mode_original
    baseline_variant = "NONE"

    if mode_original.startswith("baseline_"):
        # Treat as baseline core mode, with deterministic parameter overrides
        mode_core = "baseline"
        baseline_variant = mode_original

        # Apply variant overrides (paper-frozen decoding stays for vSAVE; vNEXT explores strong baselines)
        if mode_original == "baseline_temp09":
            args.temperature = 0.90
        elif mode_original == "baseline_topk50":
            args.top_k = 50
        elif mode_original == "baseline_ngram3":
            args.ngram_block = 3

    set_seed(int(args.seed))

    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    if "complexity" not in calib:
        raise RuntimeError("calib must contain calib['complexity'] with q50/q95")
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

    # pad token
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

    prompt_len_tokens = int(input_ids.shape[1])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    timestamp_start = datetime.now().astimezone().isoformat(timespec="seconds")

    meta = {
        "record_type": "meta",
        "exp_id": str(args.exp_id),
        "mode": mode_original,         # keep the original label for analysis
        "mode_core": mode_core,        # baseline/controlled
        "baseline_variant": baseline_variant,

        "pillar": str(args.pillar),
        "seed": int(args.seed),
        "model_id": str(args.model_id),
        "device": str(args.device),

        "decoder": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "ngram_block": int(args.ngram_block),
            "max_new_tokens": int(args.max_new_tokens),
        },

        "timestamp_start": timestamp_start,
        "prompt_id": int(prompt_id),
        "prompt_len_tokens": int(prompt_len_tokens),
        "ctx_limit": int(ctx_limit),

        # π definition
        "pi_source": "complexity",
        "q50": q50,
        "q95": q95,
        "epsilon_pi": float(args.epsilon_pi),
        "k_consecutive": int(args.k_consecutive),

        # REP + SHORT
        "rep_ngram": int(args.rep_ngram),
        "rep_threshold": float(args.rep_threshold),
        "rep_tail_chars": int(args.rep_tail_chars),
        "min_tokens": int(args.min_tokens),

        # Intervention settings
        "max_interventions": int(args.max_interventions),
        "reset_frac": float(args.reset_frac),
        "treat_budget_as_collapse": int(args.treat_budget_as_collapse),

        # Schemas
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

    # legacy collapse fields (compatible with vSAVE scripts)
    collapse_run = 0
    collapse_reason = "NONE"

    # core collapse fields (PI/REP only) for vNEXT analysis
    core_collapse_run = 0
    core_collapse_reason = "NONE"

    budget_exhausted = 0

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

        # π
        pi_raw = float(ses.hybrid_complexity(text, cfg))
        pi_norm = float(norm_q50_q95(pi_raw, q50, q95))

        # REP (char n-gram)
        rep = float(repetition_score(text, n=int(args.rep_ngram), tail_chars=int(args.rep_tail_chars)))

        pi_flag = 1 if (pi_norm > float(args.epsilon_pi)) else 0
        rep_flag = 1 if (rep > float(args.rep_threshold)) else 0
        too_short_flag = 1 if (global_generated_tokens < int(args.min_tokens)) else 0

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
        if mode_core == "baseline" and collapse_streak >= int(args.k_consecutive):
            core_collapse_run = 1
            if pi_flag:
                core_collapse_reason = "PI"
            elif rep_flag:
                core_collapse_reason = "REP"
            else:
                core_collapse_reason = "NONE"

            # legacy fields follow core for baseline
            collapse_run = 1
            collapse_reason = core_collapse_reason

        # controlled intervention rule
        if mode_core == "controlled" and collapse_streak >= int(args.k_consecutive):
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
                # budget exhausted: always report separately; legacy collapse is optional
                budget_exhausted = 1
                # terminal core collapse when intervention budget cannot recover PI/REP
                core_collapse_run = 1
                if pi_flag:
                    core_collapse_reason = "PI"
                elif rep_flag:
                    core_collapse_reason = "REP"
                else:
                    core_collapse_reason = "NONE"
                if int(args.treat_budget_as_collapse) == 1:
                    collapse_run = 1
                    collapse_reason = "BUDGET_EXCEEDED"
                # core collapse remains PI/REP only (no change)

        rows.append({
            "record_type": "step",
            "step_idx": int(step),

            "token_idx_ctx": int(token_idx_ctx),
            "token_idx_ctx_pre": int(token_idx_ctx_pre),
            "token_idx_ctx_post": int(token_idx_ctx_post) if token_idx_ctx_post is not None else None,

            "token_idx_global": int(token_idx_global),
            "generated_tokens": int(generated_tokens),

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

        # stop if legacy collapse triggered OR budget exhausted (when treat_budget_as_collapse=0 we still stop)
        if collapse_run == 1 or budget_exhausted == 1 or core_collapse_run == 1:
            break

        # --- normal decode ---
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn)
            logits = out.logits[0, -1, :]

        # optional ngram block (token-level) + EOS suppression until min_tokens
        banned = set()
        if int(args.ngram_block) and int(args.ngram_block) > 0:
            banned = banned_tokens_ngram(input_ids[0].tolist(), int(args.ngram_block))
        if eos_id is not None and global_generated_tokens < int(args.min_tokens):
            banned.add(int(eos_id))

        nxt = sample_next_token(
            logits,
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            top_k=int(args.top_k),
            banned_token_ids=banned
        )

        nxt_t = torch.tensor([[int(nxt)]], device=device, dtype=torch.long)
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

    # --- HARD GUARANTEE: TOO_SHORT must be collapse in ALL modes ---
    # Use generated token count (not prompt+generated).
    too_short_final = int(global_generated_tokens < int(args.min_tokens))
    # --- taxonomy: core vs aux ---
    stop_aux = 0
    stop_aux_reason = "NONE"
    if too_short_final == 1:
        # legacy (compat): TOO_SHORT is a collapse
        collapse_run = 1
        collapse_reason = "TOO_SHORT"
        # aux taxonomy: TOO_SHORT belongs to stop_aux, not core
        stop_aux = 1
        stop_aux_reason = "TOO_SHORT"
        # core should be PI/REP only
        core_collapse_run = 0
        core_collapse_reason = "NONE"
    elif budget_exhausted == 1:
        stop_aux = 1
        stop_aux_reason = "BUDGET_EXHAUSTED"

    # If controlled ended without legacy collapse, only map core->legacy when budget counts as collapse.
    if mode_core == "controlled" and core_collapse_run == 1 and collapse_run == 0:
        if int(args.treat_budget_as_collapse) == 1:
            collapse_run = 1
            collapse_reason = "BUDGET_EXCEEDED"
        # else keep legacy collapse_run=0; rely on budget_exhausted + core_* for analysis

    final = {
        "record_type": "final",
        "timestamp_end": timestamp_end,

        "H_pre": H_pre if (mode_core == "controlled") else None,

        "n_tokens_total": int(final_total_len),
        "n_interventions": int(interventions) if (mode_core == "controlled") else 0,

        # legacy
        "collapse_run": int(collapse_run),
        "collapse_reason": str(collapse_reason),

        # core (PI/REP only)
        "core_collapse_run": int(core_collapse_run),
        "core_collapse_reason": str(core_collapse_reason),

        # budget separated
        "budget_exhausted": int(budget_exhausted),

        # taxonomy (new)
        "collapse_core": int(core_collapse_run),
        "collapse_core_reason": str(core_collapse_reason),
        "stop_aux": int(stop_aux),
        "stop_aux_reason": str(stop_aux_reason),

        "runtime_sec": float(t1 - t0),
        "extra_tokens_due_to_intervention": float(extra_tokens_due_to_intervention) if (mode_core == "controlled") else 0.0,
    }
    rows.append(final)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(
        f"[run_vnext_with_seed_v5] wrote: {out_path} "
        f"mode={mode_original} core={mode_core} "
        f"steps={len(rows)-2} collapse_run={final['collapse_run']} "
        f"core_collapse_run={final['core_collapse_run']} "
        f"budget_exhausted={final['budget_exhausted']} "
        f"reason={final['collapse_reason']} interventions={final['n_interventions']}"
    )


if __name__ == "__main__":
    main()
