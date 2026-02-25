import argparse, json, random, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import z_collapse_lab_SES_v2_4_2_no_text_fix as ses


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


def sample_top_p(logits, temperature: float, top_p: float):
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


def ngram_repetition_score(token_ids, n=4, window=200):
    """
    Return repetition score in [0,1]:
    0 => all n-grams unique
    1 => all n-grams identical (extremely repetitive)
    """
    ids = token_ids[-window:] if len(token_ids) > window else token_ids[:]
    if len(ids) < n + 1:
        return 0.0
    grams = [tuple(ids[i:i+n]) for i in range(len(ids) - n + 1)]
    total = len(grams)
    uniq = len(set(grams))
    return float(total - uniq) / float(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", type=str, default="vSAVE-1.1")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["baseline", "controlled"], required=True)
    ap.add_argument("--pillar", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--model_id", type=str, default="gpt2")
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    # paper-fixed collapse rule (pi)
    ap.add_argument("--epsilon_pi", type=float, default=0.85)
    ap.add_argument("--k_consecutive", type=int, default=3)

    # repetition detector (rep_flag)
    ap.add_argument("--rep_ngram_n", type=int, default=4)
    ap.add_argument("--rep_window", type=int, default=200)
    ap.add_argument("--rep_threshold", type=float, default=0.35)

    # TOO_SHORT detector (early EOS)
    ap.add_argument("--min_generated_tokens", type=int, default=20)

    # paper-fixed intervention budget (controlled only)
    ap.add_argument("--max_interventions", type=int, default=5)
    ap.add_argument("--reset_frac", type=float, default=0.70)

    args = ap.parse_args()
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

    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tok))

    ctx_limit = 1024
    eos_id = tok.eos_token_id

    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=ctx_limit)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(device)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    meta = {
        "record_type": "meta",
        "exp_id": args.exp_id,
        "mode": args.mode,
        "pillar": args.pillar,
        "seed": int(args.seed),
        "model_id": args.model_id,
        "decoder": {
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "max_new_tokens": int(args.max_new_tokens),
        },
        "prompt_id": int(prompt_id),

        # π definition (1-page fixed)
        "pi_source": "complexity",
        "calib_used": True,
        "q50": q50,
        "q95": q95,
        "epsilon_pi": float(args.epsilon_pi),
        "k_consecutive": int(args.k_consecutive),

        # rep / too_short rules (for collapse_flag completeness)
        "rep_ngram_n": int(args.rep_ngram_n),
        "rep_window": int(args.rep_window),
        "rep_threshold": float(args.rep_threshold),
        "min_generated_tokens": int(args.min_generated_tokens),
    }

    rows = [meta]

    interventions = 0
    consec_pi = 0
    collapse_run = 0
    collapse_reason = "NONE"
    H_pre = None
    extra_tokens_due_to_intervention = 0.0

    cfg = ses.SESConfig()

    def decode(ids):
        return tok.decode(ids[0].tolist(), skip_special_tokens=True)

    start_len = int(input_ids.shape[1])

    for step in range(int(args.max_new_tokens)):
        text = ses._safe_truncate(decode(input_ids), 4000)

        # --- π ---
        pi_raw = float(ses.hybrid_complexity(text, cfg))
        pi_norm = float(norm_q50_q95(pi_raw, q50, q95))
        pi_flag = 1 if (pi_norm > float(args.epsilon_pi)) else 0
        consec_pi = (consec_pi + 1) if pi_flag else 0

        # --- repetition ---
        token_list = input_ids[0].tolist()
        rep_score = float(ngram_repetition_score(token_list,
                                                 n=int(args.rep_ngram_n),
                                                 window=int(args.rep_window)))
        rep_flag = 1 if (rep_score >= float(args.rep_threshold)) else 0

        # --- TOO_SHORT (early stop detection handled at final too, but we can flag early) ---
        gen_len = int(input_ids.shape[1]) - start_len
        too_short_flag = 1 if (gen_len < int(args.min_generated_tokens) and step > 3) else 0

        # --- collapse_flag (FINAL rule, patch不要) ---
        collapse_flag = 1 if (pi_flag or rep_flag or too_short_flag) else 0

        intervened = 0
        intervention_type = "NONE"

        # --- intervention FIRST (controlled only) ---
        if args.mode == "controlled" and consec_pi >= int(args.k_consecutive):
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
                    H_pre = int(step)

                # after intervention, reset streak
                consec_pi = 0

                # recompute flags post-reset is expensive; log as pre-reset snapshot
            else:
                collapse_run = 1
                collapse_reason = "PI_SUSTAINED"
        else:
            # if baseline OR controlled but not intervening, allow rep/too_short to trigger run collapse
            if rep_flag == 1:
                collapse_run = 1
                collapse_reason = "REPETITION"
            # too_short is finalized at EOS, but allow early collapse if it's clearly dying
            # (we keep it conservative here)

        rows.append({
            "record_type": "step",
            "step_idx": int(step),
            "token_idx": int(input_ids.shape[1]),
            "pi_raw": float(pi_raw),
            "pi_norm": float(pi_norm),
            "collapse_flag": int(collapse_flag),

            # extras (safe)
            "pi_flag": int(pi_flag),
            "rep_score": float(rep_score),
            "rep_flag": int(rep_flag),
            "too_short_flag": int(too_short_flag),

            "intervened": int(intervened),
            "intervention_type": str(intervention_type),
            "text_preview": text[:160].replace("\n", " "),
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

        if input_ids.shape[1] > ctx_limit:
            input_ids = input_ids[:, -ctx_limit:]
            attn = attn[:, -ctx_limit:]

        # EOS handling
        if eos_id is not None and int(nxt) == int(eos_id):
            break

    t1 = time.time()

    # finalize TOO_SHORT after generation ends
    final_gen_len = int(input_ids.shape[1]) - start_len
    if collapse_run == 0 and final_gen_len < int(args.min_generated_tokens):
        collapse_run = 1
        collapse_reason = "TOO_SHORT"

    final = {
        "record_type": "final",
        "H_pre": H_pre if (args.mode == "controlled") else None,
        "n_tokens_total": int(input_ids.shape[1]),
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

    print(f"[run_vsave_with_seed_v3] wrote: {out_path} steps={len(rows)-2} collapse_run={collapse_run} reason={collapse_reason} n_interventions={final['n_interventions']}")


if __name__ == "__main__":
    main()
