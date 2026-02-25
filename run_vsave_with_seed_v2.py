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
        s = float(filt.sum().item())
        if s <= 0:
            pick = torch.multinomial(probs, 1)
            return int(pick.item())
        filt = filt / filt.sum()
        pick = torch.multinomial(filt, 1)
        return int(sorted_idx[pick].item())

    pick = torch.multinomial(probs, 1)
    return int(pick.item())


def repetition_score(text: str, n: int = 3, tail_chars: int = 1200) -> float:
    """
    Simple n-gram repetition score on the tail of text.
    score in [0,1]: higher = more repetitive.
    """
    if not text:
        return 0.0
    t = text[-tail_chars:]
    # tokenize coarsely by whitespace to keep it language-agnostic
    toks = [x for x in t.replace("\n", " ").split(" ") if x]
    if len(toks) < n + 3:
        return 0.0
    ngrams = [tuple(toks[i:i+n]) for i in range(len(toks) - n + 1)]
    if not ngrams:
        return 0.0
    c = Counter(ngrams)
    total = len(ngrams)
    mx = max(c.values())
    # normalized: if all unique -> ~1/total, if one dominates -> high
    return clamp01(mx / max(1, total))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", type=str, default="vSAVE-1.1", help="experiment id (fixed for paper)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["baseline", "controlled"], required=True)
    ap.add_argument("--pillar", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--model_id", type=str, default="gpt2")
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    # paper-fixed collapse rule (pi)
    ap.add_argument("--epsilon_pi", type=float, default=0.85)
    ap.add_argument("--k_consecutive", type=int, default=3)

    # NEW: early EOS / too-short handling (NeurIPS-friendly)
    ap.add_argument("--min_tokens", type=int, default=64, help="if final n_tokens_total < min_tokens => collapse_run=1 (fixed rule)")

    # NEW: repetition collapse auxiliary detector (Appendix)
    ap.add_argument("--rep_ngram", type=int, default=3)
    ap.add_argument("--rep_threshold", type=float, default=0.20, help="if repetition_score > threshold for k steps => collapse_by_rep=1")
    ap.add_argument("--rep_tail_chars", type=int, default=1200)

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

        # v2 add-ons (kept deterministic)
        "min_tokens": int(args.min_tokens),
        "rep_ngram": int(args.rep_ngram),
        "rep_threshold": float(args.rep_threshold),
        "rep_tail_chars": int(args.rep_tail_chars),
    }

    rows = [meta]

    interventions = 0
    consec_pi = 0
    consec_rep = 0

    collapse_run = 0
    collapse_by_pi = 0
    collapse_by_rep = 0
    collapse_by_short = 0
    collapse_reason = "NONE"

    H_pre = None
    extra_tokens_due_to_intervention = 0.0

    cfg = ses.SESConfig()

    def decode(ids):
        return tok.decode(ids[0].tolist(), skip_special_tokens=True)

    for step in range(int(args.max_new_tokens)):
        text = ses._safe_truncate(decode(input_ids), 4000)

        # π = complexity_norm
        pi_raw = float(ses.hybrid_complexity(text, cfg))
        pi_norm = float(norm_q50_q95(pi_raw, q50, q95))

        # repetition auxiliary (Appendix)
        rep = float(repetition_score(text, n=int(args.rep_ngram), tail_chars=int(args.rep_tail_chars)))

        pi_flag = 1 if (pi_norm > float(args.epsilon_pi)) else 0
        rep_flag = 1 if (rep > float(args.rep_threshold)) else 0

        consec_pi = (consec_pi + 1) if pi_flag else 0
        consec_rep = (consec_rep + 1) if rep_flag else 0

        intervened = 0
        intervention_type = "NONE"

        # baseline collapse rule: if sustained pi OR sustained rep => collapse
        if args.mode == "baseline":
            if consec_pi >= int(args.k_consecutive):
                collapse_run = 1
                collapse_by_pi = 1
                collapse_reason = "PI_K_CONSEC"
            elif consec_rep >= int(args.k_consecutive):
                collapse_run = 1
                collapse_by_rep = 1
                collapse_reason = "REP_K_CONSEC"

        # controlled intervention (first) when pi streak hits k
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

                # reset streaks after intervention
                consec_pi = 0
                consec_rep = 0
                pi_flag = 0
                rep_flag = 0
            else:
                collapse_run = 1
                collapse_by_pi = 1
                collapse_reason = "PI_K_CONSEC_MAXINT"

        rows.append({
            "record_type": "step",
            "step_idx": int(step),
            "token_idx": int(input_ids.shape[1]),
            "pi_raw": float(pi_raw),
            "pi_norm": float(pi_norm),
            "pi_flag": int(pi_flag),
            "rep_score": float(rep),
            "rep_flag": int(rep_flag),
            "intervened": int(intervened),
            "intervention_type": str(intervention_type),
            "text_preview": text[:160].replace("\n", " "),
        })

        if collapse_run == 1:
            break

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

        if eos_id is not None and int(nxt) == int(eos_id):
            break

    t1 = time.time()

    # early EOS / too-short fixed rule (NeurIPS-friendly)
    n_tokens_total = int(input_ids.shape[1])
    if n_tokens_total < int(args.min_tokens):
        collapse_by_short = 1
        collapse_run = 1
        if collapse_reason == "NONE":
            collapse_reason = "TOO_SHORT"

    final = {
        "record_type": "final",
        "H_pre": H_pre if (args.mode == "controlled") else None,
        "n_tokens_total": n_tokens_total,
        "n_interventions": int(interventions) if (args.mode == "controlled") else 0,

        # unified collapse + reasons
        "collapse_run": int(collapse_run),
        "collapse_by_pi": int(collapse_by_pi),
        "collapse_by_rep": int(collapse_by_rep),
        "collapse_by_short": int(collapse_by_short),
        "collapse_reason": str(collapse_reason),

        "runtime_sec": float(t1 - t0),
        "extra_tokens_due_to_intervention": float(extra_tokens_due_to_intervention) if (args.mode == "controlled") else 0.0,
    }
    rows.append(final)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[run_vsave_with_seed_v2] wrote: {out_path} steps={len(rows)-2} collapse_run={final['collapse_run']} n_interventions={final['n_interventions']} reason={final['collapse_reason']}")


if __name__ == "__main__":
    main()
