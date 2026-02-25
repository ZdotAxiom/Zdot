import argparse, json, random, time
from pathlib import Path
import numpy as np
import requests

import z_collapse_lab_SES_v2_4_2_no_text_fix as ses


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


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


def safe_preview(s: str, n=160) -> str:
    s = (s or "").replace("\n", " ")
    return s[:n]


def repetition_score(text: str, n: int = 3) -> float:
    # シンプルな n-gram 反復率（0..1）
    # 1 - (unique_ngrams / total_ngrams)
    tokens = (text or "").split()
    if len(tokens) < max(n * 2, 8):
        return 0.0
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngrams.append(tuple(tokens[i:i+n]))
    total = len(ngrams)
    if total <= 0:
        return 0.0
    uniq = len(set(ngrams))
    return max(0.0, min(1.0, 1.0 - (uniq / total)))


def count_units(text: str) -> int:
    # LM Studioはtoken countを返さないことが多いので雑に文字数ベース（run-levelで一貫していればOK）
    return len(text or "")


def chat_completion(base_url: str, model: str, prompt: str, max_tokens: int, temperature: float, top_p: float, timeout: int = 120):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", type=str, default="vSAVE-1.1")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["baseline", "controlled"], required=True)
    ap.add_argument("--pillar", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--base_url", type=str, required=True)
    ap.add_argument("--model_id", type=str, required=True)

    ap.add_argument("--max_new_tokens", type=int, default=256)   # 総生成“予算”
    ap.add_argument("--chunk_tokens", type=int, default=64)      # 1 stepで生成させる量

    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top_p", type=float, default=0.95)

    # collapse rule
    ap.add_argument("--epsilon_pi", type=float, default=0.85)
    ap.add_argument("--k_consecutive", type=int, default=3)

    # v2系（短文/反復）
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--rep_ngram", type=int, default=3)
    ap.add_argument("--rep_threshold", type=float, default=0.20)

    # controlled only
    ap.add_argument("--max_interventions", type=int, default=5)
    ap.add_argument("--empty_retry", type=int, default=2, help="controlled: retry on EMPTY_GENERATION")
    ap.add_argument("--reset_frac", type=float, default=0.70)

    args = ap.parse_args()
    set_seed(args.seed)

    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    if "complexity" not in calib:
        raise RuntimeError("calib.json must contain calib['complexity'] with q50/q95")
    q50 = float(calib["complexity"]["q50"])
    q95 = float(calib["complexity"]["q95"])

    prompts = load_blocks(args.prompts)
    if not prompts:
        raise RuntimeError("prompts file has no blocks (split by blank lines)")
    prompt_id = int(args.seed % len(prompts))
    prompt0 = prompts[prompt_id]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = ses.SESConfig()

    rows = []
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
            "chunk_tokens": int(args.chunk_tokens),
        },
        "prompt_id": int(prompt_id),

        "pi_source": "complexity",
        "calib_used": True,
        "q50": q50,
        "q95": q95,
        "epsilon_pi": float(args.epsilon_pi),
        "k_consecutive": int(args.k_consecutive),

        "min_tokens": int(args.min_tokens),
        "rep_ngram": int(args.rep_ngram),
        "rep_threshold": float(args.rep_threshold),
    }
    rows.append(meta)

    t0 = time.time()

    generated = ""
    packed = prompt0
    consec = 0
    collapse_run = 0
    collapse_reason = "NONE"
    interventions = 0
    H_pre = None

    total_budget = int(args.max_new_tokens)
    step_budget = int(args.chunk_tokens)

    # 重要：max_new_tokens “総量”まで回す。v4のようなdelta短いbreakはしない。
    step = 0
    while count_units(generated) < total_budget:
        # ---- metrics init (必ず定義) ----
        pi_raw = 0.0
        pi_norm = 0.0
        pi_flag = 0
        rep = 0.0
        rep_flag = 0
        too_short = 0
        collapse_flag = 0
        intervened = 0
        intervention_type = "NONE"

        # ---- measure ----
        text_for_pi = ses._safe_truncate(packed, 4000)
        pi_raw = float(ses.hybrid_complexity(text_for_pi, cfg))
        pi_norm = float(norm_q50_q95(pi_raw, q50, q95))
        pi_flag = 1 if pi_norm > float(args.epsilon_pi) else 0

        rep = float(repetition_score(text_for_pi, n=int(args.rep_ngram)))
        rep_flag = 1 if rep > float(args.rep_threshold) else 0

        too_short = 1 if count_units(generated) < int(args.min_tokens) else 0

        # collapse_flag（step-level）
        collapse_flag = 1 if (pi_flag or rep_flag or too_short) else 0
        consec = consec + 1 if collapse_flag else 0

        # ---- controlled intervention ----
        if args.mode == "controlled" and consec >= int(args.k_consecutive):
            if interventions < int(args.max_interventions):
                # 決定的介入：RESET_CONSTRAINTS
                packed = ses._reset_with_constraints(packed, frac=float(args.reset_frac))
                packed = ses._safe_truncate(packed, 4000)
                interventions += 1
                intervened = 1
                intervention_type = "RESET_CONSTRAINTS"
                if H_pre is None:
                    H_pre = int(step)
                consec = 0
                # 介入したステップは “崩壊として確定させない”
                collapse_flag = 0
            else:
                collapse_run = 1
                collapse_reason = "MAX_INTERVENTIONS_EXCEEDED"
                # ここで止める
                rows.append({
                    "record_type": "step",
                    "step_idx": int(step),
                    "token_idx": int(count_units(generated)),
                    "pi_raw": float(pi_raw),
                    "pi_norm": float(pi_norm),
                    "pi_flag": int(pi_flag),
                    "rep_score": float(rep),
                    "rep_flag": int(rep_flag),
                    "too_short_flag": int(too_short),
                    "collapse_flag": int(1),
                    "intervened": int(intervened),
                    "intervention_type": str(intervention_type),
                    "text_preview": safe_preview(generated),
                })
                break

        # ---- generate a chunk ----
        try:
            chunk = chat_completion(
                base_url=args.base_url,
                model=args.model_id,
                prompt=packed,
                max_tokens=step_budget,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        except Exception as e:
            # LM Studio応答失敗＝collapse
            collapse_run = 1
            collapse_reason = f"API_ERROR:{type(e).__name__}"
            chunk = ""

        if not chunk.strip():
            # EMPTY_GENERATION を collapse として扱う（NaN禁止）
            pi_raw = 0.0
            pi_norm = 1.0
            pi_flag = 1
            rep = 0.0
            rep_flag = 0
            too_short = 1
            collapse_flag = 1
            collapse_run = 1
            collapse_reason = "EMPTY_GENERATION"

        # 追記
        generated = (generated + "\n" + chunk).strip()
        packed = (prompt0 + "\n\n" + generated).strip()

        rows.append({
            "record_type": "step",
            "step_idx": int(step),
            "token_idx": int(count_units(generated)),
            "pi_raw": float(pi_raw),
            "pi_norm": float(pi_norm),
            "pi_flag": int(pi_flag),
            "rep_score": float(rep),
            "rep_flag": int(rep_flag),
            "too_short_flag": int(too_short),
            "collapse_flag": int(collapse_flag),
            "intervened": int(intervened),
            "intervention_type": str(intervention_type),
            "text_preview": safe_preview(generated),
        })

        if collapse_run == 1:
            break

        step += 1
        if step > 10000:  # 念のため暴走防止
            collapse_run = 1
            collapse_reason = "STEP_GUARD"
            break

    t1 = time.time()

    final = {
        "record_type": "final",
        "H_pre": H_pre if args.mode == "controlled" else None,
        "n_tokens_total": int(count_units(generated)),
        "n_interventions": int(interventions) if args.mode == "controlled" else 0,
        "collapse_run": int(collapse_run),
        "collapse_reason": str(collapse_reason),
        "runtime_sec": float(t1 - t0),
        "extra_tokens_due_to_intervention": 0.0,
    }
    rows.append(final)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[lmstudio_v5] wrote: {out_path} steps={len(rows)-2} collapse_run={collapse_run} reason={collapse_reason}")


if __name__ == "__main__":
    main()
