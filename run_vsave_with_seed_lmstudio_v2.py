import argparse, json, time
from pathlib import Path
import requests

# --- utilities ---
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

def safe_preview(s: str, n: int = 160):
    return (s or "").replace("\n", " ")[:n]

def count_units(text: str) -> int:
    # JP/EN safe length (not whitespace-dependent)
    return len((text or "").strip())

def extract_content(data: dict) -> str:
    """
    LM Studio / OpenAI compatible:
    - choices[0].message.content
    - choices[0].text (fallback)
    """
    choices = data.get("choices", [])
    if not choices:
        return ""
    c0 = choices[0] or {}
    msg = c0.get("message")
    if isinstance(msg, dict):
        out = msg.get("content", "")
        if isinstance(out, str):
            return out
        # Newer style: content can be a list of {"type":"text","text":...}
        if isinstance(out, list):
            texts = []
            for it in out:
                if isinstance(it, dict) and "text" in it:
                    texts.append(str(it["text"]))
            return "".join(texts)
    out = c0.get("text", "")
    return out if isinstance(out, str) else ""


def build_url(base_url: str) -> str:
    b = base_url.rstrip("/")
    if b.endswith("/chat/completions"):
        return b
    return b + "/chat/completions"


def lm_chat(base_url: str, model: str, user_text: str, max_tokens: int, temperature: float, top_p: float, timeout: int = 120) -> str:
    url = build_url(base_url)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "max_tokens": int(max_tokens),     # LM Studio uses max_tokens
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    data = r.json()
    return extract_content(data)

# --- simple collapse signals (lightweight) ---
def rep_score_ngram(text: str, n: int = 3) -> float:
    s = (text or "").strip()
    if len(s) < n:
        return 0.0
    grams = [s[i:i+n] for i in range(len(s) - n + 1)]
    if not grams:
        return 0.0
    uniq = len(set(grams))
    return 1.0 - (uniq / max(1, len(grams)))

def gzip_proxy_complexity(text: str) -> float:
    # simple gzip proxy; enough for "計器生存チェック"
    import gzip
    b = (text or "").encode("utf-8")
    if not b:
        return 0.0
    c = gzip.compress(b)
    return len(c) / max(1, len(b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", type=str, default="vSAVE-1.1")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["baseline", "controlled"], required=True)
    ap.add_argument("--pillar", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    ap.add_argument("--base_url", type=str, required=True)         # e.g. http://127.0.0.1:1234/v1
    ap.add_argument("--model_id", type=str, required=True)         # e.g. openai/gpt-oss-20b

    ap.add_argument("--max_new_tokens", type=int, default=128)     # total loop steps
    ap.add_argument("--chunk_tokens", type=int, default=16)        # each API call max_tokens
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--top_p", type=float, default=0.95)

    ap.add_argument("--epsilon_pi", type=float, default=0.85)
    ap.add_argument("--k_consecutive", type=int, default=3)
    ap.add_argument("--min_tokens", type=int, default=64)          # treated as min_chars for LM sanity
    ap.add_argument("--rep_ngram", type=int, default=3)
    ap.add_argument("--rep_threshold", type=float, default=0.20)

    args = ap.parse_args()

    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    q50 = float(calib["complexity"]["q50"])
    q95 = float(calib["complexity"]["q95"])

    blocks = load_blocks(args.prompts)
    if not blocks:
        raise RuntimeError("prompts file has no blocks (split by blank lines)")
    prompt_id = int(args.seed % len(blocks))
    prompt = blocks[prompt_id]

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
            "chunk_tokens": int(args.chunk_tokens),
        },
        "prompt_id": int(prompt_id),
        "pi_source": "gzip_proxy_complexity",
        "calib_used": True,
        "q50": q50,
        "q95": q95,
        "epsilon_pi": float(args.epsilon_pi),
        "k_consecutive": int(args.k_consecutive),
        "min_tokens": int(args.min_tokens),
        "rep_ngram": int(args.rep_ngram),
        "rep_threshold": float(args.rep_threshold),
        "base_url": args.base_url,
    }

    rows = [meta]

    generated = ""
    consec = 0
    collapse_run = 0
    collapse_reason = "NONE"

    # NOTE: this is "計器生存チェック用"なので介入は実装しない（baselineで十分）
    for step in range(int(args.max_new_tokens)):
        packed = prompt if not generated.strip() else (prompt + "\n\n[SO FAR]\n" + generated).strip()

        # call LM Studio
        delta_raw = lm_chat(
            base_url=args.base_url,
            model=args.model_id,
            user_text=packed,
            max_tokens=int(args.chunk_tokens),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            timeout=120,
        )
        print('[DEBUG] delta_raw_len=', len(delta_raw), 'repr=', repr(delta_raw[:80]))
        delta = delta_raw

        if len(delta) == 0:
            collapse_run = 1
            collapse_reason = "EMPTY_GENERATION"
            # still write a step row for debugging
            pi_raw = gzip_proxy_complexity(packed)
            pi_norm = norm_q50_q95(pi_raw, q50, q95)
            rep = rep_score_ngram(generated, int(args.rep_ngram))
            too_short = 1 if (count_units(generated) < int(args.min_tokens)) else 0
            pi_flag = 1 if (pi_norm > float(args.epsilon_pi)) else 0
            rep_flag = 1 if (rep > float(args.rep_threshold)) else 0
            collapse_flag = 1 if (pi_flag or rep_flag or too_short) else 0
            rows.append({
                "record_type": "step",
                "step_idx": int(step),
                "token_idx": int(step+1),
                "pi_raw": float(pi_raw),
                "pi_norm": float(pi_norm),
                "pi_flag": int(pi_flag),
                "rep_score": float(rep),
                "rep_flag": int(rep_flag),
                "too_short_flag": int(too_short),
                "collapse_flag": int(collapse_flag),
                "intervened": 0,
                "intervention_type": "NONE",
                "text_preview": safe_preview(packed),
            })
            break

        generated = (generated + ("\n" if generated else "") + delta).strip()

        # compute signals on current text
        pi_raw = gzip_proxy_complexity(prompt + "\n\n" + generated)
        pi_norm = norm_q50_q95(pi_raw, q50, q95)
        pi_flag = 1 if (pi_norm > float(args.epsilon_pi)) else 0

        rep = rep_score_ngram(generated, int(args.rep_ngram))
        rep_flag = 1 if (rep > float(args.rep_threshold)) else 0

        too_short = 1 if (count_units(generated) < int(args.min_tokens)) else 0

        # step collapse flag (unified)
        collapse_flag = 1 if (pi_flag or rep_flag or too_short) else 0
        consec = (consec + 1) if pi_flag else 0

        # run-level collapse by sustained pi
        if consec >= int(args.k_consecutive):
            collapse_run = 1
            collapse_reason = "PI_K_CONSECUTIVE"

        rows.append({
            "record_type": "step",
            "step_idx": int(step),
            "token_idx": int(step+1),
            "pi_raw": float(pi_raw),
            "pi_norm": float(pi_norm),
            "pi_flag": int(pi_flag),
            "rep_score": float(rep),
            "rep_flag": int(rep_flag),
            "too_short_flag": int(too_short),
            "collapse_flag": int(collapse_flag),
            "intervened": 0,
            "intervention_type": "NONE",
            "text_preview": safe_preview(generated),
        })

        if collapse_run == 1:
            break

    t1 = time.time()

    final = {
        "record_type": "final",
        "H_pre": None,
        "n_tokens_total": int(count_units(generated)),
        "n_interventions": 0,
        "collapse_run": int(collapse_run),
        "collapse_reason": str(collapse_reason),
        "runtime_sec": float(t1 - t0),
        "extra_tokens_due_to_intervention": 0.0,
    }
    rows.append(final)

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[lmstudio_v2] wrote: {out_path} steps={len(rows)-2} collapse_run={collapse_run} reason={collapse_reason}")

if __name__ == "__main__":
    main()
