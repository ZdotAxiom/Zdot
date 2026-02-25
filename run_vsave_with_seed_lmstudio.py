import argparse, json, random, time
from pathlib import Path

import numpy as np
import requests


# ----------------------------
# Utils (seed, prompt blocks)
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)




def count_units(text: str) -> int:
    # robust length for JP/EN (no whitespace assumption)
    if text is None:
        return 0
    return len(text.strip())

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


# ----------------------------
# Minimal π / repetition / short detection
# (LM Studio sanity check mode)
# ----------------------------
def gzip_complexity_proxy(text: str) -> float:
    # extremely lightweight proxy:
    # bigger = more "complex / irregular"
    import gzip
    raw = text.encode("utf-8", errors="ignore")
    if len(raw) == 0:
        return 0.0
    comp = gzip.compress(raw)
    return float(len(comp)) / float(len(raw))


def rep_ngram_score(text: str, n: int = 3, tail_words: int = 250) -> float:
    # repetition score in recent window (word-based)
    words = text.replace("\n", " ").split()
    if len(words) < n + 10:
        return 0.0
    w = words[-tail_words:]
    grams = [" ".join(w[i:i+n]) for i in range(0, max(0, len(w) - n + 1))]
    if not grams:
        return 0.0
    from collections import Counter
    c = Counter(grams)
    total = sum(c.values())
    rep = sum(v for v in c.values() if v >= 2)
    return float(rep) / float(max(1, total))


def too_short_flag(word_count: int, min_tokens: int) -> int:
    return 1 if word_count < int(min_tokens) else 0


def safe_preview(text: str, k: int = 160) -> str:
    return text[:k].replace("\n", " ")


# ----------------------------
# LM Studio OpenAI-compatible client
# ----------------------------
def wait_for_server(base_url: str, timeout_sec: int = 60):
    t0 = time.time()
    last_err = None
    while time.time() - t0 < timeout_sec:
        try:
            r = requests.get(f"{base_url}/models", timeout=3)
            if r.status_code == 200:
                return True, r.json()
        except Exception as e:
            last_err = e
        time.sleep(1.0)
    return False, last_err


def call_chat_completion(base_url: str, model: str, messages, max_tokens: int,
                         temperature: float, top_p: float, seed: int | None = None,
                         timeout: int = 120):
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": False,
    }
    # LM Studio sometimes supports seed
    if seed is not None:
        payload["seed"] = int(seed)

    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    # OpenAI style: choices[0].message.content
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        # fallback: choices[0].text
        return data["choices"][0].get("text", "")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_id", type=str, default="vSAVE-1.1")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--mode", choices=["baseline", "controlled"], required=True)
    ap.add_argument("--pillar", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)

    # LM Studio
    ap.add_argument("--base_url", type=str, default="http://127.0.0.1:1234/v1")
    ap.add_argument("--model_id", type=str, default="openai/gpt-oss-20b")

    # decoding
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--chunk_tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)

    # collapse rule (paper-fixed)
    ap.add_argument("--epsilon_pi", type=float, default=0.85)
    ap.add_argument("--k_consecutive", type=int, default=3)

    # additional detectors (v2 style)
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--rep_ngram", type=int, default=3)
    ap.add_argument("--rep_threshold", type=float, default=0.20)

    # intervention (controlled only)
    ap.add_argument("--max_interventions", type=int, default=5)
    ap.add_argument("--reset_frac", type=float, default=0.70)

    # server wait
    ap.add_argument("--server_wait_sec", type=int, default=60)

    args = ap.parse_args()
    set_seed(args.seed)

    # --- wait server ---
    ok, info = wait_for_server(args.base_url, timeout_sec=int(args.server_wait_sec))
    if not ok:
        raise RuntimeError(
            f"LM Studio server not reachable at {args.base_url} "
            f"(waited {args.server_wait_sec}s). "
            f"Enable Local LLM Service and ensure Running.\nLast error: {info}"
        )

    # --- calib ---
    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    if "complexity" not in calib or "q50" not in calib["complexity"] or "q95" not in calib["complexity"]:
        raise RuntimeError("calib.json must contain calib['complexity']['q50'] and ['q95']")
    q50 = float(calib["complexity"]["q50"])
    q95 = float(calib["complexity"]["q95"])

    # --- prompts ---
    prompts = load_blocks(args.prompts)
    if not prompts:
        raise RuntimeError("prompts file has no blocks (split by blank lines)")
    prompt_id = int(args.seed % len(prompts))
    prompt = prompts[prompt_id]

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
        "lmstudio_models": info.get("data", []) if isinstance(info, dict) else None,
        "base_url": args.base_url,
    }

    rows = [meta]

    generated = ""
    interventions = 0
    consec = 0
    collapse_run = 0
    collapse_reason = "NONE"
    H_pre = None
    extra_tokens_due_to_intervention = 0.0

    # Build constraints for intervention
    constraints = (
        "\n\n[CONSTRAINTS]\n"
        "- Continue coherently.\n"
        "- Avoid repeating the same phrase.\n"
        "- Keep structure step-by-step.\n"
        "- If stuck, summarize and proceed.\n"
    )

    # Chunk-based loop
    steps = 0
    total_budget = int(args.max_new_tokens)
    chunk = max(1, int(args.chunk_tokens))

    while len(generated.split()) < total_budget:
        # current text
        text = (prompt + "\n\n" + generated).strip()
        word_count = len(text.split())

        # π + detectors
        pi_raw = float(gzip_complexity_proxy(text))
        pi_norm = float(norm_q50_q95(pi_raw, q50, q95))
        pi_flag = 1 if (pi_norm > float(args.epsilon_pi)) else 0

        rep_score = float(rep_ngram_score(text, n=int(args.rep_ngram)))
        rep_flag = 1 if (rep_score > float(args.rep_threshold)) else 0

        ts_flag = too_short_flag(word_count, int(args.min_tokens))

        collapse_flag = 1 if (pi_flag or rep_flag or ts_flag) else 0
        consec = (consec + 1) if collapse_flag else 0

        intervened = 0
        intervention_type = "NONE"

        # controlled intervention
        if args.mode == "controlled" and consec >= int(args.k_consecutive):
            if interventions < int(args.max_interventions):
                interventions += 1
                intervened = 1
                intervention_type = "RESET_CONSTRAINTS"
                if H_pre is None:
                    H_pre = int(steps)

                # reset: keep some prefix of generated and inject constraints
                keep_words = int(max(0, len(generated.split()) * float(args.reset_frac)))
                kept = " ".join(generated.split()[:keep_words])
                generated = (kept + constraints).strip()

                consec = 0
                collapse_flag = 0
            else:
                collapse_run = 1
                collapse_reason = "MAX_INTERVENTIONS_EXCEEDED"
                break

        # log step
        rows.append({
            "record_type": "step",
            "step_idx": int(steps),
            "token_idx": int(word_count),
            "pi_raw": float(pi_raw),
            "pi_norm": float(pi_norm),
            "pi_flag": int(pi_flag),
            "rep_score": float(rep_score),
            "rep_flag": int(rep_flag),
            "too_short_flag": int(ts_flag),
            "collapse_flag": int(collapse_flag),
            "intervened": int(intervened),
            "intervention_type": str(intervention_type),
            "text_preview": safe_preview(text),
        })

        if collapse_run == 1:
            break

        # request next chunk
        remaining = total_budget - len((prompt + " " + generated).split())
        if remaining <= 0:
            break
        max_out = min(chunk, remaining)

        # LM Studio safe conversation packing:
        # Some local OpenAI-compatible servers return empty text if we pass assistant history.
        # So we pack the whole context into ONE user message.
        packed = prompt
        if generated.strip():
            packed = (prompt + "\n\n[SO FAR]\n" + generated).strip()
        messages = [{"role": "user", "content": packed}]

        try:
            out_text = call_chat_completion(
                args.base_url,
                args.model_id,
                messages,
                max_tokens=max_out,
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                seed=int(args.seed),
                timeout=180,
            )
        except requests.exceptions.ConnectionError as e:
            # Most common: server not yet ready / sleeping
            raise RuntimeError(
                f"ConnectionError to {args.base_url}. "
                f"Check LM Studio Local LLM Service is Running.\n{e}"
            )
        except Exception as e:
            raise RuntimeError(f"LM Studio call failed: {e}")

        # If model returns empty, treat as collapse-ish end
        if out_text is None:
            out_text = ""
        out_text = str(out_text)
        if out_text.strip() == "":
            # short / dead response
            collapse_run = 1
            collapse_reason = "EMPTY_GENERATION"
            break

        generated = (generated + out_text).strip()
        steps += 1

        # run-level collapse by consecutive flags
        if consec >= int(args.k_consecutive):
            collapse_run = 1
            collapse_reason = "K_CONSECUTIVE_FLAGS"
            break

    t1 = time.time()

    final = {
        "record_type": "final",
        "H_pre": H_pre if (args.mode == "controlled") else None,
        "n_tokens_total": int(count_units(generated)),
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

    print(f"[lmstudio] wrote: {out_path} steps={len(rows)-2} collapse_run={collapse_run} reason={collapse_reason} n_interventions={final['n_interventions']}")


if __name__ == "__main__":
    main()
