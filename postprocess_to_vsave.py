import argparse
import json
from pathlib import Path

import pandas as pd


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def q50_q95_norm(x: float, q50: float, q95: float) -> float:
    # 1-page π: q50/q95 scaling -> [0,1]
    denom = (q95 - q50)
    if denom <= 1e-12:
        return 0.0
    return clamp01((x - q50) / denom)


def load_calib(calib_path: Path) -> dict:
    if not calib_path.exists():
        return {}
    with calib_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_qs(calib: dict, key_candidates: list[str]):
    # Accept multiple possible schemas safely
    # - calib["metrics"]["complexity"]["q50"], ["q95"]
    # - calib["complexity"]["q50"], ["q95"]
    for k in key_candidates:
        # metrics nested
        if isinstance(calib.get("metrics"), dict) and isinstance(calib["metrics"].get(k), dict):
            d = calib["metrics"][k]
            if "q50" in d and "q95" in d:
                return float(d["q50"]), float(d["q95"])
        # flat
        if isinstance(calib.get(k), dict) and "q50" in calib[k] and "q95" in calib[k]:
            return float(calib[k]["q50"]), float(calib[k]["q95"])
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--mode", required=True)      # baseline/controlled
    ap.add_argument("--pillar", required=True)    # HUM/STEM/ETH/TEMP/META
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--calib", required=True)     # calib/calib.json

    # ===== Spec-fixed params (paper) =====
    ap.add_argument("--epsilon_pi", type=float, default=0.85)
    ap.add_argument("--k_consecutive", type=int, default=3)
    ap.add_argument("--preview_chars", type=int, default=200)

    # Which raw column to treat as pi_raw base
    ap.add_argument("--pi_source", type=str, default="complexity",
                    help="raw column used as pi_raw (default: complexity)")

    args = ap.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    calib = load_calib(Path(args.calib))

    # Read CSV (run_lm flatlog)
    df = pd.read_csv(infile)

    # find text column
    text_col = None
    for c in df.columns:
        if c.lower() == "text":
            text_col = c
            break

    # determine pi_source column
    pi_col = None
    if args.pi_source in df.columns:
        pi_col = args.pi_source
    else:
        # fallback: try known candidates
        for cand in ["complexity", "gz", "ent", "pi", "pi_raw"]:
            if cand in df.columns:
                pi_col = cand
                break
    if pi_col is None:
        raise RuntimeError(f"No usable pi source column found. columns={list(df.columns)}")

    # q50/q95 from calib for this metric (or fallback identity)
    q50, q95 = get_qs(calib, [pi_col, "complexity", "gz", "ent"])
    use_norm = (q50 is not None and q95 is not None)

    meta = {
        "record_type": "meta",
        "mode": args.mode,
        "pillar": args.pillar,
        "seed": int(args.seed),
        "pi_source": pi_col,
        "epsilon_pi": float(args.epsilon_pi),
        "k_consecutive": int(args.k_consecutive),
        "preview_chars": int(args.preview_chars),
        "calib_used": bool(use_norm),
        "csv_columns": list(df.columns),
    }
    if use_norm:
        meta["q50"] = float(q50)
        meta["q95"] = float(q95)

    steps = []
    n_interventions = 0
    first_intervene_tok = None

    consec = 0
    collapse_run = 0

    for i, r in df.iterrows():
        step_idx = int(r["step"]) if "step" in df.columns else int(i)
        token_idx = step_idx

        text = ""
        if text_col is not None and isinstance(r[text_col], str):
            text = r[text_col]
        text_preview = text[: args.preview_chars] if text else ""

        pi_raw = float(r[pi_col])
        pi_norm = q50_q95_norm(pi_raw, q50, q95) if use_norm else float(pi_raw)

        # intervention
        intervened = int(r["intervene"]) if "intervene" in df.columns else 0
        if intervened == 1:
            n_interventions += 1
            if first_intervene_tok is None:
                first_intervene_tok = token_idx

        itype = "NONE"
        if intervened == 1:
            if "action" in df.columns and isinstance(r["action"], str) and r["action"].strip():
                itype = r["action"].strip().upper()
            else:
                itype = "INTERVENE"

        # collapse by fixed definition (paper)
        cflag = 1 if (pi_norm > float(args.epsilon_pi)) else 0

        if cflag == 1:
            consec += 1
        else:
            consec = 0
        if consec >= int(args.k_consecutive):
            collapse_run = 1

        steps.append({
            "record_type": "step",
            "step_idx": step_idx,
            "token_idx": token_idx,
            "pi_raw": pi_raw,
            "pi_norm": pi_norm,
            "collapse_flag": int(cflag),
            "intervened": int(intervened),
            "intervention_type": itype if intervened else "NONE",
            "text_preview": text_preview,
        })

    final = {
        "record_type": "final",
        "H_pre": first_intervene_tok,
        "n_tokens_total": int(len(steps)),
        "n_interventions": int(n_interventions) if args.mode == "controlled" else 0,
        "collapse_run": int(collapse_run),
        "runtime_sec": None,
        "extra_tokens_due_to_intervention": 0.0 if args.mode == "baseline" else None,
    }

    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_lines = [json.dumps(meta, ensure_ascii=False)]
    out_lines += [json.dumps(o, ensure_ascii=False) for o in steps]
    out_lines += [json.dumps(final, ensure_ascii=False)]
    outfile.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"[postprocess_to_vsave] wrote: {outfile} steps={len(steps)} collapse_run={collapse_run}")

if __name__ == "__main__":
    main()
