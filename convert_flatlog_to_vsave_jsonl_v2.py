import argparse
import json
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--pillar", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--preview_chars", type=int, default=200)

    # ★追加：テスト用に collapse 判定をここで作れる
    ap.add_argument("--epsilon_override", type=float, default=None,
                    help="if set, collapse_flag = 1[pi_norm > epsilon_override]")
    ap.add_argument("--k_consecutive", type=int, default=3,
                    help="k consecutive collapse_flag => collapse_run=1")

    args = ap.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    df = pd.read_csv(infile)

    meta = {
        "record_type": "meta",
        "mode": args.mode,
        "pillar": args.pillar,
        "seed": int(args.seed),
        "csv_columns": list(df.columns),
    }

    steps = []
    n_interventions = 0
    first_intervene_tok = None

    # run-level collapse判定（k連続）
    consec = 0
    collapse_run = 0

    for _, r in df.iterrows():
        step_idx = int(r["step"]) if "step" in df.columns else 0
        token_idx = step_idx

        text = ""
        if "text" in df.columns and isinstance(r["text"], str):
            text = r["text"]
        text_preview = text[: args.preview_chars] if text else ""

        # pi mapping: complexity を pi_raw/pi_norm として扱う
        pi_raw = float(r["complexity"]) if "complexity" in df.columns else 0.0
        pi_norm = pi_raw

        intervened = int(r["intervene"]) if "intervene" in df.columns else 0
        if intervened == 1:
            n_interventions += 1
            if first_intervene_tok is None:
                first_intervene_tok = token_idx

        itype = "NONE"
        if intervened == 1 and "action" in df.columns:
            v = r["action"]
            if isinstance(v, str) and v.strip():
                itype = v.strip().upper()
            else:
                itype = "INTERVENE"

        # ★collapse_flag を決める
        # 1) override があればそれを優先（テスト用）
        # 2) 無ければ CSVの collapse_or_reset を使う（本来の判定）
        if args.epsilon_override is not None:
            cflag = 1 if pi_norm > float(args.epsilon_override) else 0
        else:
            if "collapse_or_reset" in df.columns:
                cflag = int(r["collapse_or_reset"])
            elif "collapse_pred" in df.columns:
                cflag = int(r["collapse_pred"])
            else:
                cflag = 0

        # run-level k連続
        if cflag == 1:
            consec += 1
        else:
            consec = 0
        if consec >= args.k_consecutive:
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
        "n_interventions": int(n_interventions),
        "collapse_run": int(collapse_run),
        "runtime_sec": None,
        "extra_tokens_due_to_intervention": 0.0,
    }

    outfile.parent.mkdir(parents=True, exist_ok=True)
    out_lines = [json.dumps(meta, ensure_ascii=False)]
    out_lines += [json.dumps(o, ensure_ascii=False) for o in steps]
    out_lines += [json.dumps(final, ensure_ascii=False)]
    outfile.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    print(f"[convert_v2] wrote: {outfile} steps={len(steps)} collapse_run={collapse_run}")

if __name__ == "__main__":
    main()
