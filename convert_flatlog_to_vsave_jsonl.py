#!/usr/bin/env python3
from __future__ import annotations

import argparse, json
from pathlib import Path
import pandas as pd

REQ_META_KEYS = ["exp_id","mode","pillar","seed","model_id","decoder","prompt_id"]
REQ_STEP_KEYS = ["step_idx","token_idx","pi_raw","pi_norm","collapse_flag",
                 "intervened","intervention_type","text_preview"]
REQ_FINAL_KEYS = ["H_pre","n_tokens_total","n_interventions","collapse_run",
                  "runtime_sec","extra_tokens_due_to_intervention"]

def first_nonempty_line(p: Path) -> str:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                return s
    return ""

def smart_read_table(p: Path) -> pd.DataFrame:
    # .jsonl でも中身がCSV/TSVのことがあるので自動で読む
    # まずCSVとして読む→ダメならTSV
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.read_csv(p, sep="\t")

def pick_col(df: pd.DataFrame, candidates: list[str], default=None):
    for c in candidates:
        if c in df.columns:
            return c
    return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--outfile", required=True)
    ap.add_argument("--mode", required=True, choices=["baseline","controlled"])
    ap.add_argument("--pillar", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model_id", default="unknown")
    ap.add_argument("--decoder", default="unknown")
    ap.add_argument("--prompt_id", type=int, default=0)
    args = ap.parse_args()

    infile = Path(args.infile)
    outfile = Path(args.outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)

    # 既にvSAVE形式ならそのままコピー（安全策）
    first = first_nonempty_line(infile)
    if first.startswith("{") and '"record_type"' in first:
        outfile.write_text(infile.read_text(encoding="utf-8"), encoding="utf-8")
        print("[convert] already jsonl:", outfile)
        return

    df = smart_read_table(infile)

    # 必須情報の列推定
    col_token = pick_col(df, ["token_idx","token","t","tok"], None)
    col_step  = pick_col(df, ["step_idx","step","s"], None)
    col_pi_raw = pick_col(df, ["pi_raw","pi"], None)
    col_pi_norm = pick_col(df, ["pi_norm","pi01","pi_normalized"], None)
    col_cflag = pick_col(df, ["collapse_flag","collapse","is_collapse"], None)
    col_intv = pick_col(df, ["intervened","did_intervene","intervention"], None)
    col_itype = pick_col(df, ["intervention_type","type"], None)
    col_prev = pick_col(df, ["text_preview","preview"], None)
    col_runtime = pick_col(df, ["runtime_sec","runtime_s","sec"], None)
    col_Hpre = pick_col(df, ["H_pre","entropy_pre"], None)
    col_extra = pick_col(df, ["extra_tokens_due_to_intervention","extra_tokens"], None)

    # token/step が無ければ行番号から作る
    if col_token is None:
        df["token_idx__"] = range(len(df))
        col_token = "token_idx__"
    if col_step is None:
        df["step_idx__"] = range(len(df))
        col_step = "step_idx__"

    # 介入/崩壊のデフォルト整形
    if col_intv is None:
        df["intervened__"] = 0
        col_intv = "intervened__"
    if col_cflag is None:
        df["collapse_flag__"] = 0
        col_cflag = "collapse_flag__"
    if col_pi_raw is None:
        df["pi_raw__"] = 0.0
        col_pi_raw = "pi_raw__"
    if col_pi_norm is None:
        df["pi_norm__"] = 0.0
        col_pi_norm = "pi_norm__"
    if col_itype is None:
        df["intervention_type__"] = "none"
        col_itype = "intervention_type__"
    if col_prev is None:
        df["text_preview__"] = ""
        col_prev = "text_preview__"

    # meta
    exp_id = f"{args.pillar}_{args.mode}_seed{args.seed:03d}"
    meta = dict(
        record_type="meta",
        exp_id=exp_id,
        mode=args.mode,
        pillar=args.pillar,
        seed=int(args.seed),
        model_id=args.model_id,
        decoder=args.decoder,
        prompt_id=int(args.prompt_id),
    )

    # steps
    steps = []
    for _, r in df.iterrows():
        step = dict(
            record_type="step",
            step_idx=int(r[col_step]),
            token_idx=int(r[col_token]),
            pi_raw=float(r[col_pi_raw]),
            pi_norm=float(r[col_pi_norm]),
            collapse_flag=int(r[col_cflag]),
            intervened=int(r[col_intv]),
            intervention_type=str(r[col_itype]),
            text_preview=str(r[col_prev]),
        )
        steps.append(step)

    # final
    n_tokens_total = int(df[col_token].max()) + 1 if len(df) else 0
    n_interventions = int(pd.Series(df[col_intv]).astype(int).sum()) if len(df) else 0
    collapse_run = int(pd.Series(df[col_cflag]).astype(int).max()) if len(df) else 0

    final = dict(
        record_type="final",
        H_pre=float(df[col_Hpre].iloc[-1]) if col_Hpre in df.columns else 0.0,
        n_tokens_total=n_tokens_total,
        n_interventions=n_interventions,
        collapse_run=int(collapse_run),
        runtime_sec=float(df[col_runtime].iloc[-1]) if col_runtime in df.columns else 0.0,
        extra_tokens_due_to_intervention=float(df[col_extra].iloc[-1]) if col_extra in df.columns else 0.0,
    )

    # write jsonl
    with outfile.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for s in steps:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
        f.write(json.dumps(final, ensure_ascii=False) + "\n")

    print("[convert] wrote:", outfile, "steps=", len(steps))

if __name__ == "__main__":
    main()
