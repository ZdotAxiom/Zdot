#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
import pandas as pd

SEED_FILE_RE = re.compile(r"^seed_(\d+)\.jsonl$")  # accept seed_1.jsonl and seed_001.jsonl

def read_jsonl(path: Path):
    meta = None
    final = None
    steps = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            obj=json.loads(line)
            rt=obj.get("record_type","")
            if rt=="meta" and meta is None:
                meta=obj
            elif rt=="final":
                final=obj
            elif rt=="step":
                steps += 1
    return meta, final, steps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--expect", type=int, default=1, help="expected seeds per (pillar,mode)")
    ap.add_argument("--modes", nargs="*", default=["baseline","controlled"], help="modes to include")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Accept seed_1.jsonl and seed_001.jsonl
    files = sorted([p for p in runs_root.glob("*/*/*.jsonl") if SEED_FILE_RE.match(p.name)])
    if not files:
        raise SystemExit(f"[ERROR] no seed_*.jsonl under {runs_root}")

    rows=[]
    for p in files:
        mode = p.parent.parent.name
        pillar = p.parent.name
        if mode not in set(args.modes):
            continue

        meta, final, steps = read_jsonl(p)
        if meta is None or final is None:
            raise SystemExit(f"[ERROR] meta/final missing: {p}")

        m = SEED_FILE_RE.match(p.name)
        seed_from_name = int(m.group(1)) if m else -1
        seed_val = meta.get("seed", None)
        try:
            seed = int(seed_val) if seed_val is not None else seed_from_name
        except Exception:
            seed = seed_from_name
        seed_source = "meta" if seed_val is not None else "file_fallback"

        model_id = meta.get("model_id", meta.get("ollama_model", meta.get("model","UNKNOWN")))
        exp_id = meta.get("exp_id", meta.get("experiment_id","vNEXT-UNKNOWN"))
        prompt_id = int(meta.get("prompt_id", seed % 40))  # fallback: 40 blocks

        n_tokens_total = int(final.get("n_tokens_total", 0))
        n_interventions = int(final.get("n_interventions", 0))
        intervene_bin = 1 if n_interventions > 0 else 0
        intervene_density = (n_interventions / n_tokens_total) if n_tokens_total > 0 else 0.0

        H_pre = final.get("H_pre", None)

        row = dict(
            exp_id=str(exp_id),
            mode=str(mode),
            pillar=str(pillar),
            seed=int(seed),
            seed_source=str(seed_source),
            model_id=str(model_id),
            prompt_id=int(prompt_id),
            collapse_run=int(final.get("collapse_run", 0)),
            collapse_reason=str(final.get("collapse_reason","NONE")),
            core_collapse_run=int(final.get("core_collapse_run", 0)),
            core_collapse_reason=str(final.get("core_collapse_reason","NONE")),
            budget_exhausted=int(final.get("budget_exhausted", 0)),
            collapse_core=int(final.get("collapse_core", final.get("core_collapse_run", 0))),
            collapse_core_reason=str(final.get("collapse_core_reason", final.get("core_collapse_reason", "NONE"))),
            stop_aux=int(final.get("stop_aux", 0)),
            stop_aux_reason=str(final.get("stop_aux_reason", "NONE")),
            intervene_bin=int(intervene_bin),
            n_tokens_total=int(n_tokens_total),
            n_interventions=int(n_interventions),
            intervene_density=float(intervene_density),
            H_pre=H_pre,
            runtime_sec=float(final.get("runtime_sec", 0.0)),
            extra_tokens_due_to_intervention=float(final.get("extra_tokens_due_to_intervention", 0.0)),
        )
        rows.append(row)

    df=pd.DataFrame(rows)

    # ---- validate counts ----
    expect=args.expect
    for pillar in sorted(df["pillar"].unique().tolist()):
        for mode in args.modes:
            sub=df[(df["pillar"]==pillar) & (df["mode"]==mode)]
            if len(sub)!=expect:
                raise AssertionError(f"pillar={pillar} mode={mode}: expect {expect}, got {len(sub)}")

    out_csv = outdir / "runs_summary_all.csv"
    df.sort_values(["pillar","mode","seed"]).to_csv(out_csv, index=False)
    print(f"[vNEXT validate] wrote: {out_csv} rows={len(df)} modes={args.modes}")

if __name__ == "__main__":
    main()
