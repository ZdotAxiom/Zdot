#!/usr/bin/env python3
# validate_and_summarize.py
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd

REQ_META_KEYS = [
    "exp_id", "mode", "pillar", "seed", "model_id", "device", "decoder",
    "timestamp_start", "prompt_id", "prompt_len_tokens",
    "pi_source", "q50", "q95", "epsilon_pi", "k_consecutive",
    "min_tokens", "rep_ngram", "rep_threshold",
    "ctx_limit", "token_idx_schema", "too_short_schema",
]
REQ_STEP_KEYS = [
    "step_idx", "token_idx_ctx", "token_idx_global", "generated_tokens",
    "pi_raw", "pi_norm", "pi_flag",
    "rep_score", "rep_flag",
    "too_short_flag", "collapse_flag",
    "intervened", "intervention_type", "text_preview",
]
REQ_FINAL_KEYS = [
    "timestamp_end", "collapse_run", "collapse_reason",
    "H_pre", "n_tokens_total", "n_interventions",
    "runtime_sec", "extra_tokens_due_to_intervention",
]

def is_nan(x: Any) -> bool:
    return isinstance(x, float) and math.isnan(x)

def assert_no_nan(d: Dict[str, Any], context: str) -> None:
    for k, v in d.items():
        if is_nan(v):
            raise AssertionError(f"NaN detected: {context} key={k}")

def load_jsonl(p: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    meta = None
    final = None
    steps: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rt = obj.get("record_type")
            if rt not in ("meta", "step", "final"):
                raise AssertionError(f"{p} line {line_no}: bad record_type={rt}")
            assert_no_nan(obj, f"{p} line {line_no}")
            if rt == "meta":
                meta = obj
            elif rt == "step":
                steps.append(obj)
            else:
                final = obj

    if meta is None:
        raise AssertionError(f"{p}: missing meta")
    if final is None:
        raise AssertionError(f"{p}: missing final")
    if len(steps) == 0:
        raise AssertionError(f"{p}: no steps")
    return meta, steps, final

def validate_run(meta: Dict[str, Any], steps: List[Dict[str, Any]], final: Dict[str, Any], path: Path) -> None:
    for k in REQ_META_KEYS:
        if k not in meta:
            raise AssertionError(f"{path}: meta missing {k}")
    for k in REQ_FINAL_KEYS:
        if k not in final:
            raise AssertionError(f"{path}: final missing {k}")
    for s in steps:
        for k in REQ_STEP_KEYS:
            if k not in s:
                raise AssertionError(f"{path}: step missing {k}")

    mode = meta["mode"]
    if mode not in ("controlled", "baseline"):
        raise AssertionError(f"{path}: invalid mode={mode}")
    if not isinstance(meta["seed"], int):
        raise AssertionError(f"{path}: seed not int")

    token_idxs = [int(s["token_idx_global"]) for s in steps]
    if not all(t2 >= t1 for t1, t2 in zip(token_idxs, token_idxs[1:])):
        raise AssertionError(f"{path}: token_idx_global not monotonic")

    for s in steps:
        pn = float(s["pi_norm"])
        if not (0.0 <= pn <= 1.0):
            raise AssertionError(f"{path}: pi_norm out of range: {pn}")
        if int(s["generated_tokens"]) != int(s["token_idx_global"]) - int(meta["prompt_len_tokens"]):
            raise AssertionError(f"{path}: generated_tokens mismatch")

    if mode == "baseline":
        if int(final["n_interventions"]) != 0:
            raise AssertionError(f"{path}: baseline has interventions")
        if float(final["extra_tokens_due_to_intervention"]) != 0.0:
            raise AssertionError(f"{path}: baseline extra_tokens != 0")
        for s in steps:
            if int(s["intervened"]) != 0:
                raise AssertionError(f"{path}: baseline step intervened=1")
            if s["intervention_type"] != "NONE":
                raise AssertionError(f"{path}: baseline intervention_type != NONE")
        if final["H_pre"] is not None:
            raise AssertionError(f"{path}: baseline H_pre must be null")

def summarize_run(meta: Dict[str, Any], final: Dict[str, Any]) -> Dict[str, Any]:
    collapse_run = int(final["collapse_run"])
    n_interventions = int(final["n_interventions"])
    intervene_bin = 1 if n_interventions > 0 else 0

    n_tokens_total = int(final["n_tokens_total"])
    intervene_density = (n_interventions / max(1, n_tokens_total))

    return {
        "exp_id": meta["exp_id"],
        "mode": meta["mode"],
        "pillar": meta["pillar"],
        "seed": meta["seed"],
        "model_id": meta["model_id"],
        "prompt_id": meta["prompt_id"],
        "collapse_run": collapse_run,
        "intervene_bin": intervene_bin,
        "n_tokens_total": n_tokens_total,
        "n_interventions": n_interventions,
        "intervene_density": intervene_density,
        "H_pre": (final["H_pre"] if final["H_pre"] is not None else "NONE"),
        "runtime_sec": float(final["runtime_sec"]),
        "extra_tokens_due_to_intervention": float(final["extra_tokens_due_to_intervention"]),
    }

def run_validation_and_summary(runs_root: Path, out_csv: Path, expect_per_pillar: int | None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    seen_keys = set()

    for p in sorted(runs_root.rglob("seed_*.jsonl")):
        meta, steps, final = load_jsonl(p)
        validate_run(meta, steps, final, p)

        key = (meta["exp_id"], meta["mode"], meta["pillar"], meta["seed"])
        if key in seen_keys:
            raise AssertionError(f"duplicate run key: {key} at {p}")
        seen_keys.add(key)

        rows.append(summarize_run(meta, final))

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise AssertionError(f"no runs found under {runs_root}")

    if df.isna().any().any():
        raise AssertionError("NaN found in runs_summary (use null/NONE instead)")

    if expect_per_pillar is not None:
        grp = df.groupby(["mode", "pillar"]).size().reset_index(name="n")
        bad = grp[grp["n"] != expect_per_pillar]
        if len(bad) > 0:
            raise AssertionError(f"count mismatch per (mode,pillar):\n{bad}")

    df.to_csv(out_csv, index=False)
    return df

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True, help="exp/runs (contains controlled/ and baseline/)")
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--expect", type=int, default=500, help="expected runs per (mode,pillar); use 3 for smoke")
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_csv = outdir / "runs_summary_all.csv"
    df = run_validation_and_summary(runs_root, out_csv, expect_per_pillar=args.expect)
    print("OK:", out_csv, "rows=", len(df))
