#!/usr/bin/env python3
# vNEXT Table1 generator
# Reports BOTH:
#  - legacy collapse (collapse_run): "stop by PI/REP in legacy definition"
#  - core collapse (core_collapse_run): "irrecoverable after interventions / budget exhausted, etc."
#
# Output:
#  - table1_vnext.csv
#  - table1_vnext.md
#
# Designed to read: exp_vnext/summary/runs_summary_all.csv
# produced by validate_and_summarize_vnext.py

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import pandas as pd


@dataclass
class CI:
    lo: float
    hi: float

def _pct_ci(x: np.ndarray, lo=2.5, hi=97.5) -> CI:
    if len(x) == 0:
        return CI(np.nan, np.nan)
    return CI(float(np.percentile(x, lo)), float(np.percentile(x, hi)))

def _safe_mean(a: np.ndarray) -> float:
    if a.size == 0:
        return float("nan")
    return float(np.mean(a))

def _rr(events_c: float, n_c: float, events_b: float, n_b: float, eps: float = 1e-12) -> float:
    # Relative Risk = (events_c/n_c)/(events_b/n_b)
    rc = (events_c + eps) / (n_c + eps)
    rb = (events_b + eps) / (n_b + eps)
    return float(rc / rb)

def _or(events_c: float, n_c: float, events_b: float, n_b: float) -> float:
    # Odds Ratio with Haldane-Anscombe correction (+0.5)
    # a=events_c, b=non_events_c, c=events_b, d=non_events_b
    a = events_c
    b = n_c - events_c
    c = events_b
    d = n_b - events_b
    return float(((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5)))

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df

def _coerce_numeric(df: pd.DataFrame, cols: List[str], fillna: Optional[float] = None) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if s.dtype == object:
            s = s.astype(str).str.strip()
        s = pd.to_numeric(s, errors="coerce")
        if fillna is not None:
            s = s.fillna(fillna)
        df[c] = s
    return df

def _align_seed_arrays(df_p: pd.DataFrame, mode_a: str, mode_b: str, col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      seeds (aligned),
      arr_a (mode_a values),
      arr_b (mode_b values)
    Assumes 1 row per (pillar, mode, seed). If duplicates exist, takes mean per seed.
    """
    da = df_p[df_p["mode"] == mode_a].groupby("seed", as_index=False)[col].mean()
    db = df_p[df_p["mode"] == mode_b].groupby("seed", as_index=False)[col].mean()
    merged = pd.merge(da, db, on="seed", how="inner", suffixes=(f"_{mode_a}", f"_{mode_b}"))
    seeds = merged["seed"].to_numpy()
    a = merged[f"{col}_{mode_a}"].to_numpy(dtype=float)
    b = merged[f"{col}_{mode_b}"].to_numpy(dtype=float)
    return seeds, a, b

def _align_seed_arrays_multi(df_p: pd.DataFrame, mode: str, cols: List[str]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    d = df_p[df_p["mode"] == mode].groupby("seed", as_index=False)[cols].mean()
    seeds = d["seed"].to_numpy()
    out = {c: d[c].to_numpy(dtype=float) for c in cols}
    return seeds, out

def _bootstrap_table(
    baseline: np.ndarray,
    controlled: np.ndarray,
    B: int,
    seed: int
) -> Dict[str, np.ndarray]:
    """
    Bootstrap by resampling aligned seeds with replacement.
    Returns arrays for delta, rr, or, baseline_rate, controlled_rate.
    """
    rng = np.random.default_rng(seed)
    n = len(baseline)
    if n == 0:
        return {k: np.array([]) for k in ["delta", "rr", "or", "rb", "rc"]}

    idx = rng.integers(0, n, size=(B, n))

    rb = baseline[idx].mean(axis=1)
    rc = controlled[idx].mean(axis=1)
    delta = rc - rb

    # RR/OR computed from counts (events=sum) rather than mean
    events_b = baseline[idx].sum(axis=1)
    events_c = controlled[idx].sum(axis=1)
    rr = np.array([_rr(events_c[i], n, events_b[i], n) for i in range(B)], dtype=float)
    or_ = np.array([_or(events_c[i], n, events_b[i], n) for i in range(B)], dtype=float)

    return {"delta": delta, "rr": rr, "or": or_, "rb": rb, "rc": rc}

def _fmt_ci(ci: CI, digits=6) -> str:
    if np.isnan(ci.lo) or np.isnan(ci.hi):
        return "[nan, nan]"
    return f"[{ci.lo:.{digits}f}, {ci.hi:.{digits}f}]"

def _to_md_table(df: pd.DataFrame) -> str:
    # Minimal MD table renderer (no external deps)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows) + "\n"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True, help="Path to runs_summary_all.csv (vNEXT)")
    ap.add_argument("--out_csv", required=True, help="Output CSV path")
    ap.add_argument("--out_md", required=True, help="Output Markdown path")
    ap.add_argument("--bootstrap", type=int, default=20000, help="Bootstrap iterations (default=20000)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap")
    ap.add_argument(
        "--controlled_modes",
        nargs="*",
        default=["controlled"],
        help="Controlled mode names to compare against baseline (e.g., controlled100 controlled200)"
    )
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary_csv not found: {summary_path}")

    df = pd.read_csv(summary_path)

    # Expected minimal columns
    needed = [
        "pillar", "mode", "seed",
        "collapse_run", "core_collapse_run", "budget_exhausted",
        "n_interventions"
    ]
    df = _ensure_cols(df, needed)

    # Coerce numeric columns to avoid NaN propagation from stray characters/blank cells
    df = _coerce_numeric(
        df,
        [
            "collapse_run", "core_collapse_run", "budget_exhausted",
            "n_interventions", "n_tokens_total", "runtime_sec",
            "collapse_core", "stop_aux"
        ],
        fillna=0
    )

    # Optional columns used for density/runtime
    if "n_tokens_total" not in df.columns:
        # fall back: if total tokens missing, density becomes NaN
        df["n_tokens_total"] = np.nan
    if "runtime_sec" not in df.columns:
        # some summaries use runtime_total_sec; try to map
        if "runtime_total_sec" in df.columns:
            df["runtime_sec"] = df["runtime_total_sec"]
        else:
            df["runtime_sec"] = np.nan

    # Normalize mode strings just in case
    df["mode"] = df["mode"].astype(str).str.lower()
    df["pillar"] = df["pillar"].astype(str).str.upper()

    pillars = sorted(df["pillar"].unique().tolist())

    rows = []

    for pillar in pillars:
        df_p = df[df["pillar"] == pillar].copy()
        seeds_base, base_cols = _align_seed_arrays_multi(
            df_p, "baseline",
            ["runtime_sec"]
        )

        for mode_controlled in args.controlled_modes:
            # Align seeds baseline vs controlled for legacy collapse
            seeds, legacy_b, legacy_c = _align_seed_arrays(df_p, "baseline", mode_controlled, "collapse_run")
            n = len(seeds)

            # Align seeds baseline vs controlled for core collapse
            _, core_b, core_c = _align_seed_arrays(df_p, "baseline", mode_controlled, "core_collapse_run")

            # Controlled-only arrays for budget/interventions/runtime
            seeds_ctrl, ctrl_cols = _align_seed_arrays_multi(
                df_p, mode_controlled,
                ["budget_exhausted", "n_interventions", "n_tokens_total", "runtime_sec"]
            )

            # Rates
            legacy_rate_b = _safe_mean(legacy_b)
            legacy_rate_c = _safe_mean(legacy_c)
            core_rate_b = _safe_mean(core_b)
            core_rate_c = _safe_mean(core_c)

            budget_rate_c = _safe_mean(ctrl_cols["budget_exhausted"])
            intervene_rate_primary = float(np.mean((ctrl_cols["n_interventions"] > 0).astype(float))) if ctrl_cols["n_interventions"].size > 0 else float("nan")

            # intervention density: mean(n_interventions / n_tokens_total)
            if np.all(np.isnan(ctrl_cols["n_tokens_total"])):
                intervene_density_secondary = float("nan")
            else:
                denom = ctrl_cols["n_tokens_total"]
                num = ctrl_cols["n_interventions"]
                mask = np.isfinite(denom) & (denom > 0)
                intervene_density_secondary = float(np.mean((num[mask] / denom[mask]))) if np.any(mask) else float("nan")

            # runtime delta (controlled - baseline)
            runtime_b = _safe_mean(base_cols["runtime_sec"])
            runtime_c = _safe_mean(ctrl_cols["runtime_sec"])
            delta_runtime = runtime_c - runtime_b if (np.isfinite(runtime_c) and np.isfinite(runtime_b)) else float("nan")

            # Effect sizes (legacy)
            events_lb = float(np.sum(legacy_b))
            events_lc = float(np.sum(legacy_c))
            rr_legacy = _rr(events_lc, n, events_lb, n) if n > 0 else float("nan")
            or_legacy = _or(events_lc, n, events_lb, n) if n > 0 else float("nan")
            delta_legacy = legacy_rate_c - legacy_rate_b

            # Effect sizes (core)
            events_cb = float(np.sum(core_b))
            events_cc = float(np.sum(core_c))
            rr_core = _rr(events_cc, n, events_cb, n) if n > 0 else float("nan")
            or_core = _or(events_cc, n, events_cb, n) if n > 0 else float("nan")
            delta_core = core_rate_c - core_rate_b

            # Bootstrap CIs (paired by aligned seeds)
            boot_legacy = _bootstrap_table(legacy_b, legacy_c, args.bootstrap, args.seed + 17)
            ci_delta_legacy = _pct_ci(boot_legacy["delta"])
            ci_rr_legacy = _pct_ci(boot_legacy["rr"])
            ci_or_legacy = _pct_ci(boot_legacy["or"])

            boot_core = _bootstrap_table(core_b, core_c, args.bootstrap, args.seed + 23)
            ci_delta_core = _pct_ci(boot_core["delta"])
            ci_rr_core = _pct_ci(boot_core["rr"])
            ci_or_core = _pct_ci(boot_core["or"])

            rows.append({
                "pillar": pillar,
                "mode_controlled": mode_controlled,
                "n_seeds": n,

                # legacy collapse
                "legacy_collapse_rate_baseline": round(legacy_rate_b, 6),
                "legacy_collapse_rate_controlled": round(legacy_rate_c, 6),
                "delta_legacy": round(delta_legacy, 6),
                "delta_legacy_95ci": _fmt_ci(ci_delta_legacy, digits=6),
                "RR_legacy": round(rr_legacy, 6),
                "RR_legacy_95ci": _fmt_ci(ci_rr_legacy, digits=6),
                "OR_legacy": round(or_legacy, 6),
                "OR_legacy_95ci": _fmt_ci(ci_or_legacy, digits=6),

                # core collapse
                "core_collapse_rate_baseline": round(core_rate_b, 6),
                "core_collapse_rate_controlled": round(core_rate_c, 6),
                "delta_core": round(delta_core, 6),
                "delta_core_95ci": _fmt_ci(ci_delta_core, digits=6),
                "RR_core": round(rr_core, 6),
                "RR_core_95ci": _fmt_ci(ci_rr_core, digits=6),
                "OR_core": round(or_core, 6),
                "OR_core_95ci": _fmt_ci(ci_or_core, digits=6),

                # budget + intervention + cost
                "budget_exhausted_rate_controlled": round(budget_rate_c, 6),
                "intervene_rate_primary": round(intervene_rate_primary, 6),
                "intervene_density_secondary": round(intervene_density_secondary, 6) if np.isfinite(intervene_density_secondary) else "nan",
                "runtime_sec_baseline": round(runtime_b, 6) if np.isfinite(runtime_b) else "nan",
                "runtime_sec_controlled": round(runtime_c, 6) if np.isfinite(runtime_c) else "nan",
                "delta_runtime_sec": round(delta_runtime, 6) if np.isfinite(delta_runtime) else "nan",
            })

    out_df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_csv, index=False)
    out_md.write_text(_to_md_table(out_df), encoding="utf-8")

    print(f"[make_table1_vnext] wrote: {out_csv}")
    print(f"[make_table1_vnext] wrote: {out_md}")


if __name__ == "__main__":
    main()
