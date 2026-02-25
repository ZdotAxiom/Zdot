#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

CAUSE_ORDER = ["PI", "REP", "TOO_SHORT", "HIT_MAX", "BUDGET_EXCEEDED", "OTHER", "NONE"]

def read_meta_final(jsonl_path: Path):
    meta = None
    final = None
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rt = obj.get("record_type")
            if rt == "meta" and meta is None:
                meta = obj
            elif rt == "final":
                final = obj
    if meta is None or final is None:
        raise RuntimeError(f"meta/final not found: {jsonl_path}")
    return meta, final

def derive_cause(meta, final, min_tokens_default=64):
    prompt_len = int(meta.get("prompt_len_tokens", 0))
    dec = meta.get("decoder", {}) or {}
    max_new = int(dec.get("max_new_tokens", 512))
    min_tokens = int(meta.get("min_tokens", min_tokens_default))

    n_total = int(final.get("n_tokens_total", 0))
    gen = max(0, n_total - prompt_len)

    collapse_run = int(final.get("collapse_run", 0))
    collapse_reason = str(final.get("collapse_reason", "NONE"))

    # Exclusive assignment
    if gen < min_tokens:
        cause = "TOO_SHORT"
    elif gen >= max_new:
        cause = "HIT_MAX"
    elif collapse_run == 1:
        if collapse_reason in ("PI", "REP", "BUDGET_EXCEEDED"):
            cause = collapse_reason
        elif collapse_reason == "NONE":
            cause = "OTHER"
        else:
            cause = "OTHER"
    else:
        cause = "NONE"
    return cause

def rr_or(a, nA, c, nC):
    # Haldane-Anscombe correction
    a2 = a + 0.5
    b2 = (nA - a) + 0.5
    c2 = c + 0.5
    d2 = (nC - c) + 0.5
    rr = (a2 / (a2 + b2)) / (c2 / (c2 + d2))
    orv = (a2 * d2) / (b2 * c2)
    return rr, orv

def ci(x, alpha=0.05):
    lo = np.quantile(x, alpha/2)
    hi = np.quantile(x, 1-alpha/2)
    return lo, hi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_tokens_default", type=int, default=64)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    runs_root = Path(args.runs_root)

    # Collect per (pillar, mode) -> causes by seed index
    data = {}  # (pillar, mode) -> dict seed->cause
    for mode in ["baseline", "controlled"]:
        base = runs_root / mode
        if not base.exists():
            raise RuntimeError(f"missing dir: {base}")
        for pillar_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            pillar = pillar_dir.name
            for fp in pillar_dir.glob("seed_*.jsonl"):
                meta, final = read_meta_final(fp)
                seed = int(meta.get("seed", -1))
                cause = derive_cause(meta, final, min_tokens_default=args.min_tokens_default)
                data.setdefault((pillar, mode), {})[seed] = cause

    pillars = sorted({k[0] for k in data.keys()})
    out_rows = []
    B = int(args.bootstrap)

    for pillar in pillars:
        bmap = data.get((pillar, "baseline"), {})
        cmap = data.get((pillar, "controlled"), {})

        # ensure same seed set intersection
        seeds = sorted(set(bmap.keys()) & set(cmap.keys()))
        if len(seeds) == 0:
            continue

        b = np.array([bmap[s] for s in seeds], dtype=object)
        c = np.array([cmap[s] for s in seeds], dtype=object)
        nb = len(b); nc = len(c)

        # Pre-generate bootstrap index matrix for speed
        if B > 0:
            idx_b = rng.integers(0, nb, size=(B, nb))
            idx_c = rng.integers(0, nc, size=(B, nc))

        for cause in CAUSE_ORDER:
            ev_b = int(np.sum(b == cause))
            ev_c = int(np.sum(c == cause))
            rate_b = ev_b / nb
            rate_c = ev_c / nc
            delta = rate_c - rate_b
            rr, orv = rr_or(ev_c, nc, ev_b, nb)

            if B > 0:
                # counts for each bootstrap resample
                ev_bi = np.sum(b[idx_b] == cause, axis=1).astype(np.float64)
                ev_ci = np.sum(c[idx_c] == cause, axis=1).astype(np.float64)
                rb = ev_bi / nb
                rc = ev_ci / nc
                deltas = rc - rb

                # RR/OR arrays
                rr_arr = np.empty(B, dtype=np.float64)
                or_arr = np.empty(B, dtype=np.float64)
                # vectorized correction
                a2 = ev_ci + 0.5
                b2 = (nc - ev_ci) + 0.5
                c2 = ev_bi + 0.5
                d2 = (nb - ev_bi) + 0.5
                rr_arr = (a2/(a2+b2)) / (c2/(c2+d2))
                or_arr = (a2*d2) / (b2*c2)

                d_lo, d_hi = ci(deltas)
                rr_lo, rr_hi = ci(rr_arr)
                or_lo, or_hi = ci(or_arr)
            else:
                d_lo=d_hi=rr_lo=rr_hi=or_lo=or_hi=np.nan

            out_rows.append({
                "pillar": pillar,
                "cause": cause,
                "n_baseline": nb,
                "n_controlled": nc,
                "rate_baseline": rate_b,
                "rate_controlled": rate_c,
                "delta_controlled_minus_baseline": delta,
                "delta_ci95_lo": d_lo,
                "delta_ci95_hi": d_hi,
                "RR_controlled_vs_baseline": rr,
                "RR_ci95_lo": rr_lo,
                "RR_ci95_hi": rr_hi,
                "OR_controlled_vs_baseline": orv,
                "OR_ci95_lo": or_lo,
                "OR_ci95_hi": or_hi,
            })

    out = pd.DataFrame(out_rows)
    out["cause"] = pd.Categorical(out["cause"], categories=CAUSE_ORDER, ordered=True)
    out = out.sort_values(["pillar", "cause"]).reset_index(drop=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # Markdown
    def fmt_rate(x): return f"{100*x:.2f}%" if pd.notnull(x) else "NA"
    def fmt_delta(x): return f"{100*x:+.2f}pp" if pd.notnull(x) else "NA"
    def fmt_ci_pp(lo, hi):
        if pd.isnull(lo) or pd.isnull(hi): return "NA"
        return f"[{100*lo:+.2f}, {100*hi:+.2f}]pp"
    def fmt_ci(lo, hi):
        if pd.isnull(lo) or pd.isnull(hi): return "NA"
        return f"[{lo:.3f}, {hi:.3f}]"
    def fmt(x): return f"{x:.3f}" if pd.notnull(x) else "NA"

    md = []
    md.append("# Table 1b: Collapse Breakdown by Cause (Baseline vs Controlled)\n")
    md.append("Exclusive cause assignment priority: TOO_SHORT → HIT_MAX → collapse_reason (PI/REP/BUDGET_EXCEEDED/OTHER) → NONE.\n")
    md.append(f"Bootstrap: seed-level resampling, B={B}, seed={args.seed}.\n")

    for pillar in sorted(out["pillar"].unique()):
        md.append(f"\n## {pillar}\n")
        md.append("| cause | baseline | controlled | Δ (pp) | 95% CI (Δ) | RR | 95% CI (RR) | OR | 95% CI (OR) |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        sub = out[out["pillar"] == pillar]
        for _, r in sub.iterrows():
            md.append(
                f"| {r['cause']} | {fmt_rate(r['rate_baseline'])} | {fmt_rate(r['rate_controlled'])} | "
                f"{fmt_delta(r['delta_controlled_minus_baseline'])} | {fmt_ci_pp(r['delta_ci95_lo'], r['delta_ci95_hi'])} | "
                f"{fmt(r['RR_controlled_vs_baseline'])} | {fmt_ci(r['RR_ci95_lo'], r['RR_ci95_hi'])} | "
                f"{fmt(r['OR_controlled_vs_baseline'])} | {fmt_ci(r['OR_ci95_lo'], r['OR_ci95_hi'])} |"
            )

    Path(args.out_md).write_text("\n".join(md), encoding="utf-8")

    print(f"[OK] wrote: {args.out_csv}")
    print(f"[OK] wrote: {args.out_md}")
    print("[INFO] rows:", len(out))

if __name__ == "__main__":
    main()
