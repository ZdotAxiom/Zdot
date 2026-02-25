#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


CAUSE_ORDER = ["PI", "REP", "TOO_SHORT", "HIT_MAX", "BUDGET_EXCEEDED", "OTHER", "NONE"]


def read_meta_final(jsonl_path: Path):
    """
    Read only the first meta record and the final record from a vSAVE jsonl.
    Returns (meta_dict, final_dict).
    """
    meta = None
    final = None

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rt = obj.get("record_type")
            if rt == "meta" and meta is None:
                meta = obj
            elif rt == "final":
                final = obj  # final is typically the last record, but just keep updating
    if meta is None or final is None:
        raise RuntimeError(f"meta/final not found in {jsonl_path}")
    return meta, final


def derive_cause(meta, final, min_tokens_default=64):
    """
    Derive an exclusive failure cause for each run.

    Priority (exclusive):
      1) TOO_SHORT if generated_tokens < min_tokens
      2) HIT_MAX if generated_tokens >= max_new_tokens (budget saturation)
      3) if collapse_run==1 -> collapse_reason (PI/REP/BUDGET_EXCEEDED/...)
      4) else NONE
    """
    prompt_len = int(meta.get("prompt_len_tokens", 0))
    max_new = int(meta.get("decoder", {}).get("max_new_tokens", 512))

    # min_tokens may not be stored in meta in older logs; use default
    min_tokens = int(meta.get("min_tokens", min_tokens_default))

    n_total = int(final.get("n_tokens_total", 0))
    gen = max(0, n_total - prompt_len)

    collapse_run = int(final.get("collapse_run", 0))
    collapse_reason = str(final.get("collapse_reason", "NONE"))

    # Exclusive cause assignment
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

    return cause, gen


def safe_or(a, b):
    return 1 if (a or b) else 0


def rr_or_ci(a, b, c, d, alpha=0.05):
    """
    a: controlled events
    b: controlled non-events
    c: baseline events
    d: baseline non-events
    Returns RR and OR with Haldane-Anscombe correction.
    """
    # correction
    a2 = a + 0.5
    b2 = b + 0.5
    c2 = c + 0.5
    d2 = d + 0.5

    rr = (a2 / (a2 + b2)) / (c2 / (c2 + d2))
    orv = (a2 * d2) / (b2 * c2)
    return rr, orv


def bootstrap_ci(values, alpha=0.05):
    lo = np.quantile(values, alpha / 2)
    hi = np.quantile(values, 1 - alpha / 2)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, required=True,
                    help="exp/runs directory containing baseline/controlled subdirs")
    ap.add_argument("--out_csv", type=str, required=True)
    ap.add_argument("--out_md", type=str, required=True)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_tokens_default", type=int, default=64)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    runs_root = Path(args.runs_root)

    rows = []

    # iterate expected structure: exp/runs/{baseline,controlled}/{PILLAR}/seed_XXX.jsonl
    for mode_dir in ["baseline", "controlled"]:
        base = runs_root / mode_dir
        if not base.exists():
            raise RuntimeError(f"missing: {base}")

        for pillar_dir in sorted([p for p in base.iterdir() if p.is_dir()]):
            pillar = pillar_dir.name
            for jsonl_path in sorted(pillar_dir.glob("seed_*.jsonl")):
                meta, final = read_meta_final(jsonl_path)
                cause, gen_tokens = derive_cause(meta, final, min_tokens_default=args.min_tokens_default)

                rows.append({
                    "pillar": pillar,
                    "mode": mode_dir,
                    "seed": int(meta.get("seed", -1)),
                    "prompt_id": int(meta.get("prompt_id", -1)),
                    "n_tokens_total": int(final.get("n_tokens_total", 0)),
                    "generated_tokens": int(gen_tokens),
                    "collapse_run": int(final.get("collapse_run", 0)),
                    "collapse_reason": str(final.get("collapse_reason", "NONE")),
                    "cause_exclusive": cause,
                })

    df = pd.DataFrame(rows)

    # sanity
    if df.empty:
        raise RuntimeError("No runs found. Check runs_root structure.")

    # For each pillar and cause, compute baseline vs controlled rates and effect sizes
    out_rows = []

    pillars = sorted(df["pillar"].unique().tolist())

    # Precompute by pillar/mode
    for pillar in pillars:
        dfp = df[df["pillar"] == pillar].copy()

        for cause in CAUSE_ORDER:
            # event = exclusive cause equals this cause
            for mode in ["baseline", "controlled"]:
                sub = dfp[dfp["mode"] == mode]
                n = len(sub)
                ev = int((sub["cause_exclusive"] == cause).sum())
                rate = ev / n if n > 0 else np.nan
                # store later
            # compute effects (controlled vs baseline) on this cause indicator
            b = dfp[dfp["mode"] == "baseline"].copy()
            c = dfp[dfp["mode"] == "controlled"].copy()

            nb = len(b)
            nc = len(c)
            if nb == 0 or nc == 0:
                continue

            ev_b = int((b["cause_exclusive"] == cause).sum())
            ev_c = int((c["cause_exclusive"] == cause).sum())

            rate_b = ev_b / nb
            rate_c = ev_c / nc
            delta = rate_c - rate_b

            # RR / OR with correction
            rr, orv = rr_or_ci(ev_c, nc - ev_c, ev_b, nb - ev_b)

            # Bootstrap CI by resampling seeds within each mode (seed-level independence)
            B = int(args.bootstrap)
            if B > 0:
                # resample indices within each mode
                idx_b = b.index.to_numpy()
                idx_c = c.index.to_numpy()

                deltas = []
                rrs = []
                ors = []

                for _ in range(B):
                    sb = rng.choice(idx_b, size=len(idx_b), replace=True)
                    sc = rng.choice(idx_c, size=len(idx_c), replace=True)

                    ev_bi = int((df.loc[sb, "cause_exclusive"] == cause).sum())
                    ev_ci = int((df.loc[sc, "cause_exclusive"] == cause).sum())

                    rate_bi = ev_bi / len(sb)
                    rate_ci = ev_ci / len(sc)

                    deltas.append(rate_ci - rate_bi)

                    rri, ori = rr_or_ci(ev_ci, len(sc) - ev_ci, ev_bi, len(sb) - ev_bi)
                    rrs.append(rri)
                    ors.append(ori)

                d_lo, d_hi = bootstrap_ci(np.array(deltas))
                rr_lo, rr_hi = bootstrap_ci(np.array(rrs))
                or_lo, or_hi = bootstrap_ci(np.array(ors))
            else:
                d_lo = d_hi = np.nan
                rr_lo = rr_hi = np.nan
                or_lo = or_hi = np.nan

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

    # Stable sorting
    out["cause"] = pd.Categorical(out["cause"], categories=CAUSE_ORDER, ordered=True)
    out = out.sort_values(["pillar", "cause"]).reset_index(drop=True)

    # write CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    # write MD (compact)
    def fmt_rate(x): return f"{100*x:.2f}%" if pd.notnull(x) else "NA"
    def fmt_delta(x): return f"{100*x:+.2f}pp" if pd.notnull(x) else "NA"
    def fmt_ci(lo, hi, pct=False):
        if pd.isnull(lo) or pd.isnull(hi): return "NA"
        if pct:
            return f"[{100*lo:+.2f}, {100*hi:+.2f}]pp"
        return f"[{lo:.3f}, {hi:.3f}]"
    def fmt_rr(x): return f"{x:.3f}" if pd.notnull(x) else "NA"

    md_lines = []
    md_lines.append("# Table 1b: Collapse Breakdown by Cause (Baseline vs Controlled)\n")
    md_lines.append("Exclusive cause assignment priority: TOO_SHORT → HIT_MAX → collapse_reason (PI/REP/BUDGET_EXCEEDED/OTHER) → NONE.\n")
    md_lines.append(f"Bootstrap: seed-level resampling, B={args.bootstrap}, seed={args.seed}.\n")

    for pillar in pillars:
        md_lines.append(f"\n## {pillar}\n")
        md_lines.append("| cause | baseline | controlled | Δ (pp) | 95% CI (Δ) | RR | 95% CI (RR) | OR | 95% CI (OR) |")
        md_lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        sub = out[out["pillar"] == pillar]
        for _, r in sub.iterrows():
            md_lines.append(
                f"| {r['cause']} | {fmt_rate(r['rate_baseline'])} | {fmt_rate(r['rate_controlled'])} | "
                f"{fmt_delta(r['delta_controlled_minus_baseline'])} | {fmt_ci(r['delta_ci95_lo'], r['delta_ci95_hi'], pct=True)} | "
                f"{fmt_rr(r['RR_controlled_vs_baseline'])} | {fmt_ci(r['RR_ci95_lo'], r['RR_ci95_hi'])} | "
                f"{fmt_rr(r['OR_controlled_vs_baseline'])} | {fmt_ci(r['OR_ci95_lo'], r['OR_ci95_hi'])} |"
            )

    Path(args.out_md).write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[OK] wrote: {args.out_csv}")
    print(f"[OK] wrote: {args.out_md}")
    print("[INFO] rows:", len(out))


if __name__ == "__main__":
    main()
