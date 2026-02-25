#!/usr/bin/env python3
import argparse, os
import pandas as pd
import numpy as np

def wilson_ci(k, n, z=1.96):
    # Wilson score interval for binomial proportion
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi

def fmt_rate(k, n):
    return f"{k/n:.3f} ({k}/{n})" if n else "NA"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csvs", nargs="+", required=True, help="runs_summary_all.csv paths")
    ap.add_argument("--pillars", default="HUM,STEM,ETH", help="comma-separated pillars")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--prefix", default="table_hse")
    args = ap.parse_args()

    pillars = [x.strip() for x in args.pillars.split(",") if x.strip()]
    os.makedirs(args.outdir, exist_ok=True)

    # load + concat
    dfs = []
    for p in args.csvs:
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        df = pd.read_csv(p)
        df["__source_csv"] = os.path.basename(p)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # sanity: require fields
    need_cols = ["pillar","mode","collapse_run","collapse_reason","n_interventions"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    df = df[df["pillar"].isin(pillars)].copy()

    # ---------- Table 1: collapse rate + delta + RR + Wilson CI ----------
    rows = []
    for pillar in pillars:
        d = df[df["pillar"]==pillar]
        for mode in ["baseline","controlled"]:
            if mode not in set(d["mode"]):
                raise ValueError(f"{pillar}: missing mode={mode}")
        b = d[d["mode"]=="baseline"]
        c = d[d["mode"]=="controlled"]
        nb = len(b); nc = len(c)
        kb = int(b["collapse_run"].sum())
        kc = int(c["collapse_run"].sum())

        pb = kb/nb if nb else np.nan
        pc = kc/nc if nc else np.nan

        # delta (ctrl - base)
        delta = pc - pb

        # RR (ctrl/base) with safe handling
        rr = (pc/pb) if (pb and pb>0) else (0.0 if pc==0 else np.inf)

        # CIs (Wilson) for each rate
        b_lo, b_hi = wilson_ci(kb, nb)
        c_lo, c_hi = wilson_ci(kc, nc)

        rows.append({
            "pillar": pillar,
            "n_per_mode": nb,  # assume equal
            "collapse_baseline": pb,
            "collapse_baseline_ci95": f"[{b_lo:.3f},{b_hi:.3f}]",
            "collapse_controlled": pc,
            "collapse_controlled_ci95": f"[{c_lo:.3f},{c_hi:.3f}]",
            "delta_ctrl_minus_base": delta,
            "RR_ctrl_over_base": rr,
            "counts_baseline": f"{kb}/{nb}",
            "counts_controlled": f"{kc}/{nc}",
        })

    t1 = pd.DataFrame(rows)

    # pretty md for Table 1
    t1_md = []
    t1_md.append("| Pillar | n (per mode) | Collapse rate (baseline) | Collapse rate (controlled) | Δ (ctrl−base) | RR (ctrl/base) |")
    t1_md.append("|---|---:|---:|---:|---:|---:|")
    for _, r in t1.iterrows():
        pillar = r["pillar"]
        n = int(r["n_per_mode"])
        kb, nb = r["counts_baseline"].split("/")
        kc, nc = r["counts_controlled"].split("/")
        base_str = f'{float(r["collapse_baseline"]):.3f} ({kb}/{nb}) {r["collapse_baseline_ci95"]}'
        ctrl_str = f'{float(r["collapse_controlled"]):.3f} ({kc}/{nc}) {r["collapse_controlled_ci95"]}'
        delta_str = f'{float(r["delta_ctrl_minus_base"]):+.3f}'
        rr = r["RR_ctrl_over_base"]
        if np.isinf(rr):
            rr_str = "inf"
        else:
            rr_str = f"{float(rr):.3f}"
        t1_md.append(f"| {pillar} | {n} | {base_str} | {ctrl_str} | {delta_str} | {rr_str} |")

    # ---------- Table 2: reason counts + intervention stats ----------
    # reason crosstab per (pillar, mode)
    ct = pd.crosstab([df["pillar"], df["mode"]], df["collapse_reason"]).reset_index()
    # interventions stats for controlled
    dc = df[df["mode"]=="controlled"].copy()
    stats = (dc.groupby("pillar")["n_interventions"]
               .agg(mean="mean", median="median", p95=lambda x: np.quantile(x,0.95), max="max")
               .reset_index())
    # merge into one wide-ish table for md
    # Ensure columns exist
    reason_cols = [c for c in ct.columns if c not in ["pillar","mode"]]
    # Build md
    t2_md = []
    hdr = ["Pillar",
           "baseline NONE","baseline REP","baseline TOO_SHORT",
           "controlled NONE","controlled REP","controlled TOO_SHORT",
           "interv mean","median","p95","max"]
    t2_md.append("| " + " | ".join(hdr) + " |")
    t2_md.append("|" + "|".join(["---:"]*len(hdr)) + "|")
    for pillar in pillars:
        def get_reason(mode, reason):
            sub = ct[(ct["pillar"]==pillar)&(ct["mode"]==mode)]
            if sub.empty or reason not in sub.columns:
                return 0
            return int(sub.iloc[0][reason])

        b_none = get_reason("baseline","NONE")
        b_rep  = get_reason("baseline","REP")
        b_ts   = get_reason("baseline","TOO_SHORT")
        c_none = get_reason("controlled","NONE")
        c_rep  = get_reason("controlled","REP")
        c_ts   = get_reason("controlled","TOO_SHORT")

        st = stats[stats["pillar"]==pillar]
        if st.empty:
            mean=median=p95=mx=np.nan
        else:
            mean=float(st["mean"].iloc[0]); median=float(st["median"].iloc[0]); p95=float(st["p95"].iloc[0]); mx=float(st["max"].iloc[0])

        t2_md.append(
            f"| {pillar} | {b_none} | {b_rep} | {b_ts} | {c_none} | {c_rep} | {c_ts} | "
            f"{mean:.3f} | {median:.0f} | {p95:.0f} | {mx:.0f} |"
        )

    # write outputs
    out_csv1 = os.path.join(args.outdir, f"{args.prefix}_table1.csv")
    out_md1  = os.path.join(args.outdir, f"{args.prefix}_table1.md")
    out_csv2 = os.path.join(args.outdir, f"{args.prefix}_table2_reason_counts.csv")
    out_csv3 = os.path.join(args.outdir, f"{args.prefix}_table2_interventions.csv")
    out_md2  = os.path.join(args.outdir, f"{args.prefix}_table2.md")

    t1.to_csv(out_csv1, index=False)
    with open(out_md1, "w", encoding="utf-8") as f:
        f.write("\n".join(t1_md) + "\n")

    ct.to_csv(out_csv2, index=False)
    stats.to_csv(out_csv3, index=False)
    with open(out_md2, "w", encoding="utf-8") as f:
        f.write("\n".join(t2_md) + "\n")

    print("Wrote:")
    print(" -", out_csv1)
    print(" -", out_md1)
    print(" -", out_csv2)
    print(" -", out_csv3)
    print(" -", out_md2)

if __name__ == "__main__":
    main()
