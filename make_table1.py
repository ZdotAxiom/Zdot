#!/usr/bin/env python3
import argparse, math, random
from pathlib import Path
import pandas as pd

def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z * math.sqrt((p*(1-p) + z*z/(4*n))/n)) / denom
    return (max(0.0, center-half), min(1.0, center+half))

def rr_or_ci(a,b,c,d, B=20000, seed=0):
    # a=ctrl collapse, b=ctrl non, c=base collapse, d=base non
    # Haldane–Anscombe (+0.5) for RR/OR stability
    def rr_or(a,b,c,d):
        aa=a+0.5; bb=b+0.5; cc=c+0.5; dd=d+0.5
        rr = (aa/(aa+bb)) / (cc/(cc+dd))
        orr = (aa*dd) / (bb*cc)
        return rr, orr

    rr0, or0 = rr_or(a,b,c,d)
    rs = random.Random(seed)
    rrs, ors = [], []
    # bootstrap over paired seeds is handled outside; here just placeholder
    # we will compute paired bootstrap on per-seed collapse indicators.
    return rr0, or0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    # required columns
    for col in ["pillar","mode","seed","collapse_run","n_interventions","runtime_sec"]:
        if col not in df.columns:
            raise SystemExit(f"missing required column: {col}")

    # force types
    df["seed"] = df["seed"].astype(int)
    df["collapse_run"] = df["collapse_run"].astype(int)
    df["n_interventions"] = df["n_interventions"].fillna(0).astype(int)
    df["runtime_sec"] = pd.to_numeric(df["runtime_sec"], errors="coerce")

    pillars = sorted(df["pillar"].dropna().unique().tolist())

    rows = []
    rs = random.Random(args.seed)

    for pillar in pillars:
        sub = df[df["pillar"]==pillar].copy()
        base = sub[sub["mode"]=="baseline"][["seed","collapse_run","runtime_sec"]].rename(columns={"collapse_run":"y_base","runtime_sec":"rt_base"})
        ctrl = sub[sub["mode"]=="controlled"][["seed","collapse_run","runtime_sec","n_interventions"]].rename(columns={"collapse_run":"y_ctrl","runtime_sec":"rt_ctrl"})
        m = pd.merge(base, ctrl, on="seed", how="inner")

        n = len(m)
        if n == 0:
            continue

        # rates
        c_base = int(m["y_base"].sum())
        c_ctrl = int(m["y_ctrl"].sum())
        p_base = c_base / n
        p_ctrl = c_ctrl / n
        delta = p_ctrl - p_base

        # Wilson CI for rates
        base_lo, base_hi = wilson_ci(c_base, n)
        ctrl_lo, ctrl_hi = wilson_ci(c_ctrl, n)

        # Paired bootstrap over seeds for delta/RR/OR
        deltas, rrs, ors = [], [], []
        for _ in range(args.bootstrap):
            idx = [rs.randrange(n) for _ in range(n)]
            mb = m.iloc[idx]
            cb = int(mb["y_base"].sum()); cc = int(mb["y_ctrl"].sum())
            pb = cb/n; pc = cc/n
            deltas.append(pc - pb)

            # RR/OR with Haldane correction
            # ctrl: a=cc, b=n-cc ; base: c=cb, d=n-cb
            a=cc; b=n-cc; c=cb; d=n-cb
            aa=a+0.5; bb=b+0.5; cc_=c+0.5; dd=d+0.5
            rr = (aa/(aa+bb)) / (cc_/(cc_+dd))
            orr = (aa*dd) / (bb*cc_)
            rrs.append(rr); ors.append(orr)

        def pct_ci(arr):
            arr = sorted(arr)
            lo = arr[int(0.025*len(arr))]
            hi = arr[int(0.975*len(arr))-1]
            return lo, hi

        d_lo, d_hi = pct_ci(deltas)
        rr = sorted(rrs)[len(rrs)//2]
        rr_lo, rr_hi = pct_ci(rrs)
        orr = sorted(ors)[len(ors)//2]
        or_lo, or_hi = pct_ci(ors)

        intv_rate = float((m["n_interventions"]>0).mean())
        rt_base = float(m["rt_base"].mean())
        rt_ctrl = float(m["rt_ctrl"].mean())
        d_rt = rt_ctrl - rt_base

        rows.append({
            "pillar": pillar,
            "n": n,
            "collapse_base": p_base,
            "collapse_base_ci95": f"[{base_lo:.3f},{base_hi:.3f}]",
            "collapse_ctrl": p_ctrl,
            "collapse_ctrl_ci95": f"[{ctrl_lo:.3f},{ctrl_hi:.3f}]",
            "delta": delta,
            "delta_ci95": f"[{d_lo:.3f},{d_hi:.3f}]",
            "RR": rr,
            "RR_ci95": f"[{rr_lo:.3f},{rr_hi:.3f}]",
            "OR": orr,
            "OR_ci95": f"[{or_lo:.3f},{or_hi:.3f}]",
            "intervene_rate_primary": intv_rate,
            "runtime_sec_base": rt_base,
            "runtime_sec_ctrl": rt_ctrl,
            "delta_runtime_sec": d_rt,
        })

    out_df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # Markdown table for paper/appendix
    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("| Pillar | n | base | base 95%CI | ctrl | ctrl 95%CI | Δ (ctrl-base) | Δ 95%CI | RR | RR 95%CI | OR | OR 95%CI | intervene_rate | rt_base | rt_ctrl | Δrt |\n")
        f.write("|---|---:|---:|---|---:|---|---:|---|---:|---|---:|---|---:|---:|---:|---:|\n")
        for _, r in out_df.iterrows():
            f.write(
                f"| {r['pillar']} | {int(r['n'])} | {r['collapse_base']:.3f} | {r['collapse_base_ci95']} | "
                f"{r['collapse_ctrl']:.3f} | {r['collapse_ctrl_ci95']} | {r['delta']:.3f} | {r['delta_ci95']} | "
                f"{r['RR']:.3f} | {r['RR_ci95']} | {r['OR']:.3f} | {r['OR_ci95']} | "
                f"{r['intervene_rate_primary']:.3f} | {r['runtime_sec_base']:.2f} | {r['runtime_sec_ctrl']:.2f} | {r['delta_runtime_sec']:.2f} |\n"
            )

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {md_path}")

if __name__ == "__main__":
    main()
