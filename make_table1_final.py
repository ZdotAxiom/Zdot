#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd

def pct_ci(x, alpha=0.05):
    lo = np.quantile(x, alpha/2)
    hi = np.quantile(x, 1 - alpha/2)
    return lo, hi

def safe_or_rr(a, b, c, d, correction=0.5):
    """
    2x2 table:
      baseline: collapse=a, non=b
      controlled: collapse=c, non=d
    returns (RR, OR) with Haldane-Anscombe correction when needed.
    """
    # RR = (c/(c+d)) / (a/(a+b))
    # OR = (c/d) / (a/b) = c*b/(d*a)
    need_corr = (a==0) or (b==0) or (c==0) or (d==0)
    if need_corr:
        a2=a+correction; b2=b+correction; c2=c+correction; d2=d+correction
    else:
        a2=a; b2=b; c2=c; d2=d

    pb = a2/(a2+b2)
    pc = c2/(c2+d2)
    rr = pc/pb if pb>0 else np.inf
    orv = (c2*b2)/(d2*a2) if (d2*a2)>0 else np.inf
    return rr, orv

def wilson_ci(k, n, z=1.96):
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
    p = k/n if n else np.nan
    lo, hi = wilson_ci(k, n)
    return f"{p:.3f} ({k}/{n}) [{lo:.3f},{hi:.3f}]"

def paired_bootstrap(d, B, rng):
    """
    Paired resampling by seed if available.
    Returns arrays of (pb, pc, delta, rr, orv).
    """
    has_seed = "seed" in d.columns
    out = {"pb":[], "pc":[], "delta":[], "rr":[], "orv":[]}

    if has_seed:
        # ensure one row per (seed, mode)
        seeds = np.array(sorted(d["seed"].unique()))
        # build lookup dict seed->rows
        # we assume each seed has exactly one baseline and one controlled row
        # if not, we fall back to independent
        ok = True
        for s in seeds[:min(len(seeds),10)]:
            sub = d[d["seed"]==s]
            if set(sub["mode"]) != {"baseline","controlled"}:
                ok = False
                break
        if not ok:
            has_seed = False

    if has_seed:
        seeds = np.array(sorted(d["seed"].unique()))
        n = len(seeds)
        # pre-split for speed
        base_map = d[d["mode"]=="baseline"].set_index("seed")
        ctrl_map = d[d["mode"]=="controlled"].set_index("seed")

        base_y = base_map["collapse_run"].astype(int)
        ctrl_y = ctrl_map["collapse_run"].astype(int)

        for _ in range(B):
            draw = rng.integers(0, n, size=n)
            sd = seeds[draw]

            kb = int(base_y.loc[sd].sum()); nb = n
            kc = int(ctrl_y.loc[sd].sum()); nc = n

            pb = kb/nb
            pc = kc/nc
            delta = pc - pb

            a=kb; b=nb-kb; c=kc; d2=nc-kc
            rr, orv = safe_or_rr(a,b,c,d2)

            out["pb"].append(pb); out["pc"].append(pc)
            out["delta"].append(delta); out["rr"].append(rr); out["orv"].append(orv)

        for k in out:
            out[k] = np.array(out[k], dtype=float)
        return out, True

    # fallback: independent resampling within each mode
    b = d[d["mode"]=="baseline"]["collapse_run"].astype(int).to_numpy()
    c = d[d["mode"]=="controlled"]["collapse_run"].astype(int).to_numpy()
    nb = len(b); nc = len(c)
    for _ in range(B):
        bb = b[rng.integers(0, nb, size=nb)]
        cc = c[rng.integers(0, nc, size=nc)]
        kb = int(bb.sum()); kc = int(cc.sum())
        pb = kb/nb; pc = kc/nc
        delta = pc - pb
        a=kb; b0=nb-kb; c0=kc; d0=nc-kc
        rr, orv = safe_or_rr(a,b0,c0,d0)
        out["pb"].append(pb); out["pc"].append(pc)
        out["delta"].append(delta); out["rr"].append(rr); out["orv"].append(orv)
    for k in out:
        out[k] = np.array(out[k], dtype=float)
    return out, False

def col_or_na(df, col):
    return df[col] if col in df.columns else pd.Series([np.nan]*len(df))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", required=True)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pillars", default="HUM,STEM,ETH,TEMP,META")
    args = ap.parse_args()

    df = pd.read_csv(args.summary_csv)
    pillars = [x.strip() for x in args.pillars.split(",") if x.strip()]
    df = df[df["pillar"].isin(pillars)].copy()

    # required cols
    for c in ["pillar","mode","collapse_run","n_interventions"]:
        if c not in df.columns:
            raise ValueError(f"missing column: {c}")

    rng = np.random.default_rng(args.seed)
    B = int(args.bootstrap)

    rows = []
    paired_used = {}

    for pillar in pillars:
        d = df[df["pillar"]==pillar].copy()
        if set(d["mode"]) != {"baseline","controlled"}:
            raise ValueError(f"{pillar}: need both baseline and controlled in summary_csv")

        b = d[d["mode"]=="baseline"]
        c = d[d["mode"]=="controlled"]
        nb = len(b); nc = len(c)
        kb = int(b["collapse_run"].sum())
        kc = int(c["collapse_run"].sum())

        pb = kb/nb
        pc = kc/nc
        delta = pc - pb

        # point estimates RR/OR with correction
        rr, orv = safe_or_rr(kb, nb-kb, kc, nc-kc)

        # bootstrap CI
        boot, used_paired = paired_bootstrap(d, B=B, rng=rng)
        paired_used[pillar] = used_paired

        d_lo, d_hi = pct_ci(boot["delta"])
        rr_lo, rr_hi = pct_ci(boot["rr"])
        or_lo, or_hi = pct_ci(boot["orv"])

        # intervention metrics (primary: mean 1[n_interventions>0])
        int_rate_primary = float((c["n_interventions"]>0).mean())
        # density secondary if token total exists
        if "n_tokens_total" in c.columns:
            int_density = float((c["n_interventions"] / c["n_tokens_total"].replace(0,np.nan)).mean())
        else:
            int_density = np.nan

        # runtime if available
        rt_b = float(b["runtime_sec"].mean()) if "runtime_sec" in b.columns else np.nan
        rt_c = float(c["runtime_sec"].mean()) if "runtime_sec" in c.columns else np.nan
        rt_delta = rt_c - rt_b if (np.isfinite(rt_b) and np.isfinite(rt_c)) else np.nan

        rows.append({
            "pillar": pillar,
            "n_per_mode": nb,
            "collapse_rate_baseline": pb,
            "collapse_rate_controlled": pc,
            "delta_collapse_ctrl_minus_base": delta,
            "delta_collapse_ci95": f"[{d_lo:.4f},{d_hi:.4f}]",
            "RR_collapse_ctrl_over_base": rr,
            "RR_collapse_ci95": f"[{rr_lo:.4f},{rr_hi:.4f}]",
            "OR_collapse_ctrl_over_base": orv,
            "OR_collapse_ci95": f"[{or_lo:.4f},{or_hi:.4f}]",
            "counts_baseline": f"{kb}/{nb}",
            "counts_controlled": f"{kc}/{nc}",
            "intervene_rate_primary": int_rate_primary,
            "intervene_density_secondary": int_density,
            "runtime_sec_baseline_mean": rt_b,
            "runtime_sec_controlled_mean": rt_c,
            "delta_runtime_sec_mean": rt_delta,
            "bootstrap_paired_by_seed": bool(used_paired),
        })

    t = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_md) or ".", exist_ok=True)
    t.to_csv(args.out_csv, index=False)

    # markdown
    md = []
    md.append("| Pillar | n | collapse_base | collapse_ctrl | Δ (ctrl-base) | Δ 95%CI | RR | RR 95%CI | OR | OR 95%CI | intervene_rate | runtime_base | runtime_ctrl | Δruntime | paired_boot |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in t.iterrows():
        pillar = r["pillar"]
        n = int(r["n_per_mode"])
        kb, nb = r["counts_baseline"].split("/")
        kc, nc = r["counts_controlled"].split("/")
        base_str = fmt_rate(int(kb), int(nb))
        ctrl_str = fmt_rate(int(kc), int(nc))
        md.append(
            f"| {pillar} | {n} | {base_str} | {ctrl_str} | {r['delta_collapse_ctrl_minus_base']:+.3f} | {r['delta_collapse_ci95']} | "
            f"{r['RR_collapse_ctrl_over_base']:.3f} | {r['RR_collapse_ci95']} | "
            f"{r['OR_collapse_ctrl_over_base']:.3f} | {r['OR_collapse_ci95']} | "
            f"{r['intervene_rate_primary']:.3f} | "
            f"{(r['runtime_sec_baseline_mean'] if np.isfinite(r['runtime_sec_baseline_mean']) else np.nan):.1f} | "
            f"{(r['runtime_sec_controlled_mean'] if np.isfinite(r['runtime_sec_controlled_mean']) else np.nan):.1f} | "
            f"{(r['delta_runtime_sec_mean'] if np.isfinite(r['delta_runtime_sec_mean']) else np.nan):.1f} | "
            f"{'Y' if r['bootstrap_paired_by_seed'] else 'N'} |"
        )

    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("Wrote:")
    print(" -", args.out_csv)
    print(" -", args.out_md)
    print("\nPaired bootstrap by seed used:")
    for p in pillars:
        print(f" - {p}: {paired_used.get(p, False)}")

if __name__ == "__main__":
    main()
