import argparse, json, math
from pathlib import Path
import numpy as np


def read_final(jsonl_path: Path):
    last = None
    mode = pillar = None
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("record_type") == "meta":
                mode = obj.get("mode")
                pillar = obj.get("pillar")
            if obj.get("record_type") == "final":
                last = obj
    if last is None:
        raise RuntimeError(f"no final record in {jsonl_path}")
    last["_mode"] = mode
    last["_pillar"] = pillar
    last["_file"] = str(jsonl_path)
    return last


def wilson_ci(p, n, z=1.96):
    if n <= 0:
        return (0.0, 1.0)
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = (z * math.sqrt((p*(1-p)/n) + (z*z/(4*n*n)))) / denom
    return (max(0.0, center-half), min(1.0, center+half))


def rr_or(a, b, c, d):
    """
    2x2:
      controlled: collapse=a, ok=b
      baseline:   collapse=c, ok=d
    returns RR, OR with Haldane-Anscombe correction
    """
    # add 0.5 to each cell to avoid zero division
    aa, bb, cc, dd = a+0.5, b+0.5, c+0.5, d+0.5
    rr = (aa/(aa+bb)) / (cc/(cc+dd))
    orr = (aa*dd) / (bb*cc)
    return rr, orr


def bootstrap_rr_or(y_ctrl, y_base, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    n1 = len(y_ctrl)
    n0 = len(y_base)
    if n1 == 0 or n0 == 0:
        return (None, None, None, None)
    rrs = []
    ors = []
    for _ in range(B):
        s1 = rng.integers(0, n1, size=n1)
        s0 = rng.integers(0, n0, size=n0)
        a = int(np.sum(np.array(y_ctrl)[s1] == 1))
        b = int(np.sum(np.array(y_ctrl)[s1] == 0))
        c = int(np.sum(np.array(y_base)[s0] == 1))
        d = int(np.sum(np.array(y_base)[s0] == 0))
        rr, orr = rr_or(a, b, c, d)
        rrs.append(rr)
        ors.append(orr)
    rrs = np.array(rrs)
    ors = np.array(ors)
    return (
        float(np.quantile(rrs, 0.025)),
        float(np.quantile(rrs, 0.975)),
        float(np.quantile(ors, 0.025)),
        float(np.quantile(ors, 0.975)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="exp/runs")
    ap.add_argument("--outdir", type=str, default="exp/summary")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    finals = []
    for p in runs_root.rglob("*.jsonl"):
        try:
            finals.append(read_final(p))
        except Exception:
            continue

    # group by pillar/mode
    pillars = sorted({f["_pillar"] for f in finals if f["_pillar"] is not None})
    if not pillars:
        raise RuntimeError("no runs found (no meta/final parsed)")

    rows = []
    for pillar in pillars:
        base = [f for f in finals if f["_pillar"] == pillar and f["_mode"] == "baseline"]
        ctrl = [f for f in finals if f["_pillar"] == pillar and f["_mode"] == "controlled"]

        yb = [int(f.get("collapse_run", 0)) for f in base]
        yc = [int(f.get("collapse_run", 0)) for f in ctrl]

        nb = len(yb)
        nc = len(yc)
        cb = int(sum(yb))
        cc = int(sum(yc))

        pb = cb / nb if nb else 0.0
        pc = cc / nc if nc else 0.0

        # intervene_any primary (binary)
        interv_any = [1 if int(f.get("n_interventions", 0)) > 0 else 0 for f in ctrl]
        interv_rate = float(np.mean(interv_any)) if interv_any else 0.0

        # mean n_interventions / n_tokens_total (secondary)
        ratios = []
        for f in ctrl:
            nt = int(f.get("n_tokens_total", 0))
            ni = int(f.get("n_interventions", 0))
            if nt > 0:
                ratios.append(ni / nt)
        mean_int_per_tok = float(np.mean(ratios)) if ratios else 0.0

        # runtime cost
        rb = [float(f.get("runtime_sec", 0.0)) for f in base]
        rc = [float(f.get("runtime_sec", 0.0)) for f in ctrl]
        mean_rb = float(np.mean(rb)) if rb else 0.0
        mean_rc = float(np.mean(rc)) if rc else 0.0

        # RR / OR
        a = cc
        b = nc - cc
        c = cb
        d = nb - cb
        rr, orr = rr_or(a, b, c, d)

        rr_lo, rr_hi, or_lo, or_hi = bootstrap_rr_or(yc, yb, B=int(args.bootstrap), seed=int(args.seed))

        # simple CI for proportions
        pc_lo, pc_hi = wilson_ci(pc, nc) if nc else (0.0, 1.0)
        pb_lo, pb_hi = wilson_ci(pb, nb) if nb else (0.0, 1.0)

        rows.append({
            "pillar": pillar,

            "n_baseline": nb,
            "collapse_baseline": cb,
            "collapse_rate_baseline": pb,
            "collapse_rate_baseline_ci95_lo": pb_lo,
            "collapse_rate_baseline_ci95_hi": pb_hi,

            "n_controlled": nc,
            "collapse_controlled": cc,
            "collapse_rate_controlled": pc,
            "collapse_rate_controlled_ci95_lo": pc_lo,
            "collapse_rate_controlled_ci95_hi": pc_hi,

            "delta_collapse_rate": pc - pb,

            "RR_controlled_vs_baseline": rr,
            "RR_ci95_lo": rr_lo,
            "RR_ci95_hi": rr_hi,

            "OR_controlled_vs_baseline": orr,
            "OR_ci95_lo": or_lo,
            "OR_ci95_hi": or_hi,

            "intervene_any_rate_primary": interv_rate,
            "mean_int_per_token_secondary": mean_int_per_tok,

            "mean_runtime_sec_baseline": mean_rb,
            "mean_runtime_sec_controlled": mean_rc,
            "delta_runtime_sec": mean_rc - mean_rb,
        })

    # write CSV
    import csv
    out_csv = outdir / "table1_vsave.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[WROTE] {out_csv}")
    print("pillars:", pillars)
    print("bootstrap:", args.bootstrap)


if __name__ == "__main__":
    main()
