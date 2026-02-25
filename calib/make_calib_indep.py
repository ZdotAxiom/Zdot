#!/usr/bin/env python3
import argparse, json, os, statistics, sys
from pathlib import Path

def percentile(xs, q):
    # q in [0,100]
    if not xs:
        raise ValueError("empty list")
    xs = sorted(xs)
    if q <= 0:  return xs[0]
    if q >= 100:return xs[-1]
    k = (len(xs) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return d0 + d1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="independent calibration corpus (one text per line)")
    ap.add_argument("--out", required=True, help="output calib json")
    ap.add_argument("--q50", type=float, default=50.0)
    ap.add_argument("--q95", type=float, default=95.0)
    ap.add_argument("--note", type=str, default="")
    args = ap.parse_args()

    # repo root を sys.path に入れて import を安定させる
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    import run_vnext_with_seed_v5 as rv5  # same pi_raw function set

    # SESConfig は rv5.ses にいる（run_vnext_with_seed_v5.py 内でもそう呼んでる）
    cfg = rv5.ses.SESConfig()

    pi_raws = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            r = rv5.ses.hybrid_complexity(t, cfg)  # rv5 と同一の pi_raw
            pi_raws.append(float(r))

    q50 = percentile(pi_raws, args.q50)
    q95 = percentile(pi_raws, args.q95)

    out = {
        "complexity": {"q50": q50, "q95": q95},
        "calibration_corpus": os.path.basename(args.infile),
        "n_samples": len(pi_raws),
        "note": args.note
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as g:
        json.dump(out, g, ensure_ascii=False, indent=2)

    print("wrote:", args.out)
    print("n_samples:", len(pi_raws))
    print("q50:", q50, "q95:", q95)
    print("pi_raw min/median/max:",
          min(pi_raws),
          statistics.median(pi_raws),
          max(pi_raws))

if __name__ == "__main__":
    main()
