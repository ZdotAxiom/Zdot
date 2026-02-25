import argparse, json
from pathlib import Path

def read_final(p: Path):
    meta = {}
    final = None
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            if o.get("record_type") == "meta":
                meta = o
            if o.get("record_type") == "final":
                final = o
    if not final:
        return None
    final["_mode"] = meta.get("mode")
    final["_pillar"] = meta.get("pillar")
    final["_seed"] = meta.get("seed")
    return final

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", type=str, default="exp/runs")
    ap.add_argument("--pillar", type=str, required=True)
    args = ap.parse_args()

    root = Path(args.runs_root)
    finals = []
    for p in root.rglob("seed_*.jsonl"):
        f = read_final(p)
        if f and f.get("_pillar") == args.pillar:
            finals.append(f)

    for mode in ["baseline", "controlled"]:
        xs = [f for f in finals if f.get("_mode") == mode]
        n = len(xs)
        c = sum(int(f.get("collapse_run", 0)) for f in xs)
        any_int = sum(1 for f in xs if int(f.get("n_interventions", 0)) > 0)
        print(f"{args.pillar} {mode}: n={n} collapse={c} ({(c/n*100 if n else 0):.2f}%) intervene_any={any_int}/{n}")

    # show first few collapse reasons
    col = [f for f in finals if int(f.get("collapse_run", 0)) == 1]
    col = sorted(col, key=lambda x: int(x.get("_seed", 0)))[:10]
    if col:
        print("\nexamples (first 10 collapses):")
        for f in col:
            print(" seed", f.get("_seed"), "reason", f.get("collapse_reason", "N/A"), "tokens", f.get("n_tokens_total"))
    else:
        print("\nno collapses found yet (this may be OK depending on stress settings).")

if __name__ == "__main__":
    main()
