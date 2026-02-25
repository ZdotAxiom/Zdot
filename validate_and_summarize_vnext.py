#!/usr/bin/env python3
import argparse, csv, glob, json, os, re, sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

def _expand_roots(patterns: List[str]) -> List[Path]:
    roots: List[Path] = []
    for p in patterns:
        if any(ch in p for ch in ["*", "?", "["]):
            for m in glob.glob(p):
                roots.append(Path(m))
        else:
            roots.append(Path(p))
    # de-dup + keep order
    out = []
    seen = set()
    for r in roots:
        rp = str(r.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(r)
    return out

def _iter_jsonl_files(roots: List[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file() and root.suffix == ".jsonl":
            yield root
            continue
        if root.is_dir():
            for p in root.rglob("seed_*.jsonl"):
                yield p

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None: return None
        return float(x)
    except Exception:
        return None

def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None: return None
        return int(x)
    except Exception:
        return None

def _infer_mode_pillar_seed(path: Path) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    # expected: .../<mode>/<PILLAR>/seed_000.jsonl
    parts = path.parts
    mode = None
    pillar = None
    seed = None
    m = re.search(r"seed_(\d+)\.jsonl$", path.name)
    if m:
        seed = int(m.group(1))
    if len(parts) >= 3:
        pillar = parts[-2]
        mode = parts[-3]
    return mode, pillar, seed

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: 
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                # tolerate occasional broken line
                continue
    return rows

def _extract_summary(objs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # robust against schema drift:
    # - look for meta/final entries, else infer from last object
    meta = None
    final = None
    steps = []
    for o in objs:
        if "meta" in o and isinstance(o["meta"], dict):
            meta = o["meta"]
        if "final" in o and isinstance(o["final"], dict):
            final = o["final"]
        if "step" in o and isinstance(o["step"], dict):
            steps.append(o["step"])
    if final is None:
        # sometimes final is at top-level keys
        cand = objs[-1] if objs else {}
        final = cand.get("final") if isinstance(cand.get("final"), dict) else cand

    if meta is None:
        cand0 = objs[0] if objs else {}
        meta = cand0.get("meta") if isinstance(cand0.get("meta"), dict) else {}

    # Common fields (best-effort)
    out: Dict[str, Any] = {}
    out["exp_id"] = meta.get("exp_id") or final.get("exp_id")
    out["mode"] = meta.get("mode") or final.get("mode")
    out["pillar"] = meta.get("pillar") or final.get("pillar")
    out["seed"] = meta.get("seed") if meta.get("seed") is not None else final.get("seed")

    out["prompt_id"] = meta.get("prompt_id") if meta.get("prompt_id") is not None else final.get("prompt_id")
    out["prompt_hash"] = meta.get("prompt_hash") or final.get("prompt_hash")

    out["n_tokens_total"] = final.get("n_tokens_total") or final.get("tokens_total") or meta.get("n_tokens_total")
    out["n_interventions"] = final.get("n_interventions") or meta.get("n_interventions") or 0
    out["runtime_sec"] = final.get("runtime_sec") or meta.get("runtime_sec")

    # Collapse flags / reasons (best-effort)
    out["collapse_run"] = final.get("collapse_run")
    out["collapse_flag"] = final.get("collapse_flag")
    out["too_short_flag"] = final.get("too_short_flag")
    out["pi_flag_any"] = final.get("pi_flag_any") or final.get("pi_flag")
    out["rep_flag_any"] = final.get("rep_flag_any") or final.get("rep_flag")

    out["collapse_reason"] = final.get("collapse_reason") or final.get("reason") or ""

    # If collapse not explicitly provided, infer from any flags
    def truthy(v: Any) -> int:
        return 1 if v in (1, True, "1", "true", "True") else 0
    if out["collapse_run"] is None:
        # treat too_short OR pi OR rep as collapse
        out["collapse_run"] = int(
            truthy(out.get("too_short_flag")) or truthy(out.get("pi_flag_any")) or truthy(out.get("rep_flag_any")) or truthy(out.get("collapse_flag"))
        )

    # add last observed pi/rep values if present in final
    out["pi_raw_last"] = _safe_float(final.get("pi_raw_last") or final.get("pi_raw"))
    out["pi_norm_last"] = _safe_float(final.get("pi_norm_last") or final.get("pi_norm"))
    out["rep_score_last"] = _safe_float(final.get("rep_score_last") or final.get("rep_score"))

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", action="append", required=True,
                    help='root dir or glob, e.g. exp_vnext_ollama_512/runs_500_*')
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--expect", type=int, default=500,
                    help="expected runs per (pillar,mode) condition (default 500)")
    args = ap.parse_args()

    roots = _expand_roots(args.runs_root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd().resolve()

    rows: List[Dict[str, Any]] = []
    files = sorted(set(_iter_jsonl_files(roots)), key=lambda p: str(p))
    if not files:
        print("ERROR: no jsonl files found under runs_root patterns.", file=sys.stderr)
        sys.exit(2)

    for fp in files:
        objs = _read_jsonl(fp)
        s = _extract_summary(objs)
        mode, pillar, seed = _infer_mode_pillar_seed(fp)
        # fill missing from path if absent
        if not s.get("mode") and mode: s["mode"] = mode
        if not s.get("pillar") and pillar: s["pillar"] = pillar
        if s.get("seed") is None and seed is not None: s["seed"] = seed
        try:
            s["path"] = str(fp.resolve().relative_to(cwd))
        except Exception:
            s["path"] = os.path.relpath(str(fp), str(cwd))
        rows.append(s)

    # write runs_summary_all.csv
    csv_path = outdir / "runs_summary_all.csv"
    fieldnames = [
        "exp_id","mode","pillar","seed","prompt_id","prompt_hash",
        "collapse_run","collapse_flag","too_short_flag","pi_flag_any","rep_flag_any","collapse_reason",
        "n_tokens_total","n_interventions","runtime_sec",
        "pi_raw_last","pi_norm_last","rep_score_last",
        "path"
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # normalize ints/floats
            r2 = dict(r)
            for k in ["seed","prompt_id","n_tokens_total","n_interventions","collapse_run"]:
                if r2.get(k) is not None:
                    r2[k] = int(r2[k])
            for k in ["runtime_sec","pi_raw_last","pi_norm_last","rep_score_last"]:
                if r2.get(k) is not None:
                    r2[k] = float(r2[k])
            w.writerow({k: r2.get(k) for k in fieldnames})

    # quick check counts per condition
    from collections import Counter
    c = Counter((r.get("pillar"), r.get("mode")) for r in rows)
    check_path = outdir / "counts_by_condition.csv"
    with check_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pillar","mode","n_runs","ok_expect"])
        for (pillar, mode), n in sorted(c.items()):
            w.writerow([pillar, mode, n, int(n == args.expect)])

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {check_path}")
    # show any missing conditions
    bad = [(p,m,n) for (p,m),n in c.items() if n != args.expect]
    if bad:
        print("WARNING: some (pillar,mode) do not match --expect:", bad, file=sys.stderr)

if __name__ == "__main__":
    main()
