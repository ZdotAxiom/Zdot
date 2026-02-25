#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd

REASON_ORDER = ["PI", "REP", "TOO_SHORT", "BUDGET_EXCEEDED", "BUDGET_EXHAUSTED", "BUDGET", "NONE", "OTHER"]

def _to_md_table(df: pd.DataFrame) -> str:
    # Minimal MD table renderer (no tabulate dependency).
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join([header, sep] + rows)

def _canon_reason(x: str) -> str:
    if x is None:
        return "NONE"
    s = str(x).strip().upper()
    if s in {"", "NAN", "NONE"}:
        return "NONE"
    if s in {"BUDGET_EXHAUSTED"}:
        return "BUDGET_EXHAUSTED"
    if s in {"BUDGET_EXCEEDED"}:
        return "BUDGET_EXCEEDED"
    if s in {"BUDGET"}:
        return "BUDGET"
    if s in {"PI"}:
        return "PI"
    if s in {"REP"}:
        return "REP"
    if s in {"TOO_SHORT"}:
        return "TOO_SHORT"
    return "OTHER"

def _pick_reason_columns(df: pd.DataFrame):
    # New taxonomy preferred; fallback to legacy.
    cols = []
    if "collapse_core_reason" in df.columns:
        cols.append("collapse_core_reason")
    elif "core_collapse_reason" in df.columns:
        cols.append("core_collapse_reason")
    if "stop_aux_reason" in df.columns:
        cols.append("stop_aux_reason")
    if "collapse_reason" in df.columns:
        cols.append("collapse_reason")
    return cols

def _make_reason_table(df: pd.DataFrame, group_cols):
    # Build "primary_reason": prefer core if exists else legacy collapse_reason
    if "collapse_core_reason" in df.columns:
        core = df["collapse_core_reason"].map(_canon_reason)
    elif "core_collapse_reason" in df.columns:
        core = df["core_collapse_reason"].map(_canon_reason)
    else:
        core = pd.Series(["NONE"] * len(df))

    aux = df["stop_aux_reason"].map(_canon_reason) if "stop_aux_reason" in df.columns else pd.Series(["NONE"] * len(df))
    legacy = df["collapse_reason"].map(_canon_reason) if "collapse_reason" in df.columns else pd.Series(["NONE"] * len(df))

    # Primary decision:
    # - if core is PI/REP -> use it
    # - else if aux is TOO_SHORT/BUDGET_* -> use aux
    # - else use legacy
    primary = []
    for c,a,l in zip(core.tolist(), aux.tolist(), legacy.tolist()):
        if c in {"PI","REP"}:
            primary.append(c)
        elif a in {"TOO_SHORT","BUDGET_EXHAUSTED","BUDGET_EXCEEDED","BUDGET"}:
            primary.append(a)
        else:
            primary.append(l)
    df = df.copy()
    df["primary_reason"] = primary

    # Determine "collapsed" flag (if present), else infer from reason != NONE
    if "collapse_run" in df.columns:
        collapsed = df["collapse_run"].astype(int)
    else:
        collapsed = (df["primary_reason"] != "NONE").astype(int)
    df["collapsed"] = collapsed

    # If the run did not collapse, it must not contribute to reason counts.
    if "collapse_run" in df.columns:
        df.loc[df["collapse_run"].astype(int) == 0, "primary_reason"] = "NONE"

    # Aggregate: counts and rates among ALL runs + among collapsed runs
    g = df.groupby(group_cols, dropna=False)
    rows = []
    for keys, sub in g:
        if not isinstance(keys, tuple):
            keys = (keys,)
        n_all = len(sub)
        n_col = int(sub["collapsed"].sum())
        # distribution among ALL
        dist_all = sub["primary_reason"].value_counts(dropna=False).to_dict()
        # distribution among collapsed
        dist_col = sub[sub["collapsed"]==1]["primary_reason"].value_counts(dropna=False).to_dict()

        def _get(d, k): return int(d.get(k, 0))
        r = {c:v for c,v in zip(group_cols, keys)}
        r.update({
            "n_all": n_all,
            "n_collapsed": n_col,
            "collapse_rate": (n_col / n_all) if n_all>0 else 0.0,
        })
        for reason in REASON_ORDER:
            r[f"all_{reason}"] = _get(dist_all, reason)
            r[f"all_{reason}_rate"] = (_get(dist_all, reason) / n_all) if n_all>0 else 0.0
            denom = n_col if n_col>0 else 1
            r[f"col_{reason}"] = _get(dist_col, reason)
            r[f"col_{reason}_rate"] = (_get(dist_col, reason) / denom) if n_col>0 else 0.0
        rows.append(r)
    out = pd.DataFrame(rows)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--basename", default="collapse_breakdown")
    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not summary_path.exists():
        raise FileNotFoundError(f"not found: {summary_path}")

    df = pd.read_csv(summary_path)

    # Sanity prints
    print("[breakdown] rows =", len(df))
    print("[breakdown] columns found for reasons:", _pick_reason_columns(df))
    for c in ["mode","pillar","collapse_run","collapse_reason","collapse_core_reason","stop_aux_reason"]:
        if c in df.columns:
            print("[breakdown] has:", c)

    # Table 1: overall by mode
    t_mode = _make_reason_table(df, ["mode"])
    # Table 2: by pillar+mode
    if "pillar" in df.columns:
        t_pillar = _make_reason_table(df, ["pillar","mode"])
    else:
        t_pillar = pd.DataFrame()

    # Save
    p1 = outdir / f"{args.basename}_by_mode.csv"
    t_mode.to_csv(p1, index=False)
    print("[breakdown] wrote:", p1)

    if len(t_pillar):
        p2 = outdir / f"{args.basename}_by_pillar_mode.csv"
        t_pillar.to_csv(p2, index=False)
        print("[breakdown] wrote:", p2)

    # Markdown summary (ready to paste)
    md = []
    md.append(f"# Collapse Breakdown ({summary_path.name})\n")
    md.append("## By mode (all runs)\n")
    cols_show = ["mode","n_all","n_collapsed","collapse_rate"]
    cols_show += [f"all_{r}_rate" for r in ["PI","REP","TOO_SHORT","BUDGET_EXHAUSTED","BUDGET_EXCEEDED","BUDGET","OTHER","NONE"]]
    cols_show = [c for c in cols_show if c in t_mode.columns]
    md.append(_to_md_table(t_mode[cols_show].sort_values("mode")))

    if len(t_pillar):
        md.append("\n## By pillar × mode (collapse_rate + key reasons)\n")
        cols2 = ["pillar","mode","n_all","n_collapsed","collapse_rate",
                 "all_PI_rate","all_REP_rate","all_TOO_SHORT_rate",
                 "all_BUDGET_EXHAUSTED_rate","all_BUDGET_EXCEEDED_rate","all_BUDGET_rate","all_OTHER_rate"]
        cols2 = [c for c in cols2 if c in t_pillar.columns]
        md.append(_to_md_table(t_pillar[cols2].sort_values(["pillar","mode"])))

    md_path = outdir / f"{args.basename}.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("[breakdown] wrote:", md_path)

if __name__ == "__main__":
    main()
