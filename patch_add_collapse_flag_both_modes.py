import glob, json
from pathlib import Path

TARGETS = [
    "exp/runs/baseline/HUM/seed_*.jsonl",
    "exp/runs/controlled/HUM/seed_*.jsonl",
]

def patch_file(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    out=[]
    changed=0
    for line in lines:
        if not line.strip():
            continue
        obj=json.loads(line)

        if obj.get("record_type")=="step":
            if "collapse_flag" not in obj:
                # safest: if pi_flag/rep_flag exist, use them
                pi = int(obj.get("pi_flag", 0))
                rep = int(obj.get("rep_flag", 0))
                # if they don't exist, fall back to collapse_by_* style if present
                obj["collapse_flag"] = int(bool(pi) or bool(rep))
                changed += 1

        out.append(obj)

    if changed:
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")

    return changed

if __name__=="__main__":
    paths=[]
    for pat in TARGETS:
        paths += [Path(p) for p in glob.glob(pat)]
    paths = sorted(paths)

    if not paths:
        raise SystemExit("no target jsonl files found")

    total_files=0
    total_rows=0
    for p in paths:
        n = patch_file(p)
        if n:
            total_files += 1
            total_rows += n

    print(f"patched_files={total_files}/{len(paths)} added_collapse_flag_rows={total_rows}")
