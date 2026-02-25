import json, glob
from pathlib import Path

# 対象：必要なら pillar を広げてOK
TARGETS = [
    "exp/runs/baseline/HUM/seed_*.jsonl",
    "exp/runs/controlled/HUM/seed_*.jsonl",
]

def to_int01(x):
    try:
        return 1 if int(x) != 0 else 0
    except Exception:
        return 0

def patch_file(path: Path) -> int:
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    changed = 0

    for line in lines:
        if not line.strip():
            continue
        o = json.loads(line)

        if o.get("record_type") == "step":
            # 既存の pi_flag / rep_flag を最優先で使う
            pi_flag = to_int01(o.get("pi_flag", 0))
            rep_flag = to_int01(o.get("rep_flag", 0))

            new_collapse_flag = 1 if (pi_flag or rep_flag) else 0

            # 既存値と違うなら上書き
            old = o.get("collapse_flag", None)
            if old is None or to_int01(old) != new_collapse_flag:
                o["collapse_flag"] = int(new_collapse_flag)
                changed += 1

        out.append(o)

    if changed > 0:
        bak = path.with_suffix(path.suffix + ".bak_cf")
        bak.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")

    return changed

def main():
    paths=[]
    for pat in TARGETS:
        paths += [Path(p) for p in glob.glob(pat)]
    paths = sorted(paths)
    if not paths:
        raise SystemExit("no target jsonl files found")

    total_files = 0
    total_rows = 0
    for p in paths:
        n = patch_file(p)
        if n:
            total_files += 1
            total_rows += n

    print(f"patched_files={total_files}/{len(paths)} rewritten_step_rows={total_rows}")

if __name__ == "__main__":
    main()
