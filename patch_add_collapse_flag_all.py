import glob, json
from pathlib import Path

def patch_file(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    meta = None
    out = []
    changed = 0

    for line in lines:
        if not line.strip():
            continue
        obj = json.loads(line)

        if obj.get("record_type") == "meta":
            meta = obj
            out.append(obj)
            continue

        if obj.get("record_type") == "step":
            if "collapse_flag" in obj:
                out.append(obj)
                continue

            # epsilon_pi は meta から取る（無ければ 0.85）
            eps = 0.85
            if meta is not None and "epsilon_pi" in meta:
                eps = float(meta["epsilon_pi"])

            # pi_norm があればそれで collapse_flag を作る
            if "pi_norm" in obj and obj["pi_norm"] is not None:
                obj["collapse_flag"] = 1 if float(obj["pi_norm"]) > eps else 0
            else:
                obj["collapse_flag"] = 0

            out.append(obj)
            changed += 1
            continue

        out.append(obj)

    if changed > 0:
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out) + "\n", encoding="utf-8")

    return changed

if __name__ == "__main__":
    paths = sorted(Path(p) for p in glob.glob("exp/runs/baseline/HUM/seed_*.jsonl"))
    if not paths:
        raise SystemExit("no files found under exp/runs/baseline/HUM/")

    total_changed_files = 0
    total_added = 0

    for p in paths:
        added = patch_file(p)
        if added > 0:
            total_changed_files += 1
            total_added += added

    print(f"patched_files={total_changed_files} / {len(paths)}  added_collapse_flag_rows={total_added}")
