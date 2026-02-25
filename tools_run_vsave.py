import argparse
import json
import subprocess
import time
from pathlib import Path

def is_jsonl(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("r", encoding="utf-8") as f:
        ch = f.read(1)
    return ch == "{"

def run_cmd(cmd, log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as lf:
        lf.write("\n" + "="*80 + "\n")
        lf.write("[CMD] " + " ".join(cmd) + "\n")
        lf.write("="*80 + "\n")
        proc = subprocess.run(cmd, stdout=lf, stderr=lf)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (see {log_path})")

def patch_jsonl(out_path: Path, mode: str, runtime_sec: float, decoder: dict):
    lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    objs = [json.loads(l) for l in lines]

    meta = objs[0]
    steps = [o for o in objs if o.get("record_type") == "step"]
    finals = [o for o in objs if o.get("record_type") == "final"]
    if not finals:
        raise RuntimeError(f"final record missing in {out_path}")
    final = finals[0]

    meta["decoder"] = decoder

    for s in steps:
        it = s.get("intervention_type")
        if it is None:
            s["intervention_type"] = "NONE"
        elif isinstance(it, str) and it.lower() == "none":
            s["intervention_type"] = "NONE"

    if mode == "baseline":
        for s in steps:
            s["intervened"] = 0
            s["intervention_type"] = "NONE"
        final["n_interventions"] = 0
        final["extra_tokens_due_to_intervention"] = 0.0
        final["H_pre"] = None

    final["runtime_sec"] = float(runtime_sec)

    out_lines = [json.dumps(meta, ensure_ascii=False)]
    out_lines += [json.dumps(o, ensure_ascii=False) for o in steps]
    out_lines += [json.dumps(final, ensure_ascii=False) for _ in [0]]
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pillar", type=str, required=True)   # HUM/STEM/ETH/TEMP/META
    ap.add_argument("--mode", type=str, required=True)     # baseline/controlled
    ap.add_argument("--script", type=str, required=True)
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--calib", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--apply_zcp", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=512)

    # ★ここが修正点：no_text を切り替え可能に
    ap.add_argument("--no_text", type=int, default=1, choices=[0,1],
                    help="1: save no text (light), 0: keep text for pi/collapse debug")

    args, extras = ap.parse_known_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    decoder = {
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "max_new_tokens": int(args.max_new_tokens),
    }

    log_path = Path("exp/logs") / f"run_{args.pillar}_{args.mode}_seed_{args.seed:03d}.log"

    cmd = [
        "python", "run_lm_with_seed.py",
        "--seed", str(args.seed),
        "--script", args.script,
        "--",
        "run_lm",
        "--prompts", args.prompts,
        "--calib", args.calib,
        "--out", str(out_path),
        "--condition", args.mode,
    ]

    # ★no_text=1 のときだけ付ける（本番軽量）
    if args.no_text == 1:
        cmd.append("--no_text")

    if args.apply_zcp:
        cmd.append("--apply_zcp")
    # pass-through extra args to run_lm
    cmd += extras

    t0 = time.perf_counter()
    run_cmd(cmd, log_path)
    t1 = time.perf_counter()
    runtime_sec = t1 - t0

    if not is_jsonl(out_path):
        conv_cmd = [
            "python", "postprocess_to_vsave.py",
            "--infile", str(out_path),
            "--outfile", str(out_path),
            "--mode", args.mode,
            "--pillar", args.pillar,
            "--seed", str(args.seed),
        ]
        run_cmd(conv_cmd, log_path)

    patch_jsonl(out_path, args.mode, runtime_sec, decoder)
    print(f"[OK] {args.pillar} {args.mode} seed={args.seed:03d} runtime_sec={runtime_sec:.3f} -> {out_path}")

if __name__ == "__main__":
    main()
