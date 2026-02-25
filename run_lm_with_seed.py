import argparse, sys, random, runpy
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def main():
    ap = argparse.ArgumentParser(description="Seed wrapper (runpy) for SES Collapse Lab")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--script", type=str, default="z_collapse_lab_SES_v2_4_2_no_text_fix.py")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="Args passed to target script. Put `--` before them.")
    args = ap.parse_args()

    rest = args.rest
    if rest and rest[0] == "--":
        rest = rest[1:]

    set_seed(args.seed)

    # Forward argv and execute the target script as a file
    sys.argv = [args.script] + rest
    runpy.run_path(args.script, run_name="__main__")

if __name__ == "__main__":
    main()
