import argparse, sys, random
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--script", default="z_collapse_lab_SES_v2_4_2_no_text_fix.py")
    ap.add_argument("rest", nargs=argparse.REMAINDER)  # 先頭に -- を付けて渡す
    args = ap.parse_args()

    # rest は ["--", "..."] になるので、先頭の "--" を落とす
    rest = args.rest
    if rest and rest[0] == "--":
        rest = rest[1:]

    set_seed(args.seed)

    # 本体を import して main() を同一プロセスで叩く
    mod = __import__(args.script.replace(".py",""))
    sys.argv = [args.script] + rest
    mod.main()

if __name__ == "__main__":
    main()
