import argparse, json, random, inspect
from pathlib import Path

import numpy as np
import pandas as pd

# あなたが使ってる実体
import z_collapse_lab_SES_v2_4_2_no_text_fix as lab


def read_blocks(path: Path):
    raw = path.read_text(encoding="utf-8")
    # 空行区切りを想定（prompts_100.txt でやってた方式と同じ）  [oai_citation:3‡1月11日ターミナル書き出し.txt](sediment://file_000000000dcc71fdbd79e0bb9591b3aa)
    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    return blocks


def set_all_seeds(seed: int):
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
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seed_from", type=int, default=0)
    ap.add_argument("--seed_to", type=int, default=499)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--model_name", default="distilgpt2")
    ap.add_argument("--device", default="mps")  # cpu/cuda/mps
    ap.add_argument("--condition", default="META")
    ap.add_argument("--apply_zcp", action="store_true", default=True)
    ap.add_argument("--no_text", action="store_true", default=True)  # 軽量ログ
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    prompts = read_blocks(Path(args.prompts))
    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))

    # モデル読み込み（NLL用途）: calib側がuse_nllでなければ軽い
    model = tok = None
    try:
        sig = inspect.signature(lab._maybe_load_hf_lm_for_nll)
        # だいたい (model_name, device, ...) なので両方渡す
        model, tok, _dev = lab._maybe_load_hf_lm_for_nll(args.model_name, device=args.device)
    except Exception:
        pass

    # run_lm_generation_with_zcp のシグネチャに合わせて kwargs を組む
    fn = lab.run_lm_generation_with_zcp
    fn_sig = inspect.signature(fn)

    for seed in range(args.seed_from, args.seed_to + 1):
        set_all_seeds(seed)

        kwargs = {}
        # ありがちな引数名を「存在するものだけ」入れる（壊れにくい）
        cand = {
            "prompts": prompts,
            "calib": calib,
            "condition": args.condition,
            "apply_zcp": args.apply_zcp,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "model_name": args.model_name,
            "device": args.device,
            "model": model,
            "tokenizer": tok,
            "tok": tok,
            "seed": seed,
            "no_text": args.no_text,
        }
        for k, v in cand.items():
            if k in fn_sig.parameters:
                kwargs[k] = v

        df = fn(**kwargs)
        if not isinstance(df, pd.DataFrame):
            raise SystemExit(f"[FATAL] run_lm_generation_with_zcp returned {type(df)}; signature={fn_sig}")

        df["seed"] = seed
        out_csv = outdir / f"g36_meta_MAX{args.max_new_tokens}_seed{seed}.csv"
        df.to_csv(out_csv, index=False)
        print("[ok]", out_csv)

    print("[done] META seeds", args.seed_from, "..", args.seed_to, "->", outdir)


if __name__ == "__main__":
    main()
