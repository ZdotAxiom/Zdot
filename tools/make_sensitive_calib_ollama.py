#!/usr/bin/env python3
import argparse, json, time, random, re
from pathlib import Path
import urllib.request

import numpy as np

# ---- SAME as run_vnext_ollama_v1.py (keep consistent) ----
import gzip, math
def gzip_ratio(s: str) -> float:
    b = s.encode("utf-8", errors="ignore")
    if not b:
        return 0.0
    comp = gzip.compress(b)
    return len(comp) / max(1, len(b))

def char_entropy(s: str) -> float:
    if not s:
        return 0.0
    from collections import Counter
    c = Counter(s)
    n = len(s)
    H = 0.0
    for k,v in c.items():
        p = v / n
        H -= p * math.log(p + 1e-12, 2)
    return H

def pi_raw_proxy(s: str) -> float:
    gr = gzip_ratio(s)
    H  = char_entropy(s)
    invH = 1.0 / (H + 1e-6)
    return 0.75 * gr + 0.25 * invH
# ---------------------------------------------------------

def ollama_generate(model: str, prompt: str, temperature: float, top_p: float, num_predict: int, url: str, timeout: int = 120) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        }
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        obj = json.loads(resp.read().decode("utf-8", errors="ignore"))
    return obj.get("response","")

def clean_one_line(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"\n{2,}", "\n", s)
    s = s.strip()
    # 1行コーパス（後工程の扱い簡単）
    s = s.replace("\n", " ⏎ ")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma:7b")
    ap.add_argument("--n_samples", type=int, default=600)          # 300〜1000
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_corpus", default="calib/calibration_corpus_v2_ollama_gemma7b_sensitive.txt")
    ap.add_argument("--base_calib", default="calib/calib_ollama_gemma7b_v1.json")
    ap.add_argument("--out_calib",  default="calib/calib_ollama_gemma7b_v2_sensitive.json")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--num_predict", type=int, default=160)        # 熱いなら 96〜128へ
    ap.add_argument("--min_chars", type=int, default=120)          # 短すぎフィルタ
    ap.add_argument("--max_retries", type=int, default=4)
    ap.add_argument("--api_url", default="http://localhost:11434/api/generate")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 評価プロンプトと被らせない “汎用タスク” テンプレ
    base_texts = [
        "今日は天気が良いので、少し遠回りして帰った。途中で小さな公園を見つけた。",
        "新しいアプリを試したが、設定が多くて最初は戸惑った。少しずつ慣れてきた。",
        "仕事の段取りを見直したら、手戻りが減って気持ちが楽になった。",
        "旅行の計画を立てるとき、移動時間と休憩時間を先に確保すると失敗しにくい。",
        "買い物リストを作ると無駄が減るが、空腹時は判断が雑になりがちだ。",
        "短い文章でも、主語と目的が明確だと読み手の負担が減る。",
        "会議のメモは『決定事項』『宿題』『期限』だけ抜き出すと後で使いやすい。",
        "説明は具体例→一般化→注意点の順にすると誤解が減る。",
        "小さな習慣を積むほうが、いきなり大きく変えるより継続しやすい。",
        "同じ作業でも、最初にゴール条件を書いておくと迷走しない。"
    ]

    templates = [
        ("要約", "次の文章を50〜80字で要約して。文章：\n{txt}"),
        ("言い換え", "次の文章を丁寧語に言い換えて。文章：\n{txt}"),
        ("箇条書き", "次の文章の要点を3つ、箇条書きで。文章：\n{txt}"),
        ("手順化", "次の内容を手順として5ステップに分解して。内容：\n{txt}"),
        ("分類", "次の文は『事実』『意見』『感想』のどれ？理由は1行。文：\n{txt}"),
        ("短い推論", "次の状況で最も起こりやすい問題を1つ挙げ、対策を1つ。状況：\n{txt}"),
        ("英訳", "次の日本語を自然な英語に翻訳して：\n{txt}"),
        ("要点抽出", "次の文章から『目的』『制約』『次の行動』を抽出して。文章：\n{txt}"),
        ("整形", "次の文章を読みやすく整形して（改行と句読点を調整）。文章：\n{txt}"),
        ("短文生成", "次のテーマで3文だけ書いて：テーマ={theme}"),
        ("比較", "AとBの違いを2点だけ説明して：A={a}, B={b}"),
        ("定義", "次の用語を中学生向けに1〜2文で説明して：{term}"),
    ]

    themes = ["時間管理", "健康", "学習", "コミュニケーション", "旅行", "料理", "仕事の段取り", "片付け"]
    pairs  = [("紙のメモ","デジタルメモ"),("運動","睡眠"),("集中","休憩"),("計画","実行")]
    terms  = ["優先順位", "トレードオフ", "再現性", "バイアス", "要約", "推論"]

    out_path = Path(args.out_corpus)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    texts = []
    pis   = []

    need = args.n_samples
    i = 0
    while len(texts) < need:
        kind, tmpl = rng.choice(templates)
        txt = rng.choice(base_texts)

        prompt = tmpl.format(
            txt=txt,
            theme=rng.choice(themes),
            a=rng.choice(pairs)[0],
            b=rng.choice(pairs)[1],
            term=rng.choice(terms),
        )

        # 失敗・短文はリトライ
        ok = False
        for r in range(args.max_retries):
            try:
                resp = ollama_generate(
                    model=args.model,
                    prompt=prompt,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_predict=args.num_predict,
                    url=args.api_url,
                )
                s = clean_one_line(resp)
                if len(s) >= args.min_chars:
                    ok = True
                    break
            except Exception as e:
                time.sleep(0.4 * (r + 1))
                continue

        if not ok:
            continue

        pi = pi_raw_proxy(s)
        texts.append(s)
        pis.append(pi)

        i += 1
        if len(texts) % 25 == 0:
            print(f"[gen] {len(texts)}/{need} samples ...")

    # write corpus (1 sample per line)
    out_path.write_text("\n".join(texts) + "\n", encoding="utf-8")
    a = np.asarray(pis, dtype=float)

    q50 = float(np.quantile(a, 0.50))
    q95 = float(np.quantile(a, 0.95))

    # build calib json
    base = json.loads(Path(args.base_calib).read_text(encoding="utf-8"))
    base["calibration_corpus"] = str(out_path)
    base["q50"] = q50
    base["q95"] = q95
    note = base.get("note","")
    extra = f"gemma:7b sensitive calib | n={need} | temp={args.temperature} top_p={args.top_p} num_predict={args.num_predict} | seed={args.seed}"
    base["note"] = (note + " | " + extra).strip(" |")

    Path(args.out_calib).write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n[DONE] corpus:", out_path)
    print("[DONE] calib :", args.out_calib)
    print("pi_raw_proxy stats min/med/mean/p95/max:",
          float(a.min()), float(np.median(a)), float(a.mean()), float(np.quantile(a,0.95)), float(a.max()))
    print("q50:", q50, "q95:", q95)

if __name__ == "__main__":
    main()
