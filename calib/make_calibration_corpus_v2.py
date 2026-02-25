#!/usr/bin/env python3
import argparse, random, re, hashlib
from pathlib import Path

JP_TOPICS = [
    "再現性", "仮定", "検証", "計測", "誤差", "境界条件", "因果", "反例", "デバッグ", "仕様",
    "ログ", "温度", "確率", "分布", "評価", "観測", "統計", "最適化", "設計", "失敗"
]
EN_TOPICS = [
    "reproducibility", "assumption", "verification", "measurement", "error", "edge case", "causality",
    "counterexample", "debugging", "specification", "logging", "temperature", "probability",
    "distribution", "evaluation", "observation", "statistics", "optimization", "design", "failure"
]

JP_VERBS = ["整理した", "比較した", "点検した", "記録した", "観測した", "推定した", "要約した", "改善した", "分解した", "固定した"]
EN_VERBS = ["summarized", "compared", "checked", "logged", "observed", "estimated", "refined", "debugged", "validated", "fixed"]

def norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def head_sig(s: str) -> str:
    """
    単調さ抑制用の簡易署名。
    - 英文: 先頭6トークン
    - 和文(空白少): 先頭16文字
    """
    s = norm_space(s)
    if " " in s:
        toks = s.split(" ")
        return " ".join(toks[:6]).lower()
    return s[:16]

def one_line_code(rng: random.Random) -> str:
    snippets = [
        "for i in range(3): x += i  # tiny loop",
        "if score > 0.8: flag = True  # threshold",
        "data = {'a':1,'b':2}; print(len(data))",
        "import math; y = math.log(1+z)",
        "assert n_samples >= 100, 'too small'",
        "x = (q - q50) / (q95 - q50 + 1e-9)",
        "def f(x): return (x*x + 1) % 97",
        "try: parse(s)  # fallback; except: pass",
    ]
    return rng.choice(snippets)

def one_line_jsonish(rng: random.Random) -> str:
    opts = [
        '{"goal":"sanity-check","n":128,"note":"keep it simple"}',
        '{"mode":"baseline","temp":1.2,"top_p":0.95,"seed":7}',
        '{"metric":"pi_raw","q50":0.78,"q95":0.95,"eps":0.85}',
        '{"event":"step","idx":42,"rep":0.12,"pi":0.66}',
        '{"result":"ok","runtime_sec":12.3,"tokens":512}',
    ]
    return rng.choice(opts)

def one_line_math(rng: random.Random) -> str:
    forms = [
        "Let p(x)=softmax(z)_i; compare Δ = p_a - p_b under the same seed.",
        "We use pi_norm = clamp((pi_raw-q50)/(q95-q50), 0, 1) and watch its drift.",
        "A quick check: median ≠ mean when the distribution is skewed; log both.",
        "If k=3 consecutive flags occur, treat it as sustained anomaly, not noise.",
        "Compute RR and OR with smoothing to avoid zero-division in small samples.",
    ]
    return rng.choice(forms)

def jp_sentence(rng: random.Random) -> str:
    t1 = rng.choice(JP_TOPICS)
    t2 = rng.choice(JP_TOPICS)
    v = rng.choice(JP_VERBS)
    style = rng.choice(["plain", "list", "quote", "numbers", "warning"])
    if style == "plain":
        return f"{t1}の観点で{t2}を{v}が、結論は断定せずにログへ残した。"
    if style == "list":
        return f"{t1}メモ: 1)前提 2)手順 3)結果 — {t2}は短く、解釈は後回し。"
    if style == "quote":
        return f"「{t1}は目的ではなく手段」— {t2}を{v}後に、例外条件だけ追記した。"
    if style == "numbers":
        a = rng.randint(2,9); b = rng.randint(10,99)
        return f"{t1}チェック: 係数{a}、試行{b}回、外れ値は別枠で{t2}を{v}。"
    return f"注意: {t1}が過剰に固定されると{t2}が見えなくなるので、揺らぎを許して{v}。"

def en_sentence(rng: random.Random) -> str:
    t1 = rng.choice(EN_TOPICS)
    t2 = rng.choice(EN_TOPICS)
    v = rng.choice(EN_VERBS)
    style = rng.choice(["plain", "checklist", "contrast", "caveat", "microstory"])
    if style == "plain":
        return f"I {v} the {t1} notes and kept {t2} separate from interpretation."
    if style == "checklist":
        return f"Checklist: define {t1}; test an edge case; log {t2}; avoid overfitting."
    if style == "contrast":
        return f"Baseline vs controlled: hold seeds fixed, then compare {t1} and {t2} only."
    if style == "caveat":
        return f"Caveat: when {t1} is too strict, {t2} may look flat; report sensitivity."
    return f"In a small run, I {v} one anomaly, then wrote a short note about {t1}→{t2}."

def mixed_line(rng: random.Random) -> str:
    kind = rng.choice(["JP", "EN", "MATH", "CODE", "JSON"])
    if kind == "JP":
        return jp_sentence(rng)
    if kind == "EN":
        return en_sentence(rng)
    if kind == "MATH":
        return one_line_math(rng)
    if kind == "CODE":
        return one_line_code(rng)
    return one_line_jsonish(rng)

def build_corpus(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    lines = []
    sig_count = {}
    max_per_sig = 3  # 同一head_sigの上限（単調さ抑制のキモ）

    # 生成候補の比率を少しコントロール（多様性確保）
    # JP/EN多め、MATH/CODE/JSONを散発的に混ぜる
    weights = [
        ("JP", 0.38),
        ("EN", 0.38),
        ("MATH", 0.10),
        ("CODE", 0.07),
        ("JSON", 0.07),
    ]
    kinds, probs = zip(*weights)

    def pick_kind() -> str:
        r = rng.random()
        acc = 0.0
        for k, p in weights:
            acc += p
            if r <= acc:
                return k
        return "EN"

    attempts = 0
    while len(lines) < n and attempts < n * 200:
        attempts += 1
        kind = pick_kind()
        if kind == "JP":
            s = jp_sentence(rng)
        elif kind == "EN":
            s = en_sentence(rng)
        elif kind == "MATH":
            s = one_line_math(rng)
        elif kind == "CODE":
            s = one_line_code(rng)
        else:
            s = one_line_jsonish(rng)

        s = norm_space(s)

        # 1行が短すぎ/長すぎを避ける（でもバリエーションは残す）
        if len(s) < 20 or len(s) > 140:
            continue

        sig = head_sig(s)
        if sig_count.get(sig, 0) >= max_per_sig:
            continue

        lines.append(s)
        sig_count[sig] = sig_count.get(sig, 0) + 1

    if len(lines) < n:
        raise RuntimeError(f"Failed to build enough diverse lines: got {len(lines)}/{n}")
    return lines

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output txt path")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    lines = build_corpus(args.n, args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines) + "\n"
    out.write_text(text, encoding="utf-8")

    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    print(f"wrote: {out}")
    print(f"n_lines: {len(lines)}")
    print(f"sha256: {sha}")
    print("head3:", lines[:3])
    print("tail3:", lines[-3:])

if __name__ == "__main__":
    main()
