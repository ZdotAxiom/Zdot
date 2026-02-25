#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate an "independent" calibration corpus (v3) intended to keep pi_raw in a
more realistic range for GPT-2 runs (avoid code/JSON heavy patterns).

Design:
- Mix Japanese/English natural prose
- Vary topics, sentence length, and structure
- Avoid braces-heavy / code-like / JSON-like lines
- Keep head-signature diversity
"""
from __future__ import annotations
import argparse, random, textwrap

JA_TOPICS = [
    "生活", "仕事", "研究", "学習", "運動", "睡眠", "体調", "料理", "買い物", "移動",
    "文章", "議論", "要約", "観測", "検証", "再現性", "計画", "振り返り", "注意", "工夫"
]
EN_TOPICS = [
    "planning", "reading", "writing", "debugging", "measurement", "evaluation",
    "workflow", "assumptions", "tradeoffs", "reproducibility", "stress test",
    "interpretation", "summary", "notes", "constraints"
]

JA_FRAMES = [
    "今日は{topic}について、{a}→{b}→{c}の順に整理した。",
    "結論だけ先に言うと、{a}が効いて{b}が残った。だから次は{c}を試す。",
    "私は{topic}で迷ったが、{a}と{b}を比べて{c}を選んだ。",
    "観測の結果、{a}は増えたが{b}は減った。原因候補は{c}だ。",
    "小さく試してから拡大する。まず{a}、次に{b}、最後に{c}。",
    "目的は{a}で、手段は{b}だ。{c}は副作用として扱う。",
    "説明の精度を上げるため、{a}を定義し直し、{b}を区別し、{c}を固定した。",
    "失敗の理由を一つにせず、{a}／{b}／{c}に分解してログに残した。",
]

EN_FRAMES = [
    "I wrote a short note about {topic}: first {a}, then {b}, and finally {c}.",
    "The main point is {a}; {b} is a constraint, and {c} is the next experiment.",
    "In a small run, {a} looked stable while {b} drifted, so I changed {c}.",
    "I prefer consistency over speed: {a} must be fixed before I tune {b} or {c}.",
    "The observation is simple: {a} rises, {b} falls, and {c} explains the gap.",
    "I separated causes into three buckets: {a}, {b}, and {c}, to avoid confusion.",
]

JA_ATOMS = [
    "前提", "手順", "条件", "閾値", "分布", "ばらつき", "比較", "差分", "例外", "原因",
    "検算", "再実行", "固定", "記録", "分析", "改善", "推定", "確認", "要点", "結論"
]
EN_ATOMS = [
    "baseline", "controlled", "threshold", "distribution", "variance", "check",
    "repeat", "fix", "record", "verify", "estimate", "compare", "effect", "cause"
]

# A few "lightly structured" but not code-like templates (no braces-heavy lines)
LIGHT_STRUCT = [
    "メモ: 目的={a} / 重要={b} / 次={c}",
    "観測メモ: {a}が先、{b}が後、{c}は保留",
    "要約: {a}; 次に{b}; 注意点は{c}",
    "Note: goal={a}; constraint={b}; next={c}",
    "Summary: {a}; then {b}; watch {c}",
]

def choose3(rng: random.Random, pool: list[str]) -> tuple[str,str,str]:
    a = rng.choice(pool)
    b = rng.choice([x for x in pool if x != a])
    c = rng.choice([x for x in pool if x != a and x != b])
    return a,b,c

def make_line(rng: random.Random, idx: int) -> str:
    # language mix ratio
    use_ja = rng.random() < 0.68
    if use_ja:
        topic = rng.choice(JA_TOPICS)
        a,b,c = choose3(rng, JA_ATOMS)
        frame = rng.choice(JA_FRAMES)
        line = frame.format(topic=topic, a=a, b=b, c=c)
    else:
        topic = rng.choice(EN_TOPICS)
        a,b,c = choose3(rng, EN_ATOMS)
        frame = rng.choice(EN_FRAMES)
        line = frame.format(topic=topic, a=a, b=b, c=c)

    # add mild variation: optional second sentence, avoid code-like tokens
    if rng.random() < 0.45:
        if use_ja:
            a2,b2,c2 = choose3(rng, JA_ATOMS)
            tail = rng.choice([
                "ただし{a}は過信しない。{b}を見て{c}で確かめる。",
                "{a}を増やすより、{b}を減らして{c}を守る。",
                "まず{a}を一回だけやり、次に{b}を追加し、{c}は後回し。",
            ]).format(a=a2,b=b2,c=c2)
        else:
            a2,b2,c2 = choose3(rng, EN_ATOMS)
            tail = rng.choice([
                "Still, I avoid overfitting: check {a}, then validate {b} with {c}.",
                "I keep it simple: fix {a}, measure {b}, and document {c}.",
                "One step at a time: {a} first, {b} next, {c} last.",
            ]).format(a=a2,b=b2,c=c2)
        line = f"{line} {tail}"

    # occasionally output lightly structured notes (but no JSON / code)
    if rng.random() < 0.18:
        a3,b3,c3 = choose3(rng, JA_ATOMS if use_ja else EN_ATOMS)
        line = rng.choice(LIGHT_STRUCT).format(a=a3,b=b3,c=c3)

    # keep length moderate
    line = " ".join(line.split())
    return line.strip()

def head_sig(s: str, n_words: int = 6) -> str:
    s = " ".join(s.split()).strip().lower()
    if " " in s:
        toks = s.split(" ")
        return " ".join(toks[:n_words])
    return s[:16]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # generate with a simple de-dup on head signatures to keep diversity
    lines = []
    seen = set()
    guard = 0
    while len(lines) < args.n and guard < args.n * 50:
        guard += 1
        line = make_line(rng, len(lines))
        sig = head_sig(line)
        if sig in seen:
            # allow a few duplicates but keep them rare
            if rng.random() < 0.10:
                lines.append(line)
            continue
        seen.add(sig)
        lines.append(line)

    # If we failed to reach n due to de-dup, top up without de-dup strictness
    while len(lines) < args.n:
        lines.append(make_line(rng, len(lines)))

    with open(args.out, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    # basic report
    from collections import Counter
    sigc = Counter(head_sig(l) for l in lines)
    top = sigc.most_common(10)
    print("wrote:", args.out)
    print("n_lines:", len(lines))
    print("unique_head_sig:", len(sigc), "ratio:", round(len(sigc)/len(lines), 4))
    print("top10 head patterns:")
    for k,v in top:
        print(f"{v:3d}  {k}")

if __name__ == "__main__":
    main()
