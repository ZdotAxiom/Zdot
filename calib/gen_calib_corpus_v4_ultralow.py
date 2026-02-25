#!/usr/bin/env python3
import argparse, random

SUBJ = ["私","彼","彼女","父","母","友人","同僚","猫","犬"]
ACT  = ["歩く","寝る","食べる","飲む","読む","書く","洗う","買う","待つ","片付ける","出かける","戻る"]
OBJ  = ["水","お茶","コーヒー","パン","ご飯","本","メール","鍵","傘","靴","音楽","皿"]
TIME = ["朝","昼","夕方","夜","今","さっき","明日","週末"]
PLACE= ["家","会社","駅","店","公園","台所","部屋","道"]

def make_line(rng):
    # できるだけ短く・日常・単文中心（pi_rawを下げる）
    t = rng.random()
    if t < 0.55:
        # 例: "朝、私が水を飲む。"
        return f"{rng.choice(TIME)}、{rng.choice(SUBJ)}が{rng.choice(OBJ)}を{rng.choice(ACT)}。"
    elif t < 0.80:
        # 例: "家に戻る。"
        return f"{rng.choice(PLACE)}に{rng.choice(ACT)}。"
    elif t < 0.92:
        # 例: "今日は晴れ。"
        return rng.choice(["今日は晴れ。","風が強い。","少し寒い。","眠い。","静かだ。","忙しい。","落ち着く。"])
    else:
        # たまに2句（ただし短い）
        return f"{rng.choice(TIME)}に{rng.choice(PLACE)}へ行く。{rng.choice(OBJ)}を{rng.choice(ACT)}。"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    lines = [make_line(rng) for _ in range(args.n)]

    with open(args.out, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    print("wrote:", args.out)
    print("n_lines:", len(lines))
    # 簡易チェック：先頭12文字のユニーク率
    sig = {l[:12] for l in lines}
    print("unique_head12:", len(sig), "ratio:", round(len(sig)/len(lines), 4))

if __name__ == "__main__":
    main()
