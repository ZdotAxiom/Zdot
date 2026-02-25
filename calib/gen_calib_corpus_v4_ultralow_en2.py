#!/usr/bin/env python3
import argparse, random

TIME = ["Today", "Now", "Later", "This morning", "This evening", "Yesterday", "Tomorrow"]
TEMPL = [
  "I think so.",
  "I don't know.",
  "I can do that.",
  "I will do it.",
  "I want to help.",
  "It is fine.",
  "It is okay.",
  "It is a good idea.",
  "That makes sense.",
  "This is simple.",
  "This is important.",
  "Please wait.",
  "Thank you.",
  "{time}, I will try again.",
  "{time}, I will check it.",
  "{time}, I will do it now.",
  "{time}, I will write it down.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    lines = []
    for _ in range(args.n):
        t = rng.choice(TEMPL)
        lines.append(t.format(time=rng.choice(TIME)))

    with open(args.out, "w", encoding="utf-8") as f:
        for l in lines:
            f.write(l + "\n")

    sig = {l[:20] for l in lines}
    print("wrote:", args.out)
    print("n_lines:", len(lines))
    print("unique_head20:", len(sig), "ratio:", round(len(sig)/len(lines), 4))

if __name__ == "__main__":
    main()
