#!/usr/bin/env python3
import argparse, random

SUBJ = ["I","You","We","They","He","She"]
VERB = ["go","walk","eat","drink","read","write","sleep","work","play","wait","look","think"]
OBJ  = ["water","tea","coffee","bread","rice","a book","an email","a key","a bag","music","a chair","a note"]
PLACE= ["home","work","the store","the park","the room","the kitchen"]
TIME = ["today","now","later","this morning","this evening","yesterday","tomorrow"]

TEMPL = [
  "{time}, {subj} {verb}.",
  "{time}, {subj} {verb} {obj}.",
  "{subj} is at {place}.",
  "{subj} {verb}s at {place}.",
  "{subj} {verb}s {obj}.",
  "It is warm.",
  "It is cold.",
  "It is quiet.",
  "I feel tired.",
  "I feel fine.",
  "I am ready.",
  "I am done.",
]

def make_line(rng):
    t = rng.choice(TEMPL)
    subj = rng.choice(SUBJ)
    verb = rng.choice(VERB)
    obj_ = rng.choice(OBJ)
    place = rng.choice(PLACE)
    time = rng.choice(TIME)
    s = t.format(subj=subj, verb=verb, obj=obj_, place=place, time=time)
    return s

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

    sig = {l[:16] for l in lines}
    print("wrote:", args.out)
    print("n_lines:", len(lines))
    print("unique_head16:", len(sig), "ratio:", round(len(sig)/len(lines), 4))

if __name__ == "__main__":
    main()
