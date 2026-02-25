# Ż Collapse Lab vNEXT (Reproducible Protocol)

## Goal
Provide a **controlled, versioned, fully replayable** collapse-prevention evaluation protocol (not a benchmark race).

## Fixed Spec (vNEXT)
- Pillars: HUM / STEM / ETH / TEMP / META
- Prompts: `prompts_vnext/pillar_*_40.txt` (40 blocks each; selected by `seed % N`)
- Seeds:
  - Smoke: `0..29` (N=30)
  - Full: `0..499` (N=500)
- Modes: `baseline` vs `controlled` (all else identical)
- Decoder:
  - `max_new_tokens=512`, `temperature=1.2`, `top_p=0.95`
- Collapse definition (legacy):
  - `epsilon_pi=0.85`, `k_consecutive=3`, `min_tokens=64`
  - repetition: `rep_ngram=3`, `rep_threshold=0.20`
- Intervention budget:
  - default: `max_interventions=100`
  - TEMP optional stress-test: `max_interventions=200` (separate experiment)

## Outputs
- Per-run logs: `exp_vnext/runs/<mode>/<pillar>/seed_###.jsonl`
- Summary CSV: `exp_vnext/summary_ALL/runs_summary_all.csv`
- Table 1: `exp_vnext/summary_ALL/table1_vnext.{csv,md}`

## Run (Smoke 30)
```bash
for pillar in HUM STEM ETH TEMP META; do
  for mode in baseline controlled; do
    mkdir -p exp_vnext/runs/$mode/$pillar
    for s in $(seq 0 29); do
      python run_vnext_with_seed_v5.py \
        --exp_id vNEXT \
        --seed $s \
        --mode $mode \
        --pillar $pillar \
        --prompts prompts_vnext/pillar_${pillar}_40.txt \
        --calib calib/calib.json \
        --out exp_vnext/runs/$mode/$pillar/seed_$(printf "%03d" $s).jsonl \
        --model_id gpt2 \
        --device cpu \
        --max_new_tokens 512 \
        --temperature 1.2 \
        --top_p 0.95 \
        --epsilon_pi 0.85 \
        --k_consecutive 3 \
        --min_tokens 64 \
        --rep_ngram 3 \
        --rep_threshold 0.20 \
        --max_interventions 100 || exit 1
    done
  done
done
```

## Ollama (gemma:7b, 512 fixed)
See `README_vNEXT_OLLAMA_512.md` for the dedicated, reproducible 512-token Ollama protocol and commands.

## Validate + Summarize
```bash
rm -rf exp_vnext/summary_ALL
python validate_and_summarize_vnext.py \
  --runs_root exp_vnext/runs \
  --outdir exp_vnext/summary_ALL \
  --expect 30
```

## Make Table 1 (vNEXT)
```bash
python make_table1_vnext.py \
  --summary_csv exp_vnext/summary_ALL/runs_summary_all.csv \
  --out_csv exp_vnext/summary_ALL/table1_vnext.csv \
  --out_md exp_vnext/summary_ALL/table1_vnext.md \
  --bootstrap 20000 \
  --seed 0
```

## Sanity Check (TEMP controlled collapse must be empty)
```bash
python - <<'PY'
import pandas as pd
df=pd.read_csv("exp_vnext/summary_ALL/runs_summary_all.csv")
x=df[(df["pillar"]=="TEMP")&(df["mode"]=="controlled")&(df["collapse_run"]==1)]
print(x[["seed","collapse_reason","stop_aux_reason","collapse_core_reason",
         "budget_exhausted","n_interventions","n_tokens_total"]].to_string(index=False))
PY
```

## Taxonomy (vNEXT)
- `collapse_run / collapse_reason`: legacy collapse label (compat).
- `collapse_core / collapse_core_reason`: core collapse only (PI/REP).
- `stop_aux / stop_aux_reason`: auxiliary stop reasons (TOO_SHORT / BUDGET_EXHAUSTED).

## Notes
- If code changes, existing JSONL logs do not retroactively update.
- Always re-run `validate_and_summarize_vnext.py` to regenerate the summary CSV.
- TOO_SHORT判定は tokens_checked_for_tooshort := n_tokens_total（生成済み総トークン数）を用い、tokens_checked_for_tooshort < min_tokens でフラグ。
