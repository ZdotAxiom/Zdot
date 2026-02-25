# Ż Collapse Lab — vSAVE-1.1.1 (Paper-Fixed Spec)

**Goal:** Lock down the experiment spec so NeurIPS reviewers can’t attack  
**baseline不足 / methods曖昧 / 恣意性 / コスト / 再現性**.

This document is the **single source of truth** for the paper.  
**Runner code MUST match this spec.** If code and README disagree, **README wins**.

---

## 0) What’s new in vSAVE-1.1.1 (vs vSAVE-1.1)

This update increases **objectivity** without adding researcher degrees-of-freedom:

1) **Two token indices**
- `token_idx_ctx` (context length; can shrink after intervention/reset)
- `token_idx_global` (monotonic cumulative; never decreases)

✅ Fixes “token_idx monotonic” vs “reset shortens context” contradiction.

2) **TOO_SHORT is based on generated tokens**
- `generated_tokens = token_idx_global - prompt_len_tokens`
- `too_short_flag = 1[generated_tokens < min_tokens]`

✅ Removes prompt-length dependence and makes the criterion comparable across pillars.

3) **Collapse definition is identical across baseline/controlled**
- Step-level `collapse_flag = OR(pi_flag, rep_flag)`
- Run-level `collapse_run = 1 iff collapse_flag holds for k_consecutive steps`
- OR `final_generated_tokens < min_tokens` at end-of-run

✅ Eliminates any suspicion of “controlled-friendly definition”.

4) **Timestamps are required**
- `timestamp_start`, `timestamp_end` in meta/footer (ISO 8601)

✅ Stronger runtime/cost accountability.

5) Optional but recommended: **pre/post logging for intervention steps**
- Keep `*_post` fields **only as additional info** (primary metrics remain “pre”).

✅ Makes intervention behavior auditable, while preserving compatibility.

---

## A) Experimental Units (Fully Deterministic)

- `pillar ∈ {HUM, STEM, ETH, TEMP, META}`
- `seed ∈ {0..499}`
- `mode ∈ {baseline, controlled}`
- **1 run = 1 jsonl file** (append prohibited)
- Baseline vs controlled differs **ONLY** by allowing interventions (`mode`)
  - prompts / decoder / π calibration / collapse rules / logging schema are identical

---

## B) Prompts (Pillar-Specific, Deterministic Choice)

- `prompts/pillar_XXX_20.txt` (blocks separated by blank lines)
- `prompt_id = seed % N`
- **Same seed → same prompt_id in both baseline and controlled**

---

## C) Calibration (Leak-Proof)

`calib/calib.json` contains complexity calibration percentiles:

- `q50`, `q95` **must be estimated from an independent calibration corpus**
  - e.g., `N≈300` fixed texts, not from any run logs (no seed0 leakage)
- **1-page π normalization (fixed):**
  - `pi_norm = clamp((pi_raw - q50) / (q95 - q50))`

---

## D) Collapse Detection (Two-Level, Paper-Fixed)

### D.1 Step-level signals (must be logged)
- `pi_raw`, `pi_norm`
- `rep_score` (n-gram repetition score)
- `generated_tokens`
- Flags:
  - `pi_flag = 1[pi_norm > epsilon_pi]`
  - `rep_flag = 1[rep_score > rep_threshold]`

### D.2 Step-level collapse flag (paper-fixed)
```
collapse_flag = 1 if (pi_flag OR rep_flag) else 0
```

### D.3 Run-level collapse (paper-fixed)
- Maintain `collapse_streak`
- `collapse_run = 1` iff `collapse_flag==1` holds for `k_consecutive` consecutive steps
- OR `final_generated_tokens < min_tokens` at end-of-run

**Recommended fixed values (paper-default):**
- `epsilon_pi = 0.85`
- `k_consecutive = 3`
- `min_tokens = 64`
- `rep_ngram = 3`
- `rep_threshold = 0.20`
- `rep_tail_chars = 2000`

---

## E) Intervention (Controlled Only)

### E.1 When to intervene
Intervention is allowed **only** in `mode=controlled`.

- If `collapse_streak >= k_consecutive`, attempt intervention.
- Intervention budget and deterministic selection must be fixed.

### E.2 Intervention rate (paper-fixed)
**Primary (Main paper):**
- `intervene_rate_primary = mean 1[n_interventions > 0]`

**Secondary (Appendix):**
- `intervene_density_secondary = mean (n_interventions / n_tokens_total)`

---

## F) Logging Schema (All Pillars / All Modes)

Each run is JSONL:

1) `meta` (first line)
2) `step` lines
3) `final` (last line)

### F.1 Meta record (required keys)

```json
{
  "record_type": "meta",
  "exp_id": "...",
  "mode": "baseline|controlled",
  "pillar": "HUM|STEM|ETH|TEMP|META",
  "seed": 123,
  "model_id": "gpt2",
  "device": "cpu",
  "decoder": {"temperature": 1.2, "top_p": 0.95, "max_new_tokens": 512},

  "timestamp_start": "2026-01-21T08:00:00+09:00",

  "prompt_id": 7,
  "prompt_len_tokens": 75,

  "pi_source": "complexity",
  "q50": 0.91,
  "q95": 1.24,
  "epsilon_pi": 0.85,
  "k_consecutive": 3,

  "min_tokens": 64,
  "min_generated_tokens": 64,
  "rep_ngram": 3,
  "rep_threshold": 0.20,
  "rep_tail_chars": 2000,

  "ctx_limit": 1024,
  "token_idx_schema": "token_idx_ctx + token_idx_global (global monotonic)",
  "too_short_schema": "generated_tokens = token_idx_global - prompt_len_tokens",
  "too_short_eval": "final_only",
  "collapse_flag_spec": "collapse_flag = 1 if (pi_flag OR rep_flag) else 0"
}
```

### F.2 Step record (required keys)

```json
{
  "record_type": "step",
  "step_idx": 0,

  "token_idx_ctx": 76,
  "token_idx_ctx_pre": 76,
  "token_idx_ctx_post": null,
  "token_idx_global": 1,
  "generated_tokens": 1,

  "pi_raw": 1.03,
  "pi_norm": 0.77,
  "pi_flag": 0,

  "rep_score": 0.03,
  "rep_flag": 0,
  "collapse_flag": 1,

  "intervened": 0,
  "intervention_type": "NONE",

  "text_preview": "..."
}
```

**Optional (recommended for audit): pre/post fields on intervention steps**
Note: `token_idx_ctx_pre` and `token_idx_ctx_post` are always present (post is null if no intervention).
- `token_idx_pre` / `token_idx_post` (compat alias)
- `pi_raw_post`, `pi_norm_post`, `rep_score_post`
- `collapse_flag_post`
- `text_preview_post`

✅ These are *extra* fields; aggregations must use the standard (pre) fields.

### F.3 Final record (required keys)

```json
{
  "record_type": "final",
  "timestamp_end": "2026-01-21T08:10:00+09:00",

  "collapse_run": 0,
  "collapse_reason": "NONE|PI|REP|TOO_SHORT|BUDGET_EXCEEDED",

  "n_tokens_total": 512,
  "n_interventions": 0,
  "H_pre": null,

  "runtime_sec": 123.45,
  "extra_tokens_due_to_intervention": 0
}
```

**H_pre definition (paper-fixed):**
- `H_pre = first token_idx_global where intervened=1`
- baseline: `H_pre = null`

---

## G) Directory Layout (No Mixing)

```
exp/
  runs/
    baseline/
      HUM/seed_000.jsonl
      ...
    controlled/
      HUM/seed_000.jsonl
      ...
  summary/
    runs_summary_all.csv
    table1.csv
    table1.md
```

Rules:
- Keep old logs in `archive/` (never mix into `exp/runs`)
- Summary tools must only read `seed_*.jsonl`

---

## H) Gates (Mandatory)

### H.1 Smoke gate (30 runs)
3 seeds × 5 pillars × 2 modes = **30 runs**

```
python validate_and_summarize.py --runs_root exp/runs --outdir exp/summary --expect 3
```

### H.2 Full gate (500 runs)
Each pillar×mode must be exactly 500 files

```
python validate_and_summarize.py --runs_root exp/runs --outdir exp/summary --expect 500
```

Hard checks:
- NaN = 0
- `(pillar, seed, mode)` unique
- `token_idx_global` monotonic increasing within each run
- baseline must have `n_interventions = 0`

---

## K) Table 1 Auto-Generation (Required)

**Purpose:** Prevent hand-edits and lock in “controlled vs baseline” quantification.

Inputs:
- `exp/summary/runs_summary_all.csv` (generated by `validate_and_summarize.py`)

Outputs:
- `exp/summary/table1.csv`
- `exp/summary/table1.md` (paper-ready)
- (optional) `exp/summary/table1.tex`

Command (example):
```
python make_table1.py   --summary_csv exp/summary/runs_summary_all.csv   --out_csv exp/summary/table1.csv   --out_md exp/summary/table1.md   --bootstrap 20000   --seed 0
```

Table 1 columns (per pillar):
- collapse_rate_baseline / collapse_rate_controlled
- delta_collapse + 95%CI (bootstrap, seed-resampling)
- RR + 95%CI
- OR + 95%CI (with zero-cell correction)
- intervene_rate_primary (main paper)
- intervene_density_secondary (appendix)
- runtime_sec_baseline / runtime_sec_controlled / delta_runtime_sec
- extra_tokens_due_to_intervention_mean

**Policy:** Table 1 is **always regenerated from logs** (no manual edits).

---

## I) Recommended Fixed Runtime Settings (Paper Default)

- `model_id = gpt2`
- `device = cpu`
- `max_new_tokens = 512`
- `temperature = 1.2`
- `top_p = 0.95`

---

## J) Runner (Single CLI for All Pillars)

Recommended runner:
- `run_vsave_with_seed_v4.py` (must implement this README spec)

Baseline/controlled execution example:
```
for s in $(seq 0 499); do
  python run_vsave_with_seed_v4.py     --exp_id vSAVE-1.1.1     --seed $s     --mode baseline     --pillar HUM     --prompts prompts/pillar_HUM_20.txt     --calib calib/calib.json     --out exp/runs/baseline/HUM/seed_$(printf "%03d" $s).jsonl     --model_id gpt2     --device cpu     --max_new_tokens 512     --temperature 1.2     --top_p 0.95
done
```

Controlled is identical except `--mode controlled`.

---

## Appendix: Pillar Intent (for reviewer clarity)

- HUM: argumentative / philosophical critique tasks (semantic drift + repetition risk)
- STEM: multi-step quantitative reasoning (error accumulation risk)
- ETH: value conflicts and normative judgments (framing sensitivity)
- TEMP: time-budget and “stop early” constraints (early-termination risk)
- META: self-critique and strategy reflection (self-referential instability risk)
