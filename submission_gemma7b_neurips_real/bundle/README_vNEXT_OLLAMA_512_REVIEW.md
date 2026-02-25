# vNEXT-OLLAMA (gemma:7b, 512 fixed) — Reviewer-Ready Summary

This README is a reviewer-facing English summary of the vNEXT Ollama runs with **gemma:7b** and **max_new_tokens=512** fixed. It records the protocol, the provenance of results, and the final Table 1 numbers. The intent is to make the experimental setup and result lineage auditable without requiring cross-references.

---

## 1) Provenance (where the numbers come from)
Primary sources for the results in this file (submission-truth):
- Table 1 (final): `exp_vnext_ollama_gemma7b_512/summary_5000_sensitive_real/table1_gemma7b_512_sensitive.md`
- Table 1 (CSV): `exp_vnext_ollama_gemma7b_512/summary_5000_sensitive_real/table1_gemma7b_512_sensitive.csv`
- Run-level summary (5 pillars, 5,000 rows): `exp_vnext_ollama_gemma7b_512/summary_5000_sensitive_real/runs_summary_all.csv`
- Counts check: `exp_vnext_ollama_gemma7b_512/summary_5000_sensitive_real/counts_by_condition.csv`

All summary tables are generated from raw JSONL logs using `validate_and_summarize_vnext_ollama_gemma7b.py` and `make_table1_vnext_ollama_gemma7b.py`. Result values are not manually edited.

---

## 2) Experimental setup (fixed)
- Model: `gemma:7b` via Ollama
- max_new_tokens: 512 (fixed)
- Decoding: temperature = 1.2, top_p = 0.95
- Collapse detection: epsilon_pi = 0.85, k_consecutive = 3, min_tokens = 16
- Repetition: rep_ngram = 3, rep_threshold = 0.70 (ratio-based)
- Modes:
  - baseline: max_interventions = 0
  - controlled: max_interventions = 100
- Intervention: truncate context to keep_ratio = 0.70, then regenerate the current segment under fixed decoding parameters
- These values are set via CLI flags and recorded in JSONL meta for each run.
- Seeds: 0..499 for each pillar (paired baseline vs controlled)
- Pillars: HUM / STEM / ETH / TEMP / META
- Calibration: sensitive (gemma:7b)
  - `calib/calib_ollama_gemma7b_v2_sensitive.json`
  - "Sensitive" indicates the calibration corpus is not included in the public artifact; hashes or summary statistics are provided instead. For verification, we publish `exp_vnext_ollama_gemma7b_512/summary_5000_sensitive_real/runs_summary_all.csv` and `exp_vnext_ollama_gemma7b_512/summary_5000_sensitive_real/table1_gemma7b_512_sensitive.csv`.

Raw runs are recorded as JSONL with schema `meta → step* → final`. Each JSONL ends with a final record; incomplete files are treated as invalid by the validator.

---

## 3) Table 1 caption (paste-ready)
Table 1: Collapse reduction across five pillars under baseline vs controlled decoding (Gemma 7B, Ollama, 512 tokens fixed).
For each pillar, we report collapse rates with Wilson 95% confidence intervals, the absolute difference Δ = (controlled − baseline), and relative effect sizes (RR and OR). Confidence intervals for Δ/RR/OR are computed via paired bootstrap over seeds (B = 20,000), using identical seeds for baseline and controlled runs. RR/OR apply the Haldane–Anscombe correction (+0.5) when any cell count is zero. intervene_rate is the mean 1[n_interventions > 0] over controlled runs. Runtime values are mean seconds per run; Δruntime = (controlled − baseline).

---

## 4) Results summary (from `exp_vnext_ollama_gemma7b_512/summary_5000_sensitive_real/table1_gemma7b_512_sensitive.md`)
Across all five pillars, controlled decoding reduces collapse compared to baseline, with small runtime overhead. Exact values below match the final table.

| Pillar | n | collapse_base | collapse_ctrl | Δ (ctrl-base) | Δ 95%CI | RR | RR 95%CI | OR | OR 95%CI | intervene_rate | runtime_base | runtime_ctrl | Δruntime | paired_boot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HUM | 500 | 0.142 (71/500) [0.114,0.175] | 0.000 (0/500) [0.000,0.008] | -0.142 | [-0.1720,-0.1120] | 0.007 | [0.0058,0.0088] | 0.006 | [0.0048,0.0079] | 0.142 | 19.6 | 20.9 | 1.2 | Y |
| STEM | 500 | 0.302 (151/500) [0.263,0.344] | 0.004 (2/500) [0.001,0.014] | -0.298 | [-0.3380,-0.2580] | 0.016 | [0.003,0.038] | 0.011 | [0.002,0.027] | 0.302 | 19.1 | 20.7 | 1.6 | Y |
| ETH | 500 | 0.010 (5/500) [0.004,0.023] | 0.000 (0/500) [0.000,0.008] | -0.010 | [-0.0200,-0.0020] | 0.091 | [0.0476,0.3333] | 0.090 | [0.0467,0.3327] | 0.010 | 20.4 | 20.3 | -0.1 | Y |
| TEMP | 500 | 0.030 (15/500) [0.018,0.049] | 0.000 (0/500) [0.000,0.008] | -0.030 | [-0.0460,-0.0160] | 0.032 | [0.0213,0.0588] | 0.031 | [0.0203,0.0579] | 0.030 | 19.9 | 20.0 | 0.1 | Y |
| META | 500 | 0.106 (53/500) [0.082,0.136] | 0.006 (3/500) [0.002,0.017] | -0.100 | [-0.1260,-0.0740] | 0.063 | [0.010,0.137] | 0.056 | [0.009,0.125] | 0.106 | 19.8 | 20.5 | 0.7 | Y |

---

## 5) Reproducibility (commands)
Full 512-token protocol and runnable commands are in `README_vNEXT_OLLAMA_512.md` (see the Protocol / Commands sections). This file focuses on reviewer-facing summaries and validated results.
Scripts used for submission outputs:
- `scripts_gemma7b_ollama/validate_and_summarize_vnext_ollama_gemma7b.py`
- `scripts_gemma7b_ollama/make_table1_vnext_ollama_gemma7b.py`

---

## 6) Notes for reviewers
- Baseline and controlled runs use identical seeds and prompt selection, enabling paired bootstrap analysis.
- All results are derived from raw JSONL logs; no manual edits or filtering.
- When a value in this README appears numerical, its source is explicitly listed in Section 1 to allow direct verification.
- Prompt selection is deterministic given the seed (prompt_id recorded in the summary CSV).
- collapse_reason is reported as produced by the validator (e.g., REP / TOO_SHORT), without post-hoc relabeling.
