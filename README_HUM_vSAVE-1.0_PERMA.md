# HUM 完全保存版（vSAVE-HUM-1.0 / PERMA）
Ż Collapse Lab — HUM pillar  
目的：baseline不足／methods曖昧／恣意性／コスト／再現性 を “仕様” で封殺する

---

## 0. ゴール（この保存版が保証するもの）
- HUM baseline：seed 0..499（500 runs）
- HUM controlled：seed 0..499（500 runs）
- mode以外の条件を完全一致（prompt選択・decoder・π校正・判定器・ログ）
- 生ログ（JSONL）から summary（CSV）を **100%再生成可能**
- 旧ログ混在ゼロ（HUMのみが summary に存在）

---

## 1. 実験定義（Methods固定点）
### 1.1 実験単位
- pillar = HUM
- seed ∈ {0..499}
- mode ∈ {baseline, controlled}

### 1.2 Decoder固定（mode間で同一）
- max_new_tokens（例：128）
- temperature（例：1.0）
- top_p（例：0.9）
- 停止条件（EOS / 長さ上限 / truncation）も一致

### 1.3 Prompt選択（決定性）
- promptsは「空行区切りブロック」
- prompt_id = seed % n_blocks
- baseline/controlledで同一prompt_idを使用  
→ mode以外同一比較の根拠

---

## 2. π（pi）定義（リーク封じ：最重要）
### 2.1 原則
**q50/q95はseedログ由来ではなく独立校正データ（calibration corpus）から推定**

### 2.2 1ページ式（固定）
- pi_raw = hybrid_complexity(text)
- pi_norm = clamp01((pi_raw - q50) / (q95 - q50))
- pi_flag = 1[pi_norm > epsilon_pi]

固定：
- epsilon_pi = 0.85
- k_consecutive = 3（run-level collapse判定に使用）

---

## 3. 崩壊判定（run-levelが“真実”）
論文本体の崩壊率は collapse_run の平均（binary）。

- collapse_run ∈ {0,1}
- collapse_reason ∈ {PI, REP, SHORT, NONE}

step-levelは補助（デバッグ・可視化用）。

---

## 4. 介入（controlledのみ）
### 4.1 記録
- intervened：そのstepで介入したか（0/1）
- n_interventions：run内介入回数
- H_pre：初回介入step_idx（介入無しはnull）

### 4.2 intervene_rate（本文/付録で固定）
本文 primary（固定）：
- intervene_rate_primary = mean 1[n_interventions > 0]

Appendix secondary：
- mean(n_interventions / n_tokens_total)

---

## 5. ログスキーマ（JSONL）
### 5.1 1 run = 1 file
- 追記禁止
- 例：
  - exp/runs/baseline/HUM/seed_000.jsonl
  - exp/runs/controlled/HUM/seed_000.jsonl

### 5.2 record_type
(A) meta（先頭）
- exp_id, mode, pillar, seed, model_id
- decoder（temperature, top_p, max_new_tokens）
- π定義（q50,q95,epsilon_pi,k_consecutive）

(B) step（中身）
- step_idx, token_idx
- pi_raw, pi_norm, pi_flag
- rep_score, rep_flag
- collapse_flag（※パッチでmode一致保証）
- intervened, intervention_type
- text_preview

(C) final（末尾）
- H_pre
- n_tokens_total
- n_interventions
- collapse_run
- collapse_by_pi / collapse_by_rep / collapse_by_short
- collapse_reason
- runtime_sec
- extra_tokens_due_to_intervention

---

## 6. collapse_flag の最終仕様（比較不能バグを封殺）
### 6.1 過去の問題
mode間で rep_flag が立っているのに collapse_flag がズレるケースがあり、
「同条件比較が崩れる」致命点だった。

### 6.2 最終定義（mode間で必ず一致）
**collapse_flag = 1 if (pi_flag OR rep_flag) else 0**

この定義を両modeに適用し、step-levelのズレを完全排除する。

---

## 7. 実行・検証（ゲート）
### 7.1 本番ゲート（500完了後）
python validate_and_summarize.py --runs_root exp/runs --outdir exp/summary --expect 500

成功条件：
- exp/summary/runs_summary_all.csv rows = 1000
- breakdown:
  - baseline HUM 500
  - controlled HUM 500

### 7.2 STEM混在ゼロ確認
- exp/runs/*/STEM/seed_*.jsonl が 0本
- summary の groupby が HUMのみ

---

## 8. seed_117/118 “TOO_SHORT疑惑” について（結論：問題なし）
短文でも collapse_by_short=0 かつ collapse_reason=NONE を確認済み。  
さらに min_tokens がmetaに無い場合でも矛盾検査はbad_count=0。

---

## 9. 構成要素（このHUMを再現する最低セット）
- run_vsave_with_seed.py
- z_collapse_lab_SES_v2_4_2_no_text_fix.py
- validate_and_summarize.py
- patch_add_collapse_flag_both_modes.py
- patch_recompute_collapse_flag.py
- （校正）calib/calib.json（独立校正：q50/q95）
- （校正コーパス）calib/calibration_corpus.txt（独立校正データ）（または同等の独立校正テキスト）

---

## 10. 状態宣言（PERMA固定点）
HUMは以下を満たすことで “NeurIPS査読に耐える比較実験” の最小要件を満たした：
1) mode以外同一（prompt/decoder/判定器/ログ）
2) 独立校正由来のq50/q95
3) JSONLスキーマ統一・欠損ゼロ
4) collapse_flagのmode間ズレをパッチで解消
5) summary再生成ゲート通過
6) 旧ログ混在ゼロ（HUMのみ）

