# vSAVE-PILLAR-PERMA（全柱共通テンプレ / PERMA）
Ż Collapse Lab — Reproducible Pillar Template  
目的：baseline不足／methods曖昧／恣意性／コスト／再現性 を “仕様” で封殺する

このテンプレは HUM で確立した仕様を抽象化したものであり、
STEM / ETH / TEMP / META にそのまま横展開できる。

---

## 0. ゴール（このテンプレが保証するもの）
各 pillar について：

- baseline：seed 0..499（500 runs）
- controlled：seed 0..499（500 runs）
- mode以外の条件を完全一致（prompt選択・decoder・π校正・判定器・ログ）
- 生ログ（JSONL）から summary（CSV）を **100%再生成可能**
- 旧ログ混在ゼロ（対象pillarのみがsummaryに存在）

---

## 1. 実験定義（Methods固定点）
### 1.1 実験単位（全柱共通）
- pillar ∈ {HUM, STEM, ETH, TEMP, META}
- seed ∈ {0..499}
- mode ∈ {baseline, controlled}

### 1.2 比較可能性の固定（最重要）
Controlled/Baseline の差は mode（介入ON/OFF）だけに固定する：

- prompt選択ルール：同一
- decoder：同一
- π校正（q50/q95）：同一
- collapse判定：同一
- ログスキーマ：同一

---

## 2. Decoder固定（mode間で同一）
以下を baseline/controlled で完全一致に固定：

- max_new_tokens
- temperature
- top_p
- 停止条件（EOS / 長さ上限 / truncation）

---

## 3. Prompt選択（決定性）
- prompts は「空行区切りブロック（block pool）」で構成
- prompt_id = seed % N_blocks
- baseline/controlled で同一prompt_idを使用

これにより「別promptを引いた」を排除し、比較可能性を保証する。

---

## 4. π（pi）定義（リーク封じ：独立校正）
### 4.1 原則（NeurIPS強度）
**q50/q95は seedログ由来ではなく独立校正データ（calibration corpus）から推定する。**

### 4.2 1ページ式（固定）
- pi_raw = metric(text)  ※例：hybrid_complexity
- pi_norm = clamp01((pi_raw - q50) / (q95 - q50))
- pi_flag = 1[pi_norm > epsilon_pi]

固定：
- epsilon_pi = 0.85
- k_consecutive = 3

---

## 5. 崩壊判定（run-levelが“真実”）
論文本体の崩壊率は run-level の collapse_run を使用する。

- collapse_run ∈ {0,1}
- collapse_rate(pillar, mode) = mean over seeds of collapse_run

step-levelの collapse_flag は補助（デバッグ/可視化）であり、
本文の真実は collapse_run に統一する。

---

## 6. collapse_flag の最終仕様（mode間ズレ封殺）
step-level collapse_flag の mode間ズレは比較不能の致命傷になる。

最終仕様として、両modeに以下を適用する：

**collapse_flag = 1 if (pi_flag OR rep_flag) else 0**

運用：
- 欠損がある場合は collapse_flag 列を追加
- その後、上記定義で collapse_flag を再計算して上書き

---

## 7. 介入（controlledのみ）
### 7.1 run-level記録（必須）
- intervened：stepで介入したか（0/1）
- n_interventions：run内介入回数
- H_pre：初回介入 step_idx（介入無しは null）

### 7.2 intervene_rate（論文固定）
本文 primary（固定）：
- intervene_rate_primary = mean 1[n_interventions > 0]

Appendix secondary：
- mean(n_interventions / n_tokens_total)

「どっちでもいい」を論文から排除し、査読で刺されないように固定する。

---

## 8. ログスキーマ（JSONL / 1 run = 1 file）
### 8.1 絶対ルール
- 1 run = 1 jsonl file
- 追記禁止（混在事故の根絶）

出力例：
- exp/runs/baseline/{PILLAR}/seed_000.jsonl
- exp/runs/controlled/{PILLAR}/seed_000.jsonl

### 8.2 record_type
(A) meta（先頭）
- exp_id, mode, pillar, seed, model_id
- decoder（temperature, top_p, max_new_tokens）
- π定義（q50,q95,epsilon_pi,k_consecutive）

(B) step（中身）
- step_idx, token_idx
- pi_raw, pi_norm, pi_flag
- rep_score, rep_flag
- collapse_flag（パッチ適用後は mode間一致）
- intervened, intervention_type
- text_preview

(C) final（末尾）
- H_pre
- n_tokens_total
- n_interventions
- collapse_run
- collapse_reason（PI/REP/SHORT/NONE 等）
- runtime_sec
- extra_tokens_due_to_intervention

---

## 9. 品質ゲート（柱完成の条件）
### 9.1 本番ゲート（500完了後）
python validate_and_summarize.py --runs_root exp/runs --outdir exp/summary --expect 500

成功条件：
- runs_summary_all.csv rows = 1000
- breakdown：
  - baseline {PILLAR} 500
  - controlled {PILLAR} 500

### 9.2 混在ゼロ確認（必須）
- 対象pillar以外が summary に存在しないことを groupby で確認
- 対象pillar以外の seedファイルが存在しないことを確認

---

## 10. バックアップ（事故防止：自動検出）
tar に校正ファイルが入らず落ちる事故を防ぐため、
calib / corpus は find で自動検出して含める。

推奨：
- find . -name 'calib*.json'
- find . -name 'calibration_corpus*.txt'

---

## 11. 状態宣言（PERMA固定点）
このテンプレは以下を満たすことで “NeurIPS査読に耐える比較実験” の最小要件を満たす：

1) mode以外同一（prompt/decoder/π校正/判定/ログ）
2) 独立校正由来のq50/q95
3) JSONLスキーマ統一・欠損ゼロ
4) collapse_flagのmode間ズレをパッチで解消
5) summary再生成ゲート通過（rows=1000）
6) 他pillar混在ゼロ

