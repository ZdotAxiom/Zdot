# Ż Collapse Lab — 保存版 vSAVE-1.1（査読クリア仕様 / Fixed Spec）

目的：
- Controlled（介入あり）: 5 pillars × 500 seeds = 2,500 runs
- Baseline（介入なし）: 5 pillars × 500 seeds = 2,500 runs
- **mode以外は完全同一**（比較不能を構造的に排除）
- 査読の致命傷（baseline不足 / methods曖昧 / 恣意性 / コスト / 再現性）を **仕様固定**で封殺

> ✅ このREADMEは「仕様」です。  
> “真実”は **生ログ（JSONL）→ summary（再生成）→ table（自動生成）** のパイプラインで担保され、**手編集は禁止**です。

---

## 1. 絶対ルール（ズレ防止）

**R1: 1 run = 1ファイル（追記禁止）**
- 1 run = 1 JSONL（追記禁止 / 追記は混入の温床）

**R2: Controlled/Baseline は “mode” だけ違う**
- 同一：prompt選択、decoder、seed、pillar、π計測、collapse判定、ログスキーマ、集計器
- 相違：介入実行の可否（controlledのみ許可）

**R3: “真実”は生ログ → runs_summary（再生成可能）**
- 中間CSV（seed_rates等）を真実扱いしない
- 必ず「生ログ→summary」を再生成できる状態にする

**R4: NaN禁止（欠損＝仕様破綻）**
- 値が無い場合は `null`（JSON） / `"NONE"`（カテゴリ）で表す（NaNは出さない）

---

## 2. 実験軸と固定設定（完全決定的）

- pillars: `HUM, STEM, ETH, TEMP, META`
- seeds: `0..499`
- modes: `baseline, controlled`

---

## 3. Decoder（固定：論文と一致させる）

推奨固定（全run共通）：

- model_id: `gpt2`
- device: `cpu`
- max_new_tokens: `512`
- temperature: `1.2`
- top_p: `0.95`
- stop条件も mode 間で同一

> decoderは **metaに必ず記録**すること（READMEとrunが一致していることが重要）

---

## 4. Prompt選択（seed→promptが決定的）

- prompts: `prompts/pillar_XXX_20.txt`（空行区切りブロック）
- prompt_id = `seed % N`（同seedは両modeで同一prompt）
- baseline/controlledで prompt が違うのは失格

---

## 5. ログスキーマ（全pillar・全mode共通）

推奨：JSONL（1行=1レコード）  
**順序は必ず** `meta → step* → final`

### 5.1 meta（先頭行）
- record_type = `"meta"`
- exp_id, mode, pillar, seed
- model_id, decoder(max_new_tokens/temp/top_p), prompt_id
- q50, q95, epsilon_pi, k_consecutive
- timestamp_start

### 5.2 step（各ステップ行）
- record_type = `"step"`
- step_idx, token_idx（単調増加, int）
- pi_raw（float）, pi_norm（0..1）
- pi_flag（0/1）
- rep_score（0..1）, rep_flag（0/1）
- too_short_flag（0/1）
- collapse_flag（0/1）
- intervened（0/1）
- intervention_type（介入なしは `"NONE"`）
- text_preview（短い）

### 5.3 final（末尾行）
- record_type = `"final"`
- collapse_run（0/1）, collapse_reason（string）
- H_pre（初回介入 token_idx。介入なしは null）
- n_tokens_total, n_interventions
- runtime_sec
- extra_tokens_due_to_intervention（baselineは0）
- timestamp_end

---

## 6. 崩壊判定（固定：二段構え）

### 6.1 step-level flags（推奨固定）
- epsilon_pi = `0.85`
- k_consecutive = `3`
- min_tokens = `64`
- rep_ngram = `3`
- rep_threshold = `0.20`

collapse_flag は、以下の統一定義で算出：

- pi_flag = 1[pi_norm > epsilon_pi]
- rep_flag = 1[rep_score > rep_threshold]
- too_short_flag = 1[token_idx < min_tokens]
- collapse_flag = 1[(pi_flag OR rep_flag OR too_short_flag)]

> “gibberish”を主観にしない。検出器＋閾値で固定する。

### 6.2 run-level collapse_run（一次定義）
collapse_run = 1 iff：  
- collapse_flag が **k_consecutive連続**で成立

---

## 7. π（実装近似）— 1ページ仕様（査読対策）

### 7.1 メトリクス（例）
- m1 = gzip_ratio(x)
- m2 = token_entropy(x)
- m3 = nll_proxy(x)  ※評価LMを固定（例：distilgpt2）

### 7.2 キャリブレーション（固定コーパス C）
- `calib/calib.json` に complexity の q50/q95 を保持
- q50/q95 は **runログ由来ではなく独立校正コーパス（例 N≈300）** から推定（リーク封じ）

### 7.3 正規化（大きいほど危険へ統一）
s_m(x) = clip( (m(x) - q50_m) / (q95_m - q50_m), 0, 1 )

### 7.4 合成（重み固定）
pi_norm(x) = Σ_m w_m s_m(x),  Σ w_m = 1

（任意）不一致：sigma_pi(x) = std_m(s_m(x))

---

## 8. 介入辞書 O(x) と選択規則（恣意性を潰す）

### 8.1 O(x)（有限・固定）
- SPLIT
- SUMMARIZE_STATE
- CONSTRAINT_INSERT
- REWRITE_CLEAN
- RESET_PROMPT（最後の手段）

### 8.2 選択規則（決定的）
- if pi_norm > epsilon_pi が k 連続:
  - if repetition 高い → REWRITE_CLEAN
  - elif format break → CONSTRAINT_INSERT
  - elif drift → SPLIT
  - else → SUMMARIZE_STATE
- if 介入が m_fail 回連続で改善しない → RESET_PROMPT

---

## 9. Baseline（介入なし）固定仕様

- mode=baseline：介入ロジックを呼ばない
- ただし π計測・collapse判定は同一
- intervened=0, intervention_type="NONE", n_interventions=0
- extra_tokens_due_to_intervention=0
- H_pre=null

---

## 10. 指標定義（論文と一致させる）

### 10.1 collapse_rate
- collapse_rate = mean_seed collapse_run

### 10.2 H_pre
- controlled：最初に `intervened=1` になった step の token_idx
- baseline：H_pre=null（N/A）

### 10.3 intervene_rate（Primary/Secondaryを明記）
**Primary（本文固定）: binary**
- intervene_rate_primary = mean_seed 1[n_interventions > 0]

**Secondary（Appendix固定）: density**
- intervene_density_secondary = mean_seed (n_interventions / max(1, n_tokens_total))

---

## 11. Cost（必須ログ）

runごとに必ず記録：
- runtime_sec
- extra_tokens_due_to_intervention
- n_interventions

集計で：pillar別 mean±CI、baselineとの差分

---

## 12. TEMP/META のドメイン例（Appendixに5〜10本は必須）

### TEMP（テンプレ）
1) 丁寧語変換  
2) 200字要約  
3) 中学生向け言い換え  
4) 制約付き生成  
5) JSON出力  

### META（テンプレ）
1) 仮説分解  
2) 落とし穴列挙  
3) 前提/結論分離  
4) 反例提示  
5) 不確実性明示  

---

## 13. ディレクトリ（混入防止）

推奨：

```
exp/
  runs/
    baseline/HUM/seed_000.jsonl
    controlled/HUM/seed_000.jsonl
  summary/
  logs/
archive/
```

> 旧ログは `archive/` へ退避（混入防止）

---

## 14. 自動ゲート（必須）

### Gate A（スモーク）
3 seeds × 5 pillars × 2 modes = **30 runs**

チェック内容：
- スキーマ一致
- NaNゼロ
- token_idx単調増加
- (mode,pillar,seed)ユニーク
- baselineは n_interventions=0, extra_tokens=0, H_pre=null を保証

### Gate B（本番）
expect = **500**

チェック内容：
- 各(mode,pillar)ちょうど500
- runs_summaryが生ログから再生成できる

---

## 15. Table 1 自動生成（必須：差分 + 95%CI + RR/OR）

目的：手計算・恣意性・更新ズレを排除し、**比較を1枚で固定**する。

### 15.1 入力と出力
- 入力：`exp/summary/runs_summary_all.csv`（`validate_and_summarize.py` が生成）
- 出力：
  - `exp/summary/table1.csv`
  - `exp/summary/table1.md`（論文貼り付け用）

### 15.2 実行コマンド（例）

1) summary生成

```bash
python validate_and_summarize.py --runs_root exp/runs --outdir exp/summary --expect 500
```

2) Table 1 生成（bootstrap）

```bash
python make_table1.py \
  --summary_csv exp/summary/runs_summary_all.csv \
  --out_csv exp/summary/table1.csv \
  --out_md exp/summary/table1.md \
  --bootstrap 20000 \
  --seed 0
```

### 15.3 Table 1 に含める列（pillarごと）
- collapse_rate_baseline / collapse_rate_controlled
- delta_collapse（controlled-baseline）+ 95%CI（seed resampling）
- RR_collapse + 95%CI
- OR_collapse + 95%CI（ゼロ割回避補正込み）
- intervene_rate_primary（本文固定）
- intervene_density_secondary（Appendix固定）
- runtime_sec_baseline / runtime_sec_controlled / delta_runtime_sec
- extra_tokens_due_to_intervention_mean（controlled、baselineは0）

---

## 16. 使い方（最短手順）

### 16.1 Smoke（30-run）
1. 30-run smoke を回す  
2. `validate_and_summarize.py` で Gate A を通す（expect=3）

```bash
python validate_and_summarize.py --runs_root exp/runs --outdir exp/summary --expect 3

### 16.2 本番（500-run）
1. 500-run 本番を回す
2. summary生成（expect=500）
3. Table 1 を自動生成（`make_table1.py`）

✅ 最重要：**生ログ→summary→table の再生成性が再現性そのもの。**

**Repro commit:** tag `vsave-1.1`
