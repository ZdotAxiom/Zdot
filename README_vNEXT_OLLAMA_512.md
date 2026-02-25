# vNEXT-OLLAMA (gemma:7b) — Reproducible Collapse/Intervention Runs (512 tokens fixed)

このREADMEは **Ollama + gemma:7b** を使った vNEXT 実験（baseline / controlled）を、**max_new_tokens=512 固定**で再現・保存するための “完全保存用” 手順です。

- 実験は **JSONL（meta → step* → final）** に記録
- 集計は **正本スクリプト**で自動生成（下記参照）
- 校正（calibration）は **strict / sensitive** を並走可能
- 出力フォルダを分離し、vSAVE と混ざらないようにします

---

## 0) 前提（固定方針）
- **モデル**: `gemma:7b`（Ollama）
- **max_new_tokens**: **512 固定**（このREADMEでは 1024/2048 を使わない）
- **温度管理**: ローカル連続推論は発熱するため、柱ごと/seedごとに休止を推奨

---

## 1) 必要ファイル（このリポジトリにある想定）
- `run_vnext_ollama_v1.py` … Ollama 版ランナー（meta/step/final を書く）
- `validate_and_summarize_vnext.py` … runs_root を集計して summary を作る
- `prompts_vnext/`
  - `pillar_HUM_40.txt` など（柱プロンプト）
- `calib/`
  - `calib.json`（strict）
  - `calib_ollama_gemma7b_v2_sensitive.json`（sensitive）
- `calibration_corpus_v2_ollama_gemma7b_sensitive.txt`（sensitiveコーパス）

---

## 1.1) 正本スクリプト（gemma:7b 固定）
混乱防止のため、以下の3本を正本とします。
- `scripts_gemma7b_ollama/run_vnext_ollama_gemma7b_v1.py`
- `scripts_gemma7b_ollama/validate_and_summarize_vnext_ollama_gemma7b.py`
- `scripts_gemma7b_ollama/make_table1_vnext_ollama_gemma7b.py`

正本出力（提出用）は以下に固定します。
- `exp_vnext_ollama_gemma7b_512/summary_500/table1_gemma7b_512_sensitive.csv`
- `exp_vnext_ollama_gemma7b_512/summary_500/table1_gemma7b_512_sensitive.md`

---

## 2) セットアップ（最小）
### 2.1 Python venv
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
# requirements がある場合
# python -m pip install -r requirements.txt
```

### 2.2 Ollama
- Ollama をインストール済みであること
- モデル取得:
```bash
ollama pull gemma:7b
ollama list
```

---

## 3) 重要：出力フォルダ（混入防止）
このREADMEでは **exp_vnext_ollama_gemma7b_512/** を使います。

```bash
ROOT="exp_vnext_ollama_gemma7b_512"
mkdir -p "$ROOT/runs" "$ROOT/summary"
```

---

## 4) 実験パラメータ（512固定セット）
以下を **本READMEのデフォルト固定**とします。

- `--ollama_model gemma:7b`
- `--max_new_tokens 512` ✅固定
- `--max_segments 32`（512固定運用の暴走保険。安定確認後に 64 まで可）
- `--temperature 1.2`
- `--top_p 0.95`
- `--epsilon_pi 0.85`
- `--k_consecutive 3`
- `--min_tokens 16`
- `--rep_ngram 3`
- `--rep_threshold 0.70`
- `--keep_ratio 0.70`（介入時のコンテキスト保持率）
- `baseline`: `--max_interventions 0`
- `controlled`: `--max_interventions 100`（必要なら後で変更）

> NOTE: 本実験は上記の固定値を CLI で指定して実行し、その値は JSONL の meta に記録されます。

---

## 5) 校正（Calibration）
### 5.1 strict（保守的）
- `--calib calib/calib.json`

### 5.2 sensitive（gemma:7b向け）
- `--calib calib/calib_ollama_gemma7b_v2_sensitive.json`

**推奨**：同じ runs を strict と sensitive で別フォルダに分けて並走し、PIの発火差を比較する。

---

## 6) SMOKE（各柱×各mode×3seed = 30run）
### 6.1 strict smoke（512固定）
```bash
set -euo pipefail

ROOT="exp_vnext_ollama_gemma7b_512"
RUNS="$ROOT/runs_smoke_strict"
rm -rf "$RUNS"
mkdir -p "$RUNS"

for pillar in HUM STEM ETH TEMP META; do
  for mode in baseline controlled; do

    # baseline/controlled で介入上限を切り替える
    if [ "$mode" = "baseline" ]; then
      MI=0
    else
      MI=100
    fi

    for s in 0 1 2; do
      out="$RUNS/$mode/$pillar/seed_$(printf "%03d" $s).jsonl"
      mkdir -p "$(dirname "$out")"

      python run_vnext_ollama_v1.py \
        --exp_id vNEXT-OLLAMA-SMOKE-512-STRICT \
        --seed "$s" \
        --mode "$mode" \
        --pillar "$pillar" \
        --prompts "prompts_vnext/pillar_${pillar}_40.txt" \
        --calib "calib/calib.json" \
        --out "$out" \
        --ollama_model "gemma:7b" \
        --max_new_tokens 512 \
        --max_segments 32 \
        --temperature 1.2 \
        --top_p 0.95 \
        --epsilon_pi 0.85 \
        --k_consecutive 3 \
        --min_tokens 16 \
        --rep_ngram 3 \
        --rep_threshold 0.70 \
        --keep_ratio 0.70 \
        --max_interventions "$MI"
    done
  done
done

python validate_and_summarize_vnext.py \
  --runs_root "$RUNS" \
  --outdir "$ROOT/summary_smoke_strict" \
  --expect 3
```

### 6.2 sensitive smoke（512固定）
`--calib` だけ差し替えます。

```bash
set -euo pipefail

ROOT="exp_vnext_ollama_gemma7b_512"
RUNS="$ROOT/runs_smoke_sensitive"
rm -rf "$RUNS"
mkdir -p "$RUNS"

for pillar in HUM STEM ETH TEMP META; do
  for mode in baseline controlled; do

    if [ "$mode" = "baseline" ]; then
      MI=0
    else
      MI=100
    fi

    for s in 0 1 2; do
      out="$RUNS/$mode/$pillar/seed_$(printf "%03d" $s).jsonl"
      mkdir -p "$(dirname "$out")"

      python run_vnext_ollama_v1.py \
        --exp_id vNEXT-OLLAMA-SMOKE-512-SENSITIVE \
        --seed "$s" \
        --mode "$mode" \
        --pillar "$pillar" \
        --prompts "prompts_vnext/pillar_${pillar}_40.txt" \
        --calib "calib/calib_ollama_gemma7b_v2_sensitive.json" \
        --out "$out" \
        --ollama_model "gemma:7b" \
        --max_new_tokens 512 \
        --max_segments 32 \
        --temperature 1.2 \
        --top_p 0.95 \
        --epsilon_pi 0.85 \
        --k_consecutive 3 \
        --min_tokens 16 \
        --rep_ngram 3 \
        --rep_threshold 0.70 \
        --keep_ratio 0.70 \
        --max_interventions "$MI"
    done
  done
done

python validate_and_summarize_vnext.py \
  --runs_root "$RUNS" \
  --outdir "$ROOT/summary_smoke_sensitive" \
  --expect 3
```

---

## 7) 30-seed（各柱×各mode×30seed = 300run）
3seed の `for s in ...` を `seq 0 29` に変更すればOKです（MI切替＋`--max_segments 32` は必須）。

例（strict）：
```bash
set -euo pipefail
ROOT="exp_vnext_ollama_gemma7b_512"
RUNS="$ROOT/runs_30_strict"
rm -rf "$RUNS"
mkdir -p "$RUNS"

for pillar in HUM STEM ETH TEMP META; do
  for mode in baseline controlled; do

    if [ "$mode" = "baseline" ]; then
      MI=0
    else
      MI=100
    fi

    for s in $(seq 0 29); do
      out="$RUNS/$mode/$pillar/seed_$(printf "%03d" $s).jsonl"
      mkdir -p "$(dirname "$out")"

      python run_vnext_ollama_v1.py \
        --exp_id vNEXT-OLLAMA-30-512-STRICT \
        --seed "$s" \
        --mode "$mode" \
        --pillar "$pillar" \
        --prompts "prompts_vnext/pillar_${pillar}_40.txt" \
        --calib "calib/calib.json" \
        --out "$out" \
        --ollama_model "gemma:7b" \
        --max_new_tokens 512 \
        --max_segments 32 \
        --temperature 1.2 \
        --top_p 0.95 \
        --epsilon_pi 0.85 \
        --k_consecutive 3 \
        --min_tokens 16 \
        --rep_ngram 3 \
        --rep_threshold 0.70 \
        --keep_ratio 0.70 \
        --max_interventions "$MI"
    done
  done
done

python validate_and_summarize_vnext.py \
  --runs_root "$RUNS" \
  --outdir "$ROOT/summary_30_strict" \
  --expect 30
```

---

## 8) 保存・再現性（環境レポート出力）
完全保存用に、実験実行前/後どちらでも良いので環境をダンプします。

```bash
cat > env_capture_ollama7b_512.sh <<'SH'
set -euo pipefail
OUT="env_report_ollama_7b_512.txt"
{
  echo "=== DATE ==="; date
  echo; echo "=== SYSTEM ==="; sw_vers || true; uname -a || true
  echo; echo "=== CPU/MEM ==="; sysctl -n machdep.cpu.brand_string || true; sysctl -n hw.memsize || true
  echo; echo "=== PYTHON ==="; which python || true; python -V || true
  echo; echo "=== PIP FREEZE ==="; python -m pip -V || true; python -m pip freeze || true
  echo; echo "=== OLLAMA ==="
  (command -v ollama && ollama --version) || echo "ollama not found"
  (command -v ollama && ollama list) || true
  echo; echo "=== FILE HASH (key) ==="
  for f in run_vnext_ollama_v1.py validate_and_summarize_vnext.py; do
    [ -f "$f" ] && shasum -a 256 "$f" || echo "missing: $f"
  done
  echo; echo "=== CALIB HASH ==="
  for f in calib/calib.json calib/calib_ollama_gemma7b_v2_sensitive.json; do
    [ -f "$f" ] && shasum -a 256 "$f" || echo "missing: $f"
  done
  echo; echo "=== PROMPTS ==="
  ls -la prompts_vnext 2>/dev/null || true
} > "$OUT"
echo "[OK] wrote $OUT"
SH

bash env_capture_ollama7b_512.sh
```

---

## 9) よくある確認ポイント
### 9.1 PI が 0 張り付きに見える
- strict 校正では **pi_norm が発火しない**ことがある（分布レンジ問題）
- sensitive 校正で比較し、**PIの校正依存性**として論文化できる

### 9.2 REP が先に立って PI reason が見えない
- baseline で REP が先に崩壊すると PI が観測されにくい
- PI観測だけしたい場合は **rep_threshold を一時的に上げる**（例: 0.99）
  ※ただし本番条件から外れるので、観測実験として区別して記録する

### 9.3 発熱
- 512でも連続実行は熱が出る
- 柱ごとに休止、扇風機、ファン固定など運用ルールをログ化推奨

---

## 10) 研究用の“固定宣言”（本文/Appendix用）
- “All experiments in this section fix `max_new_tokens=512` for comparability and thermal stability on a local Mac setup.”
- “We log meta/step/final records in JSONL and generate all summary tables from raw logs without manual edits.”
