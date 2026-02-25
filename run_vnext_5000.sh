#!/usr/bin/env bash
set -euo pipefail

# ====== config ======
EXP_ID="vNEXT-FULL"
RUNS_ROOT="exp_vnext_full/runs"
SUMMARY_DIR="exp_vnext_full/summary"

PILLARS=("HUM" "STEM" "ETH" "TEMP" "META")
MODES=("baseline" "controlled")

SEED_START=0
SEED_END=499  # inclusive

PROMPTS_DIR="prompts_vnext"
CALIB="calib/calib.json"

MODEL_ID="${MODEL_ID:-gpt2}"
DEVICE="${DEVICE:-cpu}"          # cpu / mps など（あなたの実装に合わせる）
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TEMPERATURE="${TEMPERATURE:-1.2}"
TOP_P="${TOP_P:-0.95}"

EPSILON_PI="${EPSILON_PI:-0.85}"
K_CONSEC="${K_CONSEC:-3}"
MIN_TOKENS="${MIN_TOKENS:-64}"
REP_NGRAM="${REP_NGRAM:-3}"
REP_THRESHOLD="${REP_THRESHOLD:-0.20}"
MAX_INTERVENTIONS="${MAX_INTERVENTIONS:-100}"

# 熱対策：各run後に休憩（秒）
SLEEP_SEC="${SLEEP_SEC:-0}"      # 例: SLEEP_SEC=15 ./run_vnext_5000.sh

# 途中停止しても壊れないようにログ
LOG_DIR="exp_vnext_full/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] logging to: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

# ====== precheck ======
if [[ ! -f "$CALIB" ]]; then
  echo "[FATAL] calib not found: $CALIB"
  exit 1
fi

# ====== run ======
echo "== [vNEXT 5000-run] start =="
echo "[INFO] EXP_ID=$EXP_ID RUNS_ROOT=$RUNS_ROOT SUMMARY_DIR=$SUMMARY_DIR"
echo "[INFO] MODEL_ID=$MODEL_ID DEVICE=$DEVICE seeds=${SEED_START}..${SEED_END}"

for pillar in "${PILLARS[@]}"; do
  prompts="${PROMPTS_DIR}/pillar_${pillar}_40.txt"
  if [[ ! -f "$prompts" ]]; then
    echo "[FATAL] prompts not found: $prompts"
    exit 1
  fi

  for mode in "${MODES[@]}"; do
    for ((s=SEED_START; s<=SEED_END; s++)); do
      out="${RUNS_ROOT}/${mode}/${pillar}/seed_$(printf "%03d" "$s").jsonl"
      mkdir -p "$(dirname "$out")"

      # 既にあればスキップ（途中再開用）
      if [[ -s "$out" ]]; then
        echo "[SKIP] exists: $out"
        continue
      fi

      echo
      echo "---- RUN pillar=$pillar mode=$mode seed=$s ----"
      python run_vnext_with_seed_v5.py \
        --exp_id "$EXP_ID" \
        --seed "$s" \
        --mode "$mode" \
        --pillar "$pillar" \
        --prompts "$prompts" \
        --calib "$CALIB" \
        --out "$out" \
        --model_id "$MODEL_ID" \
        --device "$DEVICE" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top_p "$TOP_P" \
        --epsilon_pi "$EPSILON_PI" \
        --k_consecutive "$K_CONSEC" \
        --min_tokens "$MIN_TOKENS" \
        --rep_ngram "$REP_NGRAM" \
        --rep_threshold "$REP_THRESHOLD" \
        --max_interventions "$MAX_INTERVENTIONS"

      if [[ "$SLEEP_SEC" -gt 0 ]]; then
        sleep "$SLEEP_SEC"
      fi
    done
  done
done

echo
echo "== [vNEXT 5000-run] finished runs =="
echo "== summary =="
python validate_and_summarize_vnext.py \
  --runs_root "$RUNS_ROOT" \
  --outdir "$SUMMARY_DIR" \
  --expect 500

echo "== [DONE] summary at: $SUMMARY_DIR =="
