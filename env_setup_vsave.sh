#!/usr/bin/env bash
set -euo pipefail

echo "== [vSAVE env] start =="

# ---- 0) sanity: project root ----
if [ ! -f "requirements.lock.txt" ]; then
  echo "[ERROR] requirements.lock.txt not found in current dir."
  exit 1
fi

# ---- 1) python version (pyenv) ----
if [ -f ".python-version" ]; then
  echo "[INFO] .python-version exists:"
  cat .python-version
else
  echo "[WARN] .python-version not found. (recommended) create it manually."
fi

# ---- 2) venv recreate (clean) ----
if [ -d ".venv" ]; then
  echo "[INFO] remove existing .venv"
  rm -rf .venv
fi

PYBIN="${PYBIN:-python3}"

echo "[INFO] create venv with: $PYBIN"
$PYBIN -m venv .venv

echo "[INFO] activate venv"
source .venv/bin/activate

echo "[INFO] upgrade pip"
python -m pip install --upgrade pip

# ---- 3) install from lock (the source of truth) ----
echo "[INFO] install requirements.lock.txt"
pip install -r requirements.lock.txt

# ---- 4) save environment snapshot ----
echo "[INFO] save requirements.now.txt"
pip freeze | sort > requirements.now.txt

echo "[INFO] sorted lock snapshot -> requirements.lock.sorted.txt"
sort requirements.lock.txt > requirements.lock.sorted.txt

echo "== [DIFF lock vs now] (empty = perfect) =="
diff -u requirements.lock.sorted.txt requirements.now.txt || true

# ---- 5) record python + pip info ----
echo "[INFO] write env_report.txt"
{
  echo "=== python ==="
  python -V
  which python
  echo
  echo "=== pip ==="
  pip -V
  echo
  echo "=== pip freeze (top 60) ==="
  head -60 requirements.now.txt
} > env_report.txt

echo "== [vSAVE env] done =="
echo "[OK] .venv ready, requirements.now.txt saved, env_report.txt saved."
