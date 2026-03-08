#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

VENV_DIR="${REPO_ROOT}/venv_catboost_residual"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -d "${VENV_DIR}" ]]; then
  echo "[20_setup_catboost_residual_env] venv already exists: ${VENV_DIR}"
else
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  echo "[20_setup_catboost_residual_env] created venv: ${VENV_DIR}"
fi

# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install catboost numpy pandas scikit-learn

python - <<'PY'
import catboost
print("catboost version:", catboost.__version__)
PY

echo "[20_setup_catboost_residual_env] done. Export CATBOOST_VENV_DIR=${VENV_DIR} when submitting Slurm jobs."
