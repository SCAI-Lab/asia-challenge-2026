#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/00_env.sh"

# Install into the primary (TabPFN) venv.
source "${ASIA2026_VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install catboost

python - <<'PY'
import catboost
print("catboost version:", catboost.__version__)
PY
