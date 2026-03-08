#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

SAITS_PYPOTS_VENV_DIR="${SAITS_PYPOTS_VENV_DIR:-${REPO_ROOT}/venv_saits_pypots}"
BOOTSTRAP_PY="${BOOTSTRAP_PY:-${REPO_ROOT}/venv/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
PYPOTS_DIR="${PYPOTS_DIR:-${REPO_ROOT}/third_party/PyPOTS}"

if [[ ! -d "${SAITS_PYPOTS_VENV_DIR}" ]]; then
  if [[ -x "${BOOTSTRAP_PY}" ]]; then
    "${BOOTSTRAP_PY}" -m virtualenv "${SAITS_PYPOTS_VENV_DIR}" || "${BOOTSTRAP_PY}" -m venv "${SAITS_PYPOTS_VENV_DIR}"
  else
    "${PYTHON_BIN}" -m venv "${SAITS_PYPOTS_VENV_DIR}"
  fi
fi

# shellcheck disable=SC1090
source "${SAITS_PYPOTS_VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install torch + core deps (GPU wheels via TORCH_INDEX_URL).
python -m pip install --index-url "${TORCH_INDEX_URL}" torch
python -m pip install numpy pandas scikit-learn tqdm tabpfn

if [[ ! -d "${PYPOTS_DIR}" ]]; then
  mkdir -p "$(dirname "${PYPOTS_DIR}")"
  git clone https://github.com/WenjieDu/PyPOTS.git "${PYPOTS_DIR}"
fi

python -m pip install -e "${PYPOTS_DIR}"

python - <<'PY'
import torch
import numpy
import pandas
import sklearn
import pypots

print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("pandas", pandas.__version__)
print("sklearn", sklearn.__version__)
print("pypots", pypots.__version__)
PY
