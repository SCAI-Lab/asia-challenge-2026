#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/00_env.sh"

echo "[11_create_venv] Loading modules..."

module purge || true
module load stack/2024-06 || true

# Prefer CUDA-enabled python module if available (loads CUDA/NCCL/etc per ETH docs).
module load python_cuda/3.11.6 || module load python/3.12.8 || module load python/3.11.6 || module load python/3.10.13

# External access requires eth_proxy on Euler.
module load eth_proxy

python -V

if [ -d "${ASIA2026_VENV_DIR}" ]; then
  echo "[11_create_venv] Venv exists, skipping to avoid overwrite: ${ASIA2026_VENV_DIR}"
  exit 0
fi

python -m venv "${ASIA2026_VENV_DIR}"
source "${ASIA2026_VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "[11_create_venv] Installing PyTorch (CUDA wheels)..."
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124

echo "[11_create_venv] Installing requirements..."
python -m pip install -r "${ASIA2026_REPO_DIR}/requirements.txt"

python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

echo "[11_create_venv] Done -> ${ASIA2026_VENV_DIR}"
