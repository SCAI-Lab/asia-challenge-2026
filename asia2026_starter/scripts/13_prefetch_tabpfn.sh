#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/00_env.sh"

# External access requires eth_proxy on Euler.
module purge || true
module load stack/2024-06 || true
module load eth_proxy

source "${ASIA2026_VENV_DIR}/bin/activate"

# During setup we ALLOW internet access (eth_proxy) so TabPFN can download its weights.
export HF_HUB_OFFLINE=0

python - <<'PY'
import numpy as np
from tabpfn import TabPFNRegressor

X = np.random.randn(256, 16).astype('float32')
y = np.random.randn(256).astype('float32')

model = TabPFNRegressor(device='cuda' if __import__('torch').cuda.is_available() else 'cpu', ignore_pretraining_limits=True)
model.fit(X, y)
pred = model.predict(X[:4])
print('[13_prefetch_tabpfn] OK, pred[:2]=', pred[:2])
PY

echo "[13_prefetch_tabpfn] TabPFN weights prefetched (cache under $HF_HOME / $XDG_CACHE_HOME)."
