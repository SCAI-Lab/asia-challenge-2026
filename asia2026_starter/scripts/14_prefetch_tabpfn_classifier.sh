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
from tabpfn import TabPFNClassifier

X = np.random.randn(256, 16).astype("float32")
y = np.random.choice([0, 1, 2], size=256).astype("int64")

model = TabPFNClassifier(
    device="cuda" if __import__("torch").cuda.is_available() else "cpu",
    ignore_pretraining_limits=True,
)
model.fit(X, y)
proba = model.predict_proba(X[:4])
print("[14_prefetch_tabpfn_classifier] OK, proba[:2]=", proba[:2])
PY

echo "[14_prefetch_tabpfn_classifier] TabPFN classifier weights prefetched (cache under $HF_HOME / $XDG_CACHE_HOME)."
