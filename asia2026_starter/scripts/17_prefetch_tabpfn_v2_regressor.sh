#!/usr/bin/env bash
set -euo pipefail

# Prefetch TabPFN v2 regressor weights for TabImpute+ offline runs.

ASIA2026_ROOT="${ASIA2026_ROOT:-/cluster/scratch/${USER}/Acads/SCAI/asia2026_starter}"
TABIM_VENV="${ASIA2026_ROOT}/venv_tabimpute_plus"
export HF_HOME="${ASIA2026_ROOT}/hf_home_tabimpute_plus"
export HF_HUB_CACHE="${HF_HOME}/hub"
export XDG_CACHE_HOME="${ASIA2026_ROOT}/xdg_cache"
export TABPFN_CACHE_DIR="${XDG_CACHE_HOME}/tabpfn"

mkdir -p "${HF_HUB_CACHE}" "${TABPFN_CACHE_DIR}"

module purge || true
module load stack/2024-06 || true
module load python_cuda/3.11.6 || module load python/3.12.8 || module load python/3.11.6 || module load python/3.10.13
module load eth_proxy

if [ ! -f "${TABIM_VENV}/bin/activate" ]; then
  echo "[17_prefetch_tabpfn_v2_regressor] Missing venv: ${TABIM_VENV}" >&2
  echo "[17_prefetch_tabpfn_v2_regressor] Run scripts/16_prefetch_tabimpute_plus.sh first." >&2
  exit 1
fi
source "${TABIM_VENV}/bin/activate"

export HF_HUB_OFFLINE=0

python - <<'PY'
import os
from pathlib import Path

from tabpfn import TabPFNRegressor
from huggingface_hub import hf_hub_download

cache_dir = Path(os.environ.get("TABPFN_CACHE_DIR", ""))
if not cache_dir:
    raise SystemExit("TABPFN_CACHE_DIR is not set")
cache_dir.mkdir(parents=True, exist_ok=True)
target = cache_dir / "tabpfn-v2-regressor.ckpt"

if not target.exists():
    print("[17_prefetch_tabpfn_v2_regressor] initializing TabPFNRegressor to trigger download")
    TabPFNRegressor(device="cpu")

if not target.exists():
    print("[17_prefetch_tabpfn_v2_regressor] fallback to hf_hub_download")
    hf_hub_download(
        repo_id="Prior-Labs/TabPFN-v2-reg",
        filename="tabpfn-v2-regressor.ckpt",
        local_dir=str(cache_dir),
        local_dir_use_symlinks=False,
    )

if not target.exists():
    raise SystemExit(f"TabPFN v2 regressor checkpoint not found at {target}")

print("[17_prefetch_tabpfn_v2_regressor] cached:", target)
PY

printf '%s\n' "[17_prefetch_tabpfn_v2_regressor] Done. Cache under ${TABPFN_CACHE_DIR}."
