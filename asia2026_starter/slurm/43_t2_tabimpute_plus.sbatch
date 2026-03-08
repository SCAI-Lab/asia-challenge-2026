#!/bin/bash
#SBATCH --job-name=asia2026_t2_tabimpute_plus
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

module purge || true
module load stack/2024-06 || true
module load python_cuda/3.11.6 || module load python/3.12.8 || module load python/3.11.6 || module load python/3.10.13

ASIA2026_ROOT="/cluster/scratch/${USER}/Acads/SCAI/asia2026_starter"
TABIM_VENV="${ASIA2026_ROOT}/venv_tabimpute_plus"
export HF_HOME="${ASIA2026_ROOT}/hf_home_tabimpute_plus"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1
export XDG_CACHE_HOME="${ASIA2026_ROOT}/xdg_cache"
export TABPFN_CACHE_DIR="${XDG_CACHE_HOME}/tabpfn"
export TABPFN_REGRESSOR_MODEL_PATH="${TABPFN_CACHE_DIR}/tabpfn-v2-regressor.ckpt"

echo "[t2_tabimpute_plus] venv=${TABIM_VENV}"
echo "[t2_tabimpute_plus] hf_home=${HF_HOME}"

source "${TABIM_VENV}/bin/activate"

export PYTHONPATH="/cluster/scratch/${USER}/Acads/SCAI/asia2026_starter/src:${PYTHONPATH:-}"

echo "[t2_tabimpute_plus] running tabimpute_plus"
python /cluster/scratch/${USER}/Acads/SCAI/asia2026_starter/scripts/run_tabimpute_t2.py \
  --data-root /cluster/scratch/${USER}/Acads/SCAI/asia2026_starter/data/staged \
  --run-root ${ASIA2026_ROOT}/runs \
  --row-chunk 8 \
  --mode tabimpute_plus

echo "[t2_tabimpute_plus] done"
