#!/bin/bash
#SBATCH --job-name=asia2026_t2_tabimpute
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
TABIM_VENV="${ASIA2026_ROOT}/venv_tabimpute"
export HF_HOME="${ASIA2026_ROOT}/hf_home_tabimpute"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_OFFLINE=1

echo "[t2_tabimpute] venv=${TABIM_VENV}"
echo "[t2_tabimpute] hf_home=${HF_HOME}"

source "${TABIM_VENV}/bin/activate"

export PYTHONPATH="/cluster/scratch/${USER}/Acads/SCAI/asia2026_starter/src:${PYTHONPATH:-}"

echo "[t2_tabimpute] running tabimpute"
python /cluster/scratch/${USER}/Acads/SCAI/asia2026_starter/scripts/run_tabimpute_t2.py \
  --data-root /cluster/scratch/${USER}/Acads/SCAI/asia2026_starter/data/staged \
  --run-root ${ASIA2026_ROOT}/runs \
  --row-chunk 8 \
  --mode tabimpute

echo "[t2_tabimpute] done"
