#!/bin/bash
#SBATCH --job-name=asia2026_t1_tabimpute
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

echo "[t1_tabimpute] venv=${TABIM_VENV}"
echo "[t1_tabimpute] hf_home=${HF_HOME}"
echo "[t1_tabimpute] calibrate=${CALIBRATE:-0} row_chunk=${ROW_CHUNK:-4} context_n=${CONTEXT_N:-128}"

source "${TABIM_VENV}/bin/activate"

export PYTHONPATH="/cluster/scratch/${USER}/Acads/SCAI/asia2026_starter/src:${PYTHONPATH:-}"

echo "[t1_tabimpute] running tabimpute"
CALIBRATE_FLAG=""
if [ "${CALIBRATE:-0}" = "1" ]; then
  CALIBRATE_FLAG="--calibrate"
fi
ROW_CHUNK_FLAG=""
if [ -n "${ROW_CHUNK:-}" ]; then
  ROW_CHUNK_FLAG="--row-chunk ${ROW_CHUNK}"
fi
CONTEXT_FLAG=""
if [ -n "${CONTEXT_N:-}" ]; then
  CONTEXT_FLAG="--context-n ${CONTEXT_N}"
fi
python /cluster/scratch/${USER}/Acads/SCAI/asia2026_starter/scripts/run_tabimpute_t1.py \
  --data-root /cluster/scratch/${USER}/Acads/SCAI/asia2026_starter/data/staged \
  --run-root ${ASIA2026_ROOT}/runs \
  --mode tabimpute \
  ${CALIBRATE_FLAG} \
  ${ROW_CHUNK_FLAG} \
  ${CONTEXT_FLAG}

echo "[t1_tabimpute] done"
