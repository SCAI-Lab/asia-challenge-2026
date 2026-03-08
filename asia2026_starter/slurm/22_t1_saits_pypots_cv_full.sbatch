#!/bin/bash
#SBATCH --job-name=asia2026_t1_saits_pypots_cv_full
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts_saits_pypots/00_env_saits_pypots.sh"
source "${ASIA2026_SAITS_VENV_DIR}/bin/activate"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$SLURM_SUBMIT_DIR/scripts_saits_pypots/run_t1_saits_pypots_cv.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_SAITS_RUNS_DIR}" \
  --device cuda \
  --n-splits 5
