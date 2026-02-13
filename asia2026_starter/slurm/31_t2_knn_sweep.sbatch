#!/bin/bash
#SBATCH --job-name=asia2026_t2_knn_sweep
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/00_env.sh"
source "${ASIA2026_VENV_DIR}/bin/activate"

export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

python "$SLURM_SUBMIT_DIR/scripts/tune_knn_sweep.py" \
  --track 2 \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --do-cv 1 \
  --n-splits 5
