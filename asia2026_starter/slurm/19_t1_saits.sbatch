#!/bin/bash
#SBATCH --job-name=asia2026_t1_saits
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/00_env.sh"
SAITS_VENV_DIR="${SAITS_VENV_DIR:-${ASIA2026_REPO_DIR}/venv_saits_virt}"
source "${SAITS_VENV_DIR}/bin/activate"

export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

python "$SLURM_SUBMIT_DIR/scripts/run_t1_saits.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}"
