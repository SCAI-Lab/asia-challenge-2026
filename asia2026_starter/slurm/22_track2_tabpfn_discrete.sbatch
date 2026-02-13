#!/bin/bash
#SBATCH --job-name=asia2026_t2_tabpfn_discrete
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

source "$SLURM_SUBMIT_DIR/scripts/00_env.sh"
source "${ASIA2026_VENV_DIR}/bin/activate"

export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

cd "${ASIA2026_REPO_DIR}"

echo "[t2_tabpfn_discrete] running tabpfn_25_discrete"
python "$SLURM_SUBMIT_DIR/scripts/run_tabpfn_t2_discrete.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --do-cv 1 \
  --n-splits 3

echo "[t2_tabpfn_discrete] done"
