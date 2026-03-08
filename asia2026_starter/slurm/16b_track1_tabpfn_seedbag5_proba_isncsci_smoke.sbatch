#!/bin/bash
#SBATCH --job-name=asia2026_t1_discrete_seedbag5_proba_isncsci_smoke
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/00_env.sh"
source "${ASIA2026_VENV_DIR}/bin/activate"

export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

# Force single-threaded preprocessing to reduce ARPACK instability.
export TABPFN_NUM_WORKERS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$SLURM_SUBMIT_DIR/scripts/run_t1_discrete_seedbag5_proba_isncsci.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --n-splits 2 \
  --limit-rows 256 \
  --limit-test-rows 64 \
  --limit-targets 8 \
  --mask-frac 0.05
