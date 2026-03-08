#!/bin/bash
#SBATCH --job-name=asia2026_t1_tabpfn_safe_discrete_overlay
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/00_env.sh"
source "${ASIA2026_VENV_DIR}/bin/activate"

export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TABPFN_CACHE_DIR="${XDG_CACHE_HOME}/tabpfn"

# Force single-threaded preprocessing to reduce ARPACK instability.
export TABPFN_NUM_WORKERS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python "$SLURM_SUBMIT_DIR/scripts/run_tabpfn_t1_safe_discrete_overlay.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --baseline-csv "/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter/Task1_Discrete.csv" \
  --target-topk 15 \
  --seeds 11 22 33 69 \
  --do-cv 1 \
  --n-splits 5
