#!/bin/bash
#SBATCH --job-name=asia2026_t1_discrete_seedbag_proba_custom
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

# Force single-threaded preprocessing to reduce ARPACK instability.
export TABPFN_NUM_WORKERS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SEEDS="${SEEDS:-11,22,33,44,55}"
NO_CV="${NO_CV:-1}"
N_SPLITS="${N_SPLITS:-5}"
MASK_FRAC="${MASK_FRAC:-0.1}"

NO_CV_FLAG=""
if [[ "${NO_CV}" == "1" ]]; then
  NO_CV_FLAG="--no-cv"
fi

python "$SLURM_SUBMIT_DIR/scripts/run_t1_discrete_seedbag_proba_custom.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --seeds "${SEEDS}" \
  --n-splits "${N_SPLITS}" \
  --mask-frac "${MASK_FRAC}" \
  ${NO_CV_FLAG}
