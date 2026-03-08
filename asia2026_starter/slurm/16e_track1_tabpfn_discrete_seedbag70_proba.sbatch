#!/bin/bash
#SBATCH --job-name=asia2026_t1_discrete_seedbag70_proba
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

NO_CV="${NO_CV:-0}"
N_SEEDS="${N_SEEDS:-5}"
TRAIN_FRAC="${TRAIN_FRAC:-0.7}"
SEED_LIST="${SEED_LIST:-}"

NO_CV_FLAG=""
if [[ "${NO_CV}" == "1" ]]; then
  NO_CV_FLAG="--no-cv"
fi

SEED_LIST_FLAG=""
if [[ -n "${SEED_LIST}" ]]; then
  SEED_LIST_FLAG="--seed-list ${SEED_LIST}"
fi

python "$SLURM_SUBMIT_DIR/scripts/run_t1_discrete_seedbag70_proba.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --train-frac "${TRAIN_FRAC}" \
  --n-seeds "${N_SEEDS}" \
  ${SEED_LIST_FLAG} \
  ${NO_CV_FLAG}
