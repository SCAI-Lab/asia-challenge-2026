#!/bin/bash
#SBATCH --job-name=asia2026_t1_autotabpfn_overlay_T24
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/00_env.sh"
source "${ASIA2026_VENV_DIR}/bin/activate"

export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

BASELINE_CSV="${BASELINE_CSV:-}"
if [[ -z "${BASELINE_CSV}" ]]; then
  echo "Missing BASELINE_CSV (path to best Track1 submission csv)." >&2
  exit 1
fi

PRESETS="${PRESETS:-medium_quality}"
MAX_TIME="${MAX_TIME:-180}"
N_ENSEMBLE_MODELS="${N_ENSEMBLE_MODELS:-20}"
N_JOBS="${N_JOBS:-8}"
RAISE_ON_NO_MODELS_FITTED="${RAISE_ON_NO_MODELS_FITTED:-0}"
TARGET_TOPK="${TARGET_TOPK:-24}"
DYNAMIC_STACKING="${DYNAMIC_STACKING:-1}"

python "$SLURM_SUBMIT_DIR/scripts/run_t1_autotabpfn_overlay_T24.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --baseline-csv "${BASELINE_CSV}" \
  --target-topk "${TARGET_TOPK}" \
  --n-jobs "${N_JOBS}" \
  --max-time "${MAX_TIME}" \
  --presets "${PRESETS}" \
  --n-ensemble-models "${N_ENSEMBLE_MODELS}" \
  --raise-on-no-models-fitted "${RAISE_ON_NO_MODELS_FITTED}" \
  --dynamic-stacking "${DYNAMIC_STACKING}"
