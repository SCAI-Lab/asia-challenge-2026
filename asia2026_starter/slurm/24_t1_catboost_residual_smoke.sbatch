#!/bin/bash
#SBATCH --job-name=asia2026_t1_catboost_residual_smoke
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=rtx_4090:1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/00_env.sh"
if [[ -n "${CATBOOST_VENV_DIR:-}" ]]; then
  export ASIA2026_VENV_DIR="$CATBOOST_VENV_DIR"
fi

# shellcheck source=/dev/null
source "${ASIA2026_VENV_DIR}/bin/activate"

export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

DEVICE="${DEVICE:-gpu}"

if [[ -z "${TABPFN_RUN_DIR:-}" ]]; then
  TABPFN_RUN_DIR=$(ls -td "${ASIA2026_RUNS_DIR}"/t1_discrete_seedbag5_proba__* 2>/dev/null | head -n 1 || true)
fi
if [[ -z "${TABPFN_RUN_DIR}" || ! -d "${TABPFN_RUN_DIR}" ]]; then
  echo "[smoke] Missing TABPFN_RUN_DIR. Set TABPFN_RUN_DIR or TABPFN_OOF_PATH/TABPFN_SUBMISSION_PATH." >&2
  exit 1
fi

TABPFN_OOF_PATH="${TABPFN_OOF_PATH:-${TABPFN_RUN_DIR}/oof_predictions_train.npz}"
TABPFN_SUBMISSION_PATH="${TABPFN_SUBMISSION_PATH:-${TABPFN_RUN_DIR}/predictions_test.csv}"

if [[ ! -f "${TABPFN_OOF_PATH}" ]]; then
  echo "[smoke] Missing OOF file: ${TABPFN_OOF_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TABPFN_SUBMISSION_PATH}" ]]; then
  echo "[smoke] Missing TabPFN submission: ${TABPFN_SUBMISSION_PATH}" >&2
  exit 1
fi

echo "[smoke] Using TABPFN_RUN_DIR=${TABPFN_RUN_DIR}"

echo "[smoke] Training residual blocks"
python "$SLURM_SUBMIT_DIR/scripts/train_t1_catboost_residual_blocks.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --oof-path "${TABPFN_OOF_PATH}" \
  --device "${DEVICE}" \
  --limit-rows 50 \
  --iterations 50 \
  --od-wait 20

CATBOOST_RUN=$(ls -td "${ASIA2026_RUNS_DIR}"/t1_catboost_residual_blocks_* 2>/dev/null | head -n 1)
if [[ -z "${CATBOOST_RUN}" || ! -d "${CATBOOST_RUN}" ]]; then
  echo "[smoke] Failed to locate CatBoost residual run dir." >&2
  exit 1
fi

echo "[smoke] Predicting residual blends"
python "$SLURM_SUBMIT_DIR/scripts/predict_t1_tabpfn_plus_catboost.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --catboost-run "${CATBOOST_RUN}" \
  --tabpfn-submission "${TABPFN_SUBMISSION_PATH}" \
  --alphas "0.1,0.2,0.3" \
  --limit-test-rows 50

echo "[smoke] Done. CatBoost run: ${CATBOOST_RUN}"
