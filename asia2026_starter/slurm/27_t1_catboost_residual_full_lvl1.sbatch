#!/bin/bash
#SBATCH --job-name=asia2026_t1_catboost_residual_full_lvl1
#SBATCH --time=24:00:00
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
TOPK_TARGETS="${TOPK_TARGETS:-0}"
TOPK_SOURCE="${TOPK_SOURCE:-test}"

if [[ -z "${TABPFN_RUN_DIR:-}" ]]; then
  TABPFN_RUN_DIR=$(ls -td "${ASIA2026_RUNS_DIR}"/t1_discrete_seedbag5_proba__* 2>/dev/null | head -n 1 || true)
fi
if [[ -z "${TABPFN_RUN_DIR}" || ! -d "${TABPFN_RUN_DIR}" ]]; then
  echo "[full] Missing TABPFN_RUN_DIR. Set TABPFN_RUN_DIR or TABPFN_OOF_PATH/TABPFN_SUBMISSION_PATH." >&2
  exit 1
fi

TABPFN_OOF_PATH="${TABPFN_OOF_PATH:-${TABPFN_RUN_DIR}/oof_predictions_train.npz}"
TABPFN_SUBMISSION_PATH="${TABPFN_SUBMISSION_PATH:-${TABPFN_RUN_DIR}/predictions_test.csv}"

if [[ ! -f "${TABPFN_OOF_PATH}" ]]; then
  echo "[full] Missing OOF file: ${TABPFN_OOF_PATH}" >&2
  exit 1
fi
if [[ ! -f "${TABPFN_SUBMISSION_PATH}" ]]; then
  echo "[full] Missing TabPFN submission: ${TABPFN_SUBMISSION_PATH}" >&2
  exit 1
fi

echo "[full] Using TABPFN_RUN_DIR=${TABPFN_RUN_DIR}"

echo "[full] Training residual blocks"
python "$SLURM_SUBMIT_DIR/scripts/train_t1_catboost_residual_blocks_lvl1.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --oof-path "${TABPFN_OOF_PATH}" \
  --device "${DEVICE}" \
  --topk-targets "${TOPK_TARGETS}" \
  --topk-source "${TOPK_SOURCE}"

CATBOOST_RUN=$(ls -td "${ASIA2026_RUNS_DIR}"/t1_catboost_residual_blocks_* 2>/dev/null | head -n 1)
if [[ -z "${CATBOOST_RUN}" || ! -d "${CATBOOST_RUN}" ]]; then
  echo "[full] Failed to locate CatBoost residual run dir." >&2
  exit 1
fi

echo "[full] Predicting residual blends"
python "$SLURM_SUBMIT_DIR/scripts/predict_t1_tabpfn_plus_catboost_lvl1.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --catboost-run "${CATBOOST_RUN}" \
  --tabpfn-submission "${TABPFN_SUBMISSION_PATH}" \
  --alphas "0.0,0.0125,0.025,0.0375,0.05,0.0675,0.075,0.0875,0.1125,0.125,0.1375,0.15,0.1675,0.175,0.1875,-0.05,-0.10,-0.15,-0.20,-0.25,-0.30,-0.35,-0.40,-0.45,-0.50" \


echo "[full] Done. CatBoost run: ${CATBOOST_RUN}"
