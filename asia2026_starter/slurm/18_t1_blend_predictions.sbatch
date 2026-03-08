#!/bin/bash
#SBATCH --job-name=asia2026_t1_blend_predictions
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/00_env.sh"
source "${ASIA2026_VENV_DIR}/bin/activate"

export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

TABPFN_RUN="${TABPFN_RUN:-}"
CATBOOST_RUN="${CATBOOST_RUN:-}"
ALPHAS="${ALPHAS:-0.60,0.70,0.80,0.90}"

if [[ -z "${TABPFN_RUN}" || -z "${CATBOOST_RUN}" ]]; then
  echo "Missing TABPFN_RUN or CATBOOST_RUN env var." >&2
  exit 1
fi

OUT_DIR="${ASIA2026_RUNS_DIR}/blend_t1_stage3"

python "$SLURM_SUBMIT_DIR/scripts/blend_t1_predictions.py" \
  --tabpfn-run "${TABPFN_RUN}" \
  --catboost-run "${CATBOOST_RUN}" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --out-dir "${OUT_DIR}" \
  --alphas "${ALPHAS}"
