#!/bin/bash
#SBATCH --job-name=t2_anchor_corr
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

module purge || true
module load stack/2024-06 || true
module load python_cuda/3.11.6 || module load python/3.12.8 || module load python/3.11.6 || module load python/3.10.13

TRACK2_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
VENV_DIR="${TRACK2_DIR}/venv_track2"
DATA_ROOT="${TRACK2_DIR}/data"
RUN_ROOT="${TRACK2_DIR}/runs"

source "${VENV_DIR}/bin/activate"
export PYTHONPATH="${TRACK2_DIR}:${PYTHONPATH:-}"

BASE_SUB="${TRACK2_DIR}/files/predictions_test__t2_hedge_pairwise_shrink__t2_hedge_pairwise_shrink__20260315_213234__jobnojid__yp2096zc.csv"
FEATURES_TEST="${DATA_ROOT}/features_test_2.csv"
LABELS_TRAIN="${DATA_ROOT}/labels_train_2.csv"

cd "${TRACK2_DIR}"

echo "[t2_anchor_correction] starting from ${BASE_SUB}"
python "Method_1/code/run_t2_anchor_correction.py" \
  --submission "${BASE_SUB}" \
  --features-test "${FEATURES_TEST}" \
  --labels-train "${LABELS_TRAIN}" \
  --run-root "${RUN_ROOT}"

echo "[t2_anchor_correction] done"
