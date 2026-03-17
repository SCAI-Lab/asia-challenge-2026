#!/bin/bash
#SBATCH --job-name=t2_hedge_pairwise
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
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

BASE_CSV="${TRACK2_DIR}/Method_1/data/submissions/t2_tabpfn_25_discrete_bag5__20260214_014329__job57139435__on5h0rmx.csv"

cd "${TRACK2_DIR}"

echo "[t2_hedge_pairwise] hedging ${BASE_CSV}"
python "Method_1/code/run_t2_hedge_pairwise_shrink.py" \
  --base-csv "${BASE_CSV}" \
  --data-root "${DATA_ROOT}" \
  --run-root "${RUN_ROOT}"

echo "[t2_hedge_pairwise] done"
