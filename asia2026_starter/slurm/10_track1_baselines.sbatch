#!/bin/bash
#SBATCH --job-name=asia2026_t1_baselines
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

source "$SLURM_SUBMIT_DIR/scripts/00_env.sh"
source "${ASIA2026_VENV_DIR}/bin/activate"

export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1

cd "${ASIA2026_REPO_DIR}"

for METHOD in baseline_time_mean baseline_strat_time_mean baseline_knn15; do
  echo "[t1_baselines] running $METHOD"
  python -m asia2026.run \
    --track 1 \
    --method "$METHOD" \
    --data-root "${ASIA2026_DATA_DIR}" \
    --run-root "${ASIA2026_RUNS_DIR}" \
    --do-cv 1 \
    --n-splits 5
done

echo "[t1_baselines] done"
