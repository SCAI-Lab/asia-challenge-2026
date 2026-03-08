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
export TABPFN_CACHE_DIR="${XDG_CACHE_HOME}/tabpfn"
export HF_HUB_OFFLINE=1

# Keep BLAS threading in check; AutoTabPFN parallelism uses n_jobs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("tabpfn_extensions") else 1)
PY
then
  python -m pip install "tabpfn-extensions[post_hoc_ensembles]"
fi

N_JOBS="${N_JOBS:-${SLURM_CPUS_PER_TASK}}"
BASELINE_CSV="${BASELINE_CSV:-/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter/runs/t1_discrete_seedbag5_proba__20260222_120028__job58116257__zkw773g3/predictions_test.csv}"
PRESETS="${PRESETS:-medium_quality}"
MAX_TIME="${MAX_TIME:-180}"
N_ENSEMBLE_MODELS="${N_ENSEMBLE_MODELS:-20}"

echo "[autotabpfn] PRESETS=${PRESETS} MAX_TIME=${MAX_TIME} N_ENSEMBLE_MODELS=${N_ENSEMBLE_MODELS}"

python "$SLURM_SUBMIT_DIR/scripts/run_t1_autotabpfn_overlay_T24.py" \
  --data-root "${ASIA2026_DATA_DIR}" \
  --run-root "${ASIA2026_RUNS_DIR}" \
  --baseline-csv "${BASELINE_CSV}" \
  --target-topk 24 \
  --max-time "${MAX_TIME}" \
  --presets "${PRESETS}" \
  --n-jobs "${N_JOBS}" \
  --n-ensemble-models "${N_ENSEMBLE_MODELS}"
