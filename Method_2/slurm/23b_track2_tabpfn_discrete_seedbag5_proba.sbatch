#!/bin/bash
#SBATCH --job-name=t2_seedbag5_proba_m2
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

module purge || true
module load stack/2024-06 || true
module load python_cuda/3.11.6 || module load python/3.12.8 || module load python/3.11.6 || module load python/3.10.13

TRACK2_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
CONDA_DIR="${TRACK2_DIR}/miniconda"
CONDA_ENV_DIR="${TRACK2_DIR}/conda_env_311"
DATA_ROOT="${TRACK2_DIR}/data"
RUN_ROOT="${TRACK2_DIR}/runs"

source "${CONDA_DIR}/bin/activate"
conda activate "${CONDA_ENV_DIR}"

export PYTHONNOUSERSITE=1
export PYTHONPATH="${TRACK2_DIR}:${PYTHONPATH:-}"
export TABPFN_MODEL_CACHE_DIR="${TRACK2_DIR}/tabpfn_cache"
export HF_HUB_OFFLINE=1

# Force single-threaded preprocessing for stability on 1 CPU.
export TABPFN_NUM_WORKERS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "${TRACK2_DIR}"

echo "[t2_discrete_seedbag5_proba] GPU inventory"
nvidia-smi -L || true
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total --format=csv,noheader || true

echo "[t2_discrete_seedbag5_proba] running seedbag5 proba"
python "Method_2/code/run_tabpfn_t2_discrete_seedbag5_proba.py" \
  --data-root "${DATA_ROOT}" \
  --run-root "${RUN_ROOT}" \
  --do-cv 1 \
  --n-splits 5

echo "[t2_discrete_seedbag5_proba] done"
