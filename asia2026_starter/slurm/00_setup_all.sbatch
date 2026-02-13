#!/bin/bash
#SBATCH --job-name=asia2026_setup
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

echo "[setup] submit_dir=$SLURM_SUBMIT_DIR"

# Build venv + data + TabPFN weights in the starter repo.
bash "$SLURM_SUBMIT_DIR/scripts/10_setup_project.sh"
bash "$SLURM_SUBMIT_DIR/scripts/11_create_venv.sh"
bash "$SLURM_SUBMIT_DIR/scripts/12_stage_data.sh"
bash "$SLURM_SUBMIT_DIR/scripts/13_prefetch_tabpfn.sh"

echo "[setup] done"
