#!/usr/bin/env bash
set -euo pipefail

# Submit the full pipeline (setup -> baselines -> TabPFN) with Slurm dependencies.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"

echo "[submit_all] Submitting from: ${ROOT}"...

# 1) setup
SETUP_JOB_ID=$(sbatch --parsable "${ROOT}/slurm/00_setup_all.sbatch")
echo "[submit_all] Setup job: ${SETUP_JOB_ID}"

# 2) baselines (CPU)
T1_BASE_JOB_ID=$(sbatch --parsable --dependency=afterok:${SETUP_JOB_ID} "${ROOT}/slurm/10_track1_baselines.sbatch")
T2_BASE_JOB_ID=$(sbatch --parsable --dependency=afterok:${SETUP_JOB_ID} "${ROOT}/slurm/20_track2_baselines.sbatch")
echo "[submit_all] Track1 baselines: ${T1_BASE_JOB_ID}"
echo "[submit_all] Track2 baselines: ${T2_BASE_JOB_ID}"

# 3) TabPFN (GPU)
T1_PFN_JOB_ID=$(sbatch --parsable --dependency=afterok:${SETUP_JOB_ID} "${ROOT}/slurm/11_track1_tabpfn.sbatch")
T2_PFN_JOB_ID=$(sbatch --parsable --dependency=afterok:${SETUP_JOB_ID} "${ROOT}/slurm/21_track2_tabpfn.sbatch")
echo "[submit_all] Track1 TabPFN: ${T1_PFN_JOB_ID}"
echo "[submit_all] Track2 TabPFN: ${T2_PFN_JOB_ID}"

echo "[submit_all] Done. Monitor with: squeue -u $USER"
