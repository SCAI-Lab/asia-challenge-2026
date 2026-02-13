#!/usr/bin/env bash
set -euo pipefail

ROOT="/cluster/scratch/ppurkayastha/Acads/SCAI"
ZIP_NAME="${ROOT}/asia2026.zip"

INCLUDE_DIRS=(
  "asia2026_starter"
  "slurm_logs"
)

EXCLUDE_PATTERNS=(
  "asia2026_starter/venv/*"
  "asia2026_starter/xdg_cache/*"
  "asia2026_starter/pip_cache/*"
  "asia2026_starter/hf_home/*"
  "asia2026_starter/slurm_logs/*"
  "asia2026_starter/__pycache__/*"
)

echo "Include: ${INCLUDE_DIRS[*]}"
echo "Exclude: ${EXCLUDE_PATTERNS[*]}"

cd "$ROOT"
zip -r "$ZIP_NAME" "${INCLUDE_DIRS[@]}" -x "${EXCLUDE_PATTERNS[@]}"
