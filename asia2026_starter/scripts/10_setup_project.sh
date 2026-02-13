#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "${HERE}/00_env.sh"

mkdir -p "${ASIA2026_ROOT}" \
  "${ASIA2026_REPO_DIR}" \
  "${ASIA2026_DATA_DIR}" \
  "${ASIA2026_RAW_ZIPS_DIR}" \
  "${ASIA2026_RUNS_DIR}" \
  "${ASIA2026_LOGS_DIR}" \
  "${HF_HOME}" \
  "${XDG_CACHE_HOME}" \
  "${PIP_CACHE_DIR}"

echo "[10_setup_project] Using repo in-place -> ${ASIA2026_REPO_DIR}"
