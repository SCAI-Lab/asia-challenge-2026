#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

# Base project env vars.
source "${REPO_ROOT}/scripts/00_env.sh"

export ASIA2026_SAITS_VENV_DIR="${ASIA2026_SAITS_VENV_DIR:-${REPO_ROOT}/venv_saits_pypots}"
export ASIA2026_SAITS_RUNS_DIR="${ASIA2026_SAITS_RUNS_DIR:-${REPO_ROOT}/runs_saits_pypots}"
export PYTHONPATH="${ASIA2026_REPO_DIR}/src:${PYTHONPATH:-}"
