#!/usr/bin/env bash
set -euo pipefail

# ---------- User-tunable (defaults are safe) ----------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

# Project root is the starter repo (everything stays under this path).
export ASIA2026_ROOT="${ASIA2026_ROOT:-${REPO_ROOT}}"

# Optional: Hugging Face token (needed for gated models such as TabPFN v2.5).
# Put your token here or export it in your shell before submitting jobs.
export HF_TOKEN="${HF_TOKEN:-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}"
# Some libraries read these names instead; keep them in sync.
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN}}"
export HF_HUB_TOKEN="${HF_HUB_TOKEN:-${HF_TOKEN}}"

# ---------- Derived paths ----------

export ASIA2026_REPO_DIR="${ASIA2026_ROOT}"
export ASIA2026_DATA_DIR="${ASIA2026_ROOT}/data/staged"
export ASIA2026_RAW_ZIPS_DIR="${ASIA2026_ROOT}/data/raw_zips"
export ASIA2026_RUNS_DIR="${ASIA2026_ROOT}/runs"
export ASIA2026_LOGS_DIR="${ASIA2026_ROOT}/logs"
export ASIA2026_VENV_DIR="${ASIA2026_ROOT}/venv"

# ---------- Cache control (prevent $HOME quota issues) ----------

export HF_HOME="${ASIA2026_ROOT}/hf_home"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HUB_CACHE}"
export XDG_CACHE_HOME="${ASIA2026_ROOT}/xdg_cache"
export PIP_CACHE_DIR="${ASIA2026_ROOT}/pip_cache"

# Offline safety in compute jobs (setup job prefetches weights)
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

# TabPFN telemetry opt-out
export TABPFN_DISABLE_TELEMETRY="${TABPFN_DISABLE_TELEMETRY:-1}"

# Small quality-of-life defaults
export PYTHONUNBUFFERED=1
