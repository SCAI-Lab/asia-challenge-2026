#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/00_env.sh"

source "${ASIA2026_VENV_DIR}/bin/activate"

RAW_ZIPS_DIR="${ASIA2026_RAW_ZIPS_DIR}"
mkdir -p "${ASIA2026_DATA_DIR}" "${RAW_ZIPS_DIR}"

if [ ! -f "${RAW_ZIPS_DIR}/Share_Track1.zip" ] || [ ! -f "${RAW_ZIPS_DIR}/Share_Track2.zip" ]; then
  echo "[12_stage_data] ERROR: Missing raw zips under ${RAW_ZIPS_DIR}"
  exit 1
fi

python "${ASIA2026_REPO_DIR}/scripts/unpack_data.py" \
  --track1-zip "${RAW_ZIPS_DIR}/Share_Track1.zip" \
  --track2-zip "${RAW_ZIPS_DIR}/Share_Track2.zip" \
  --out-root "${ASIA2026_DATA_DIR}"

echo "[12_stage_data] Done -> ${ASIA2026_DATA_DIR}"
