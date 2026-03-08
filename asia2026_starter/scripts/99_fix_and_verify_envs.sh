#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
source "${HERE}/00_env.sh"

log() {
  echo "[fix_envs] $*"
}

bootstrap_python() {
  local candidates=(
    "${ASIA2026_ROOT}/venv_saits_pypots/bin/python"
    "${ASIA2026_ROOT}/venv_tabimpute/bin/python"
    "${ASIA2026_ROOT}/venv_tabimpute_plus/bin/python"
    "${ASIA2026_ROOT}/venv_saits_virt/bin/python"
  )
  local c
  for c in "${candidates[@]}"; do
    if [ -x "${c}" ]; then
      echo "${c}"
      return 0
    fi
  done
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return 0
  fi
  return 1
}

ensure_venv() {
  local venv_path="$1"
  if [ ! -x "${venv_path}/bin/python" ]; then
    echo "[fix_envs] missing venv python: ${venv_path}/bin/python" >&2
    exit 1
  fi
}

recreate_venv() {
  local venv_path="$1"
  local py_bin
  py_bin="$(bootstrap_python)" || {
    echo "[fix_envs] missing bootstrap python (python3/python) to recreate venv" >&2
    exit 1
  }
  rm -rf "${venv_path}"
  if ! "${py_bin}" -m venv "${venv_path}"; then
    if "${py_bin}" -m virtualenv --version >/dev/null 2>&1; then
      "${py_bin}" -m virtualenv "${venv_path}"
    else
      echo "[fix_envs] failed to create venv with ${py_bin}" >&2
      exit 1
    fi
  fi
  clean_distutils_pth "${venv_path}"
  log "recreated venv: ${venv_path}"
}

ensure_pip() {
  local venv_path="$1"
  local py="${venv_path}/bin/python"
  local rebuilt=0
  if ! "${py}" -m pip --version >/dev/null 2>&1; then
    log "pip missing/broken in ${venv_path}; running ensurepip"
    "${py}" -m ensurepip --upgrade >/dev/null 2>&1 || true
  fi
  if ! "${py}" -m pip --version >/dev/null 2>&1; then
    log "pip still unavailable in ${venv_path}; recreating venv"
    recreate_venv "${venv_path}"
    py="${venv_path}/bin/python"
    "${py}" -m ensurepip --upgrade >/dev/null 2>&1 || true
    rebuilt=1
  fi
  if ! "${py}" -m pip --version >/dev/null 2>&1; then
    echo "[fix_envs] pip still unavailable after recreate in ${venv_path}" >&2
    exit 1
  fi
  echo "${rebuilt}"
}

clean_distutils_pth() {
  local venv_path="$1"
  local pth="${venv_path}/lib/python3.11/site-packages/distutils-precedence.pth"
  if [ -f "${pth}" ]; then
    rm -f "${pth}"
    log "removed distutils-precedence.pth from ${venv_path}"
  fi
}

check_pkgs() {
  local label="$1"
  shift
  local pkgs="$*"
  PKGS="${pkgs}" python - <<'PY'
import importlib.util
import os
import sys

pkgs = os.environ["PKGS"].split()
missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
if missing:
    print("[fix_envs] missing packages:", ", ".join(missing))
    sys.exit(1)
print("[fix_envs] ok packages:", ", ".join(pkgs))
PY
  log "${label} packages OK"
}

patch_tabimpute_interface() {
  local repo_root="$1"
  TABIM_REPO="${repo_root}" python - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["TABIM_REPO"]) / "mcpfn/src/tabimpute/interface.py"
txt = p.read_text()

if "tabpfn_extensions" in txt and "from tabpfn import TabPFNClassifier" in txt:
    print("[fix_envs] interface.py already patched:", p)
    raise SystemExit(0)

marker_start = "from sklearn.linear_model import LinearRegression\n"
marker_end = "import importlib.resources as resources\n"
if marker_start not in txt or marker_end not in txt:
    raise SystemExit("[fix_envs] Unexpected interface.py layout; aborting patch.")

prefix, rest = txt.split(marker_start, 1)
_, suffix = rest.split(marker_end, 1)

patched_block = (
    "from sklearn.linear_model import LinearRegression\n"
    "try:\n"
    "    from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised\n"
    "except Exception:\n"
    "    try:\n"
    "        from tabpfn import TabPFNClassifier, TabPFNRegressor\n"
    "        unsupervised = None\n"
    "    except Exception:\n"
    "        TabPFNClassifier = None\n"
    "        TabPFNRegressor = None\n"
    "        unsupervised = None\n"
    "import importlib.resources as resources\n"
)

p.write_text(prefix + patched_block + suffix)
print("[fix_envs] patched interface.py:", p)
PY
}

verify_tabimpute_cache() {
  local hf_home="$1"
  local repo_id="Tabimpute/TabImpute"
  local cache_dir="${hf_home}/hub"
  local snapshots_dir="${cache_dir}/models--Tabimpute--TabImpute/snapshots"

  if [ -d "${snapshots_dir}" ] && [ "$(find "${snapshots_dir}" -maxdepth 1 -mindepth 1 -type d | wc -l)" -gt 0 ]; then
    log "TabImpute cache present: ${snapshots_dir}"
    return 0
  fi

  log "TabImpute cache missing; downloading to ${cache_dir}"
  HF_HOME="${hf_home}" HF_HUB_CACHE="${cache_dir}" HF_HUB_OFFLINE=0 python - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = "Tabimpute/TabImpute"
cache_dir = os.environ["HF_HUB_CACHE"]
snapshot_download(repo_id=repo_id, cache_dir=cache_dir, local_files_only=False)
print("[fix_envs] downloaded:", repo_id, "to", cache_dir)
PY
}

ensure_tabpfn_v2_regressor() {
  local cache_dir="${XDG_CACHE_HOME}/tabpfn"
  local target="${cache_dir}/tabpfn-v2-regressor.ckpt"

  mkdir -p "${cache_dir}"
  if [ -f "${target}" ]; then
    log "TabPFN v2 regressor present: ${target}"
    return 0
  fi

  log "TabPFN v2 regressor missing; downloading to ${target}"
  HF_HUB_OFFLINE=0 TABPFN_CACHE_DIR="${cache_dir}" python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

cache_dir = Path(os.environ["TABPFN_CACHE_DIR"])
cache_dir.mkdir(parents=True, exist_ok=True)
target = cache_dir / "tabpfn-v2-regressor.ckpt"
hf_hub_download(
    repo_id="Prior-Labs/TabPFN-v2-reg",
    filename="tabpfn-v2-regressor.ckpt",
    local_dir=str(cache_dir),
    local_dir_use_symlinks=False,
)
if not target.exists():
    raise SystemExit(f"missing {target}")
print("[fix_envs] downloaded:", target)
PY
}

ensure_tabpfn_v25_weights() {
  local cache_dir="${XDG_CACHE_HOME}/tabpfn"
  local reg="${cache_dir}/tabpfn-v2.5-regressor-v2.5_default.ckpt"
  local clf="${cache_dir}/tabpfn-v2.5-classifier-v2.5_default.ckpt"

  if [ -f "${reg}" ] && [ -f "${clf}" ]; then
    log "TabPFN v2.5 weights present under ${cache_dir}"
    return 0
  fi

  log "TabPFN v2.5 weights missing; downloading to ${cache_dir}"
  HF_HUB_OFFLINE=0 TABPFN_CACHE_DIR="${cache_dir}" python - <<'PY'
from pathlib import Path

from tabpfn import TabPFNClassifier, TabPFNRegressor

cache_dir = Path(__import__("os").environ["TABPFN_CACHE_DIR"])
cache_dir.mkdir(parents=True, exist_ok=True)

TabPFNRegressor(device="cpu", ignore_pretraining_limits=True)
TabPFNClassifier(device="cpu", ignore_pretraining_limits=True)
print("[fix_envs] TabPFN v2.5 download triggered")
PY

  if [ ! -f "${reg}" ] || [ ! -f "${clf}" ]; then
    echo "[fix_envs] missing TabPFN v2.5 weights after download: ${cache_dir}" >&2
    exit 1
  fi
}

verify_data_files() {
  local base="${ASIA2026_DATA_DIR}"
  local t1="${base}/track1"
  local t2="${base}/track2"
  local t1_files=(features_train_1.csv features_test_1.csv labels_train_1.csv metadata_train_1.csv metadata_test_1.csv)
  local t2_files=(features_train_2.csv features_test_2.csv labels_train_2.csv metadata_train_2.csv metadata_test_2.csv)

  for f in "${t1_files[@]}"; do
    if [ ! -f "${t1}/${f}" ]; then
      echo "[fix_envs] missing Track1 file: ${t1}/${f}" >&2
      exit 1
    fi
  done
  for f in "${t2_files[@]}"; do
    if [ ! -f "${t2}/${f}" ]; then
      echo "[fix_envs] missing Track2 file: ${t2}/${f}" >&2
      exit 1
    fi
  done
  log "Track1/Track2 staged data OK"
}

main() {
  log "repo: ${REPO_ROOT}"
  log "ASIA2026_ROOT: ${ASIA2026_ROOT}"

  ensure_venv "${ASIA2026_VENV_DIR}"
  ensure_venv "${ASIA2026_ROOT}/venv_tabimpute"
  ensure_venv "${ASIA2026_ROOT}/venv_tabimpute_plus"
  ensure_venv "${ASIA2026_ROOT}/venv_saits_pypots"
  ensure_venv "${ASIA2026_ROOT}/venv_saits_virt"

  # ---- main venv (TabPFN + CatBoost) ----
  log "fixing main venv: ${ASIA2026_VENV_DIR}"
  clean_distutils_pth "${ASIA2026_VENV_DIR}"
  ensure_pip "${ASIA2026_VENV_DIR}" >/dev/null
  source "${ASIA2026_VENV_DIR}/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install --index-url "https://download.pytorch.org/whl/cu124" torch
  python -m pip install -r "${ASIA2026_REPO_DIR}/requirements.txt"
  python -m pip install catboost typing_extensions six
  check_pkgs "main venv" numpy pandas sklearn torch tqdm tabpfn catboost typing_extensions six
  ensure_tabpfn_v25_weights
  deactivate

  # ---- tabimpute venv ----
  log "fixing tabimpute venv: ${ASIA2026_ROOT}/venv_tabimpute"
  clean_distutils_pth "${ASIA2026_ROOT}/venv_tabimpute"
  ensure_pip "${ASIA2026_ROOT}/venv_tabimpute" >/dev/null
  source "${ASIA2026_ROOT}/venv_tabimpute/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install "torch>=2.2,<3" "scikit-learn==1.4.2" numpy pandas scipy joblib einops huggingface-hub tqdm typing_extensions six "tabpfn==2.1.3"
  python -m pip install -e "${ASIA2026_ROOT}/third_party/tabimpute_tabular/mcpfn"
  patch_tabimpute_interface "${ASIA2026_ROOT}/third_party/tabimpute_tabular"
  check_pkgs "tabimpute venv" numpy pandas sklearn torch tabpfn tabimpute huggingface_hub typing_extensions six
  verify_tabimpute_cache "${ASIA2026_ROOT}/hf_home_tabimpute"
  deactivate

  # ---- tabimpute plus venv ----
  log "fixing tabimpute-plus venv: ${ASIA2026_ROOT}/venv_tabimpute_plus"
  clean_distutils_pth "${ASIA2026_ROOT}/venv_tabimpute_plus"
  ensure_pip "${ASIA2026_ROOT}/venv_tabimpute_plus" >/dev/null
  source "${ASIA2026_ROOT}/venv_tabimpute_plus/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
  python -m pip install "torch>=2.2,<3" "scikit-learn==1.4.2" numpy pandas scipy joblib einops huggingface-hub tqdm typing_extensions six "tabpfn==2.1.3"
  python -m pip install -e "${ASIA2026_ROOT}/third_party/tabimpute_tabular/mcpfn"
  patch_tabimpute_interface "${ASIA2026_ROOT}/third_party/tabimpute_tabular"
  check_pkgs "tabimpute-plus venv" numpy pandas sklearn torch tabpfn tabimpute huggingface_hub typing_extensions six
  verify_tabimpute_cache "${ASIA2026_ROOT}/hf_home_tabimpute_plus"
  ensure_tabpfn_v2_regressor
  deactivate

  # ---- SAITS (PyPOTS) venv ----
  log "verifying SAITS PyPOTS venv: ${ASIA2026_ROOT}/venv_saits_pypots"
  clean_distutils_pth "${ASIA2026_ROOT}/venv_saits_pypots"
  ensure_pip "${ASIA2026_ROOT}/venv_saits_pypots" >/dev/null
  source "${ASIA2026_ROOT}/venv_saits_pypots/bin/activate"
  if ! check_pkgs "saits pypots venv" numpy pandas torch tqdm pypots tabpfn typing_extensions six; then
    deactivate
    log "repairing SAITS PyPOTS venv via setup script"
    bash "${ASIA2026_REPO_DIR}/scripts_saits_pypots/00_setup_saits_pypots_env.sh"
    source "${ASIA2026_ROOT}/venv_saits_pypots/bin/activate"
    check_pkgs "saits pypots venv" numpy pandas torch tqdm pypots tabpfn typing_extensions six
  fi
  deactivate

  # ---- SAITS (custom) venv ----
  log "verifying SAITS custom venv: ${ASIA2026_ROOT}/venv_saits_virt"
  clean_distutils_pth "${ASIA2026_ROOT}/venv_saits_virt"
  ensure_pip "${ASIA2026_ROOT}/venv_saits_virt" >/dev/null
  source "${ASIA2026_ROOT}/venv_saits_virt/bin/activate"
  if ! check_pkgs "saits custom venv" numpy pandas torch tqdm; then
    deactivate
    log "repairing SAITS custom venv via setup script"
    bash "${ASIA2026_REPO_DIR}/scripts/19_setup_saits_env.sh"
    source "${ASIA2026_ROOT}/venv_saits_virt/bin/activate"
    check_pkgs "saits custom venv" numpy pandas torch tqdm
  fi
  deactivate

  # ---- data + weights checks ----
  verify_data_files
  log "all checks complete"
}

main "$@"
