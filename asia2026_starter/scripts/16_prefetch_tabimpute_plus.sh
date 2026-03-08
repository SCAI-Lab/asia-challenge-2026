#!/usr/bin/env bash
set -euo pipefail

# Simple, pinned TabImpute setup in a dedicated venv.

ASIA2026_ROOT="${ASIA2026_ROOT:-/cluster/scratch/${USER}/Acads/SCAI/asia2026_starter}"
TABIM_VENV="${ASIA2026_ROOT}/venv_tabimpute_plus"
TABIM_REPO="${ASIA2026_ROOT}/third_party/tabimpute_tabular"
export HF_HOME="${ASIA2026_ROOT}/hf_home_tabimpute_plus"
export HF_HUB_CACHE="${HF_HOME}/hub"
READY_MARKER="${TABIM_VENV}/.tabimpute_plus_ready"

mkdir -p "${ASIA2026_ROOT}/third_party" "${HF_HUB_CACHE}"

# External access requires eth_proxy on Euler.
module purge || true
module load stack/2024-06 || true
module load python_cuda/3.11.6 || module load python/3.12.8 || module load python/3.11.6 || module load python/3.10.13
module load eth_proxy

python -V

if [ ! -d "${TABIM_VENV}" ]; then
  python -m venv "${TABIM_VENV}"
elif [ ! -f "${TABIM_VENV}/bin/activate" ]; then
  echo "[16_prefetch_tabimpute_plus] venv incomplete: ${TABIM_VENV}" >&2
  echo "[16_prefetch_tabimpute_plus] Removing and recreating." >&2
  rm -rf "${TABIM_VENV}"
  python -m venv "${TABIM_VENV}"
fi
source "${TABIM_VENV}/bin/activate"

NEED_INSTALL=0
if [ ! -f "${READY_MARKER}" ]; then
  NEED_INSTALL=1
fi
if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("tabimpute") else 1)
PY
then
  NEED_INSTALL=1
fi

if [ "${NEED_INSTALL}" -eq 1 ]; then
  python -m pip install -U pip wheel setuptools
  python -m pip install "torch>=2.2,<3" "scikit-learn==1.4.2" numpy pandas scipy joblib einops huggingface-hub tqdm
  python -m pip install "tabpfn==2.1.3"
fi

if [ ! -d "${TABIM_REPO}/.git" ]; then
  git clone https://github.com/jacobf18/tabular.git "${TABIM_REPO}"
fi
if [ "${NEED_INSTALL}" -eq 1 ]; then
  python -m pip install -e "${TABIM_REPO}/mcpfn"
fi

# Patch interface.py to make tabpfn_extensions optional with tabpfn fallback.
export TABIM_REPO="${TABIM_REPO}"
python - <<'PY'
import os
from pathlib import Path

p = Path(os.environ["TABIM_REPO"]) / "mcpfn/src/tabimpute/interface.py"
txt = p.read_text()
marker_start = "from sklearn.linear_model import LinearRegression\n"
marker_end = "import importlib.resources as resources\n"
if marker_start not in txt or marker_end not in txt:
    raise SystemExit("[16_prefetch_tabimpute_plus] Unexpected interface.py layout; aborting patch.")

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
print("[16_prefetch_tabimpute_plus] Patched:", p)
PY

python - <<'PY'
import os
import py_compile
from pathlib import Path

p = Path(os.environ["TABIM_REPO"]) / "mcpfn/src/tabimpute/interface.py"
py_compile.compile(str(p), doraise=True)
print("[16_prefetch_tabimpute_plus] interface.py syntax OK")
PY

# During setup we ALLOW internet access (eth_proxy) so TabImpute can download its weights.
export HF_HUB_OFFLINE=0

python - <<'PY'
import numpy as np
from tabimpute.interface import ImputePFN

X = np.random.randn(8, 16).astype("float32")
X[np.random.rand(*X.shape) < 0.2] = np.nan
imputer = ImputePFN(device="cpu")
out = imputer.impute(X.copy())
print("[16_prefetch_tabimpute_plus] OK, out shape:", out.shape)
PY

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${READY_MARKER}"

printf '%s\n' "[16_prefetch_tabimpute_plus] TabImpute checkpoint prefetched. Cache under ${HF_HOME} / ${XDG_CACHE_HOME}."
printf '%s\n' "[16_prefetch_tabimpute_plus] TabImpute+ requires tabpfn-extensions; see notes if you want a separate env."
