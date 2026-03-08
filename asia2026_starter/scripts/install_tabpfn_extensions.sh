#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"

source "${REPO_ROOT}/scripts/00_env.sh"
source "${ASIA2026_VENV_DIR}/bin/activate"

python -m pip install -U "setuptools<70"
python -m pip install "tabpfn-extensions[post_hoc_ensembles]"

python - <<'PY'
import importlib.util

print("tabpfn_extensions present:", bool(importlib.util.find_spec("tabpfn_extensions")))

candidates = [
    "tabpfn_extensions.post_hoc_ensembles",
    "tabpfn_extensions.auto_tabpfn",
    "tabpfn_extensions",
]

AutoTabPFNClassifier = None
for mod in candidates:
    try:
        m = __import__(mod, fromlist=["AutoTabPFNClassifier"])
        if hasattr(m, "AutoTabPFNClassifier"):
            AutoTabPFNClassifier = m.AutoTabPFNClassifier
            print(f"Imported AutoTabPFNClassifier from {mod}")
            break
    except Exception as exc:
        print(f"Import failed from {mod}: {exc}")

if AutoTabPFNClassifier is None:
    raise SystemExit("AutoTabPFNClassifier not found in tabpfn_extensions.")
print("OK: AutoTabPFNClassifier is available.")
PY
