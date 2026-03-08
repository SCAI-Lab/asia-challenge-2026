#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Iterable

import pandas as pd


METRIC_KEYS = [
    "r2_sensory",
    "rmse_sensory",
    "mae_sensory",
    "r2_all",
    "rmse_all",
    "r2_sensory_imputed_only",
    "rmse_sensory_imputed_only",
    "mae_sensory_imputed_only",
    "r2_all_imputed_only",
    "rmse_all_imputed_only",
    "mae_all_imputed_only",
    "mae_all",
]


def _iter_run_dirs(runs_root: Path) -> Iterable[Path]:
    if not runs_root.exists():
        return []
    for p in sorted(runs_root.iterdir()):
        if not p.is_dir():
            continue
        # Include all run directories (baselines, tabpfn, sweeps, etc.).
        yield p


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_metrics(data: dict) -> dict:
    # Most files store metrics under "overall" (cv_metrics.json) or "cv_overall" (run_summary.json).
    if isinstance(data, dict):
        if "overall" in data and isinstance(data["overall"], dict):
            return data["overall"]
        if "cv_overall" in data and isinstance(data["cv_overall"], dict):
            return data["cv_overall"]
        if "train_metrics" in data and isinstance(data["train_metrics"], dict):
            return data["train_metrics"]
    return data


def _row_for_metrics(run_name: str, path: Path, data: dict) -> str:
    metrics = _extract_metrics(data)
    row = [run_name, path.name]
    for k in METRIC_KEYS:
        row.append(str(metrics.get(k)))
    return "\t".join(row)


def _extract_job_id(run_name: str) -> str:
    m = re.search(r"__job(\d+)__", run_name)
    return m.group(1) if m else ""


def _sacct_status(job_ids: list[str]) -> dict[str, str]:
    if not job_ids:
        return {}
    cmd = [
        "sacct",
        "-j",
        ",".join(job_ids),
        "--format=JobID,State",
        "-X",
        "-n",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return {}
    status = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            status[parts[0]] = parts[1]
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report metrics for all baseline and TabPFN runs."
    )
    parser.add_argument(
        "--runs-root",
        default=os.environ.get("ASIA2026_RUNS_DIR", ""),
        help="Path to runs directory (defaults to $ASIA2026_RUNS_DIR).",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root) if args.runs_root else None
    if runs_root is None:
        print("ERROR: --runs-root not set and ASIA2026_RUNS_DIR is empty.")
        return 2

    if not runs_root.exists():
        print(f"ERROR: runs root not found: {runs_root}")
        return 2

    rows = []
    processed_files = []
    skipped_files = []
    for run_dir in _iter_run_dirs(runs_root):
        cv_metrics = run_dir / "cv_metrics.json"
        run_summary = run_dir / "run_summary.json"

        # Prefer run_summary.json for a single row per run.
        if run_summary.exists():
            if run_summary.stat().st_size == 0:
                skipped_files.append(str(run_summary))
            else:
                data = _load_json(run_summary)
                rows.append(_row_for_metrics(run_dir.name, run_summary, data))
                processed_files.append(str(run_summary))
        elif cv_metrics.exists():
            if cv_metrics.stat().st_size == 0:
                skipped_files.append(str(cv_metrics))
            else:
                data = _load_json(cv_metrics)
                rows.append(_row_for_metrics(run_dir.name, cv_metrics, data))
                processed_files.append(str(cv_metrics))
        # If a run has no metrics files, record it as skipped.
        if not cv_metrics.exists() and not run_summary.exists():
            skipped_files.append(f"{run_dir.name}/(no_metrics)")

    cols = ["run_dir", "metrics_file"] + METRIC_KEYS
    parsed_rows = []
    for row in rows:
        parts = row.split("\t")
        parsed_rows.append(parts + [""] * (len(cols) - len(parts)))

    print("Processed files:")
    for p in processed_files:
        print(f"  {p}")
    print("Skipped files:")
    for p in skipped_files:
        print(f"  {p}")
    print()

    df = pd.DataFrame(parsed_rows, columns=cols)
    if "run_dir" in df.columns:
        df["job_id"] = df["run_dir"].map(_extract_job_id)
        status_map = _sacct_status([j for j in df["job_id"].tolist() if j])
        df["job_state"] = df["job_id"].map(lambda j: status_map.get(j, ""))
    with pd.option_context("display.max_rows", None, "display.max_columns", None):
        print(df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
