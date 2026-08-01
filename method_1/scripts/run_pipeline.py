#!/usr/bin/env python3
# ruff: noqa: E402
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from utils.runtime import ensure_dir, make_run_id, read_json, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "runs"


def _env_with_repo_on_path() -> Dict[str, str]:
    env = os.environ.copy()
    repo_root = str(REPOSITORY_ROOT)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = repo_root + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = repo_root
    return env


def _run_script(script_name: str, extra_args: list[str]) -> None:
    script = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script), *extra_args]
    LOGGER.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, cwd=REPOSITORY_ROOT, env=_env_with_repo_on_path(), check=True)


def _find_run_dir(step_root: Path) -> Path:
    run_dirs = sorted((p for p in step_root.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise RuntimeError(f"No run directory created under {step_root}")
    return run_dirs[-1]


def _submission_csv(run_dir: Path) -> Path:
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = read_json(summary_path)
        submission_csv = summary.get("submission_csv")
        if submission_csv:
            out_csv = Path(submission_csv)
            if out_csv.exists():
                return out_csv
            raise FileNotFoundError(f"run_summary.json points to missing CSV: {out_csv}")
    out_csv = run_dir / "predictions_test.csv"
    if not out_csv.exists():
        raise FileNotFoundError(f"Missing output CSV: {out_csv}")
    return out_csv


def _maybe_append_limit_rows(cmd: list[str], limit_rows: Optional[int]) -> list[str]:
    if limit_rows is None:
        return cmd
    return [*cmd, "--limit-rows", str(limit_rows)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    ap.add_argument("--limit-rows", type=int, default=None, help="Optional training-row cap for the base model.")
    args = ap.parse_args()

    pipeline_id = make_run_id(prefix="method_1_pipeline")
    pipeline_root = ensure_dir(Path(args.run_root) / pipeline_id)

    step_roots = {
        "base": ensure_dir(pipeline_root / "00_base_model"),
        "pairwise_shrinkage": ensure_dir(pipeline_root / "01_pairwise_shrinkage"),
        "anchor_correction": ensure_dir(pipeline_root / "02_anchor_correction"),
        "extended_anchor_correction": ensure_dir(pipeline_root / "03_extended_anchor_correction"),
    }

    data_root = Path(args.data_root)
    features_test = data_root / "features_test_2.csv"
    features_train = data_root / "features_train_2.csv"
    labels_train = data_root / "labels_train_2.csv"

    _run_script(
        "train_cross_validated_tabpfn.py",
        _maybe_append_limit_rows(
            [
                "--data-root",
                str(data_root),
                "--run-root",
                str(step_roots["base"]),
            ],
            args.limit_rows,
        ),
    )
    base_run = _find_run_dir(step_roots["base"])
    base_csv = _submission_csv(base_run)

    _run_script(
        "apply_pairwise_shrinkage.py",
        [
            "--base-csv",
            str(base_csv),
            "--data-root",
            str(data_root),
            "--run-root",
            str(step_roots["pairwise_shrinkage"]),
        ],
    )
    pairwise_shrinkage_run = _find_run_dir(step_roots["pairwise_shrinkage"])
    pairwise_shrinkage_csv = _submission_csv(pairwise_shrinkage_run)

    _run_script(
        "apply_anchor_correction.py",
        [
            "--base-csv",
            str(pairwise_shrinkage_csv),
            "--features-test",
            str(features_test),
            "--labels-train",
            str(labels_train),
            "--run-root",
            str(step_roots["anchor_correction"]),
        ],
    )
    anchor_correction_run = _find_run_dir(step_roots["anchor_correction"])
    anchor_correction_csv = _submission_csv(anchor_correction_run)

    _run_script(
        "apply_extended_anchor_correction.py",
        [
            "--base-csv",
            str(anchor_correction_csv),
            "--features-test",
            str(features_test),
            "--features-train",
            str(features_train),
            "--labels-train",
            str(labels_train),
            "--run-root",
            str(step_roots["extended_anchor_correction"]),
        ],
    )
    extended_anchor_correction_run = _find_run_dir(step_roots["extended_anchor_correction"])
    final_csv = _submission_csv(extended_anchor_correction_run)

    pipeline_csv = pipeline_root / "predictions_test.csv"
    shutil.copy2(final_csv, pipeline_csv)

    summary = {
        "method": "method_1_pipeline",
        "created_at": utc_now_iso(),
        "pipeline_root": str(pipeline_root),
        "data_root": str(data_root),
        "limit_rows": args.limit_rows,
        "stage_dirs": {
            "base": str(step_roots["base"]),
            "pairwise_shrinkage": str(step_roots["pairwise_shrinkage"]),
            "anchor_correction": str(step_roots["anchor_correction"]),
            "extended_anchor_correction": str(step_roots["extended_anchor_correction"]),
        },
        "steps": {
            "base": {
                "run_dir": str(base_run),
                "submission_csv": str(base_csv),
            },
            "pairwise_shrinkage": {
                "run_dir": str(pairwise_shrinkage_run),
                "submission_csv": str(pairwise_shrinkage_csv),
            },
            "anchor_correction": {
                "run_dir": str(anchor_correction_run),
                "submission_csv": str(anchor_correction_csv),
            },
            "extended_anchor_correction": {
                "run_dir": str(extended_anchor_correction_run),
                "submission_csv": str(final_csv),
            },
        },
        "final_submission_csv": str(pipeline_csv),
    }
    write_json(pipeline_root / "pipeline_summary.json", summary)
    LOGGER.info("Method 1 pipeline complete: %s", pipeline_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
