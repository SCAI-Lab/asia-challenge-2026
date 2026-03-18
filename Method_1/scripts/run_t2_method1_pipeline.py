#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from utils.utils import ensure_dir, make_run_id, read_json, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)

TRACK2_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = TRACK2_ROOT / "data"
DEFAULT_RUN_ROOT = TRACK2_ROOT / "runs"


def _env_with_repo_on_path() -> Dict[str, str]:
    env = os.environ.copy()
    repo_root = str(TRACK2_ROOT)
    if env.get("PYTHONPATH"):
        env["PYTHONPATH"] = repo_root + os.pathsep + env["PYTHONPATH"]
    else:
        env["PYTHONPATH"] = repo_root
    return env


def _run_script(script_name: str, extra_args: list[str]) -> None:
    script = SCRIPT_DIR / script_name
    cmd = [sys.executable, str(script), *extra_args]
    LOGGER.info("Running %s", " ".join(cmd))
    subprocess.run(cmd, cwd=TRACK2_ROOT, env=_env_with_repo_on_path(), check=True)


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

    pipeline_id = make_run_id(prefix="t2_method1_pipeline")
    pipeline_root = ensure_dir(Path(args.run_root) / pipeline_id)

    step_roots = {
        "base": ensure_dir(pipeline_root / "00_discrete_bag"),
        "hedge": ensure_dir(pipeline_root / "01_pairwise_shrink"),
        "anchor": ensure_dir(pipeline_root / "02_anchor_correction"),
        "extend": ensure_dir(pipeline_root / "03_extend_obs_anchor"),
    }

    data_root = Path(args.data_root)
    features_test = data_root / "features_test_2.csv"
    features_train = data_root / "features_train_2.csv"
    labels_train = data_root / "labels_train_2.csv"

    _run_script(
        "run_tabpfn_t2_discrete_bag5.py",
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
        "run_t2_hedge_pairwise_shrink.py",
        [
            "--base-csv",
            str(base_csv),
            "--data-root",
            str(data_root),
            "--run-root",
            str(step_roots["hedge"]),
        ],
    )
    hedge_run = _find_run_dir(step_roots["hedge"])
    hedge_csv = _submission_csv(hedge_run)

    _run_script(
        "run_t2_anchor_correction.py",
        [
            "--base-cv",
            str(hedge_csv),
            "--features-test",
            str(features_test),
            "--labels-train",
            str(labels_train),
            "--run-root",
            str(step_roots["anchor"]),
        ],
    )
    anchor_run = _find_run_dir(step_roots["anchor"])
    anchor_csv = _submission_csv(anchor_run)

    _run_script(
        "run_t2_extend_obs_anchor.py",
        [
            "--base-cv",
            str(anchor_csv),
            "--features-test",
            str(features_test),
            "--features-train",
            str(features_train),
            "--labels-train",
            str(labels_train),
            "--run-root",
            str(step_roots["extend"]),
        ],
    )
    extend_run = _find_run_dir(step_roots["extend"])
    final_csv = _submission_csv(extend_run)

    pipeline_csv = pipeline_root / "predictions_test.csv"
    shutil.copy2(final_csv, pipeline_csv)

    summary = {
        "method": "t2_method1_pipeline",
        "created_at": utc_now_iso(),
        "pipeline_root": str(pipeline_root),
        "data_root": str(data_root),
        "limit_rows": args.limit_rows,
        "stage_dirs": {
            "base": str(step_roots["base"]),
            "hedge": str(step_roots["hedge"]),
            "anchor": str(step_roots["anchor"]),
            "extend": str(step_roots["extend"]),
        },
        "steps": {
            "base": {
                "run_dir": str(base_run),
                "submission_csv": str(base_csv),
            },
            "hedge": {
                "run_dir": str(hedge_run),
                "submission_csv": str(hedge_csv),
            },
            "anchor": {
                "run_dir": str(anchor_run),
                "submission_csv": str(anchor_csv),
            },
            "extend": {
                "run_dir": str(extend_run),
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
