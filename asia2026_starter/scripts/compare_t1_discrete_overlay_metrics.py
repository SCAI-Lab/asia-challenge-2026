#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from asia2026.data import load_track
from asia2026.utils import ensure_dir, make_run_id, utc_now_iso, write_json

LOG = logging.getLogger("compare_discrete_overlay")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def _load_pred(path: Path, ids: pd.Series) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        raise ValueError(f"Missing ID column in {path}")
    df = df.set_index("ID")
    missing = ids[~ids.isin(df.index)]
    if len(missing) > 0:
        raise ValueError(f"{path} missing {len(missing)} IDs from training data")
    return df.loc[ids]


def _align_predictions(
    path: Path,
    train_ids: pd.Series,
    test_ids: pd.Series,
) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)
    if "ID" not in df.columns:
        if "whatID" in df.columns:
            df = df.rename(columns={"whatID": "ID"})
        else:
            first_col = df.columns[0]
            if first_col.lower().endswith("id"):
                df = df.rename(columns={first_col: "ID"})
            else:
                raise ValueError(f"Missing ID column in {path}")
    df = df.set_index("ID")
    if train_ids.isin(df.index).all():
        return df.loc[train_ids], "train"
    if test_ids.isin(df.index).all():
        return df.loc[test_ids], "test"
    raise ValueError(
        f"{path} does not contain all train or test IDs "
        f"(train_missing={int((~train_ids.isin(df.index)).sum())}, "
        f"test_missing={int((~test_ids.isin(df.index)).sum())})"
    )


def _compute_per_target(
    y_true: pd.DataFrame,
    X_features: pd.DataFrame,
    pred: pd.DataFrame,
    target_cols: List[str],
    mask_mode: str,
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for c in target_cols:
        if c not in pred.columns:
            raise ValueError(f"Prediction missing column: {c}")
        if c not in X_features.columns:
            raise ValueError(f"Feature missing column for mask: {c}")
        if mask_mode == "imputed":
            mask = X_features[c].isna().to_numpy()
        else:
            mask = (~X_features[c].isna()).to_numpy()
        y_c = y_true[c].to_numpy()[mask]
        p_c = pred[c].to_numpy()[mask]
        out[c] = {
            "rmse": _rmse(y_c, p_c),
            "mae": _mae(y_c, p_c),
            "r2": _r2(y_c, p_c),
            "n_imputed": int(mask.sum()),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument(
        "--base-csv",
        type=str,
        default="/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter/Task1_Discrete.csv",
    )
    p.add_argument("--overlay-csv", type=str, required=True)
    p.add_argument("--min-delta", type=float, default=0.0)
    p.add_argument("--min-r2-delta", type=float, default=0.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data = load_track(1, args.data_root)
    Xtr = data.X_train.copy()
    ytr = data.y_train.copy()
    Xte = data.X_test.copy()
    target_cols = data.target_cols

    base_path = Path(args.base_csv)
    overlay_path = Path(args.overlay_csv)

    LOG.info("Loading base predictions: %s", base_path)
    base_pred, base_split = _align_predictions(base_path, Xtr["ID"], Xte["ID"])
    LOG.info("Loading overlay predictions: %s", overlay_path)
    overlay_pred, overlay_split = _align_predictions(overlay_path, Xtr["ID"], Xte["ID"])

    if base_split != overlay_split:
        raise ValueError(f"Base split ({base_split}) and overlay split ({overlay_split}) differ")

    if base_split == "train":
        X_features = Xtr
        y_true = ytr
        mask_mode = "imputed"
    else:
        LOG.warning("Predictions match test IDs; using observed-only metrics on X_test.")
        LOG.warning("Imputed-only metrics require train/OOF predictions.")
        X_features = Xte
        y_true = Xte
        mask_mode = "observed"

    LOG.info("Computing per-target metrics (mask_mode=%s)", mask_mode)
    base_metrics = _compute_per_target(y_true, X_features, base_pred, target_cols, mask_mode)
    overlay_metrics = _compute_per_target(y_true, X_features, overlay_pred, target_cols, mask_mode)

    rows = []
    improved = []
    for c in target_cols:
        b = base_metrics[c]
        o = overlay_metrics[c]
        rmse_diff = b["rmse"] - o["rmse"]
        mae_diff = b["mae"] - o["mae"]
        r2_diff = o["r2"] - b["r2"]
        rows.append(
            {
                "target": c,
                "n_imputed": b["n_imputed"],
                "rmse_base": b["rmse"],
                "rmse_overlay": o["rmse"],
                "rmse_diff_base_minus_overlay": rmse_diff,
                "mae_base": b["mae"],
                "mae_overlay": o["mae"],
                "mae_diff_base_minus_overlay": mae_diff,
                "r2_base": b["r2"],
                "r2_overlay": o["r2"],
                "r2_diff_overlay_minus_base": r2_diff,
            }
        )
        rmse_improved = np.isfinite(rmse_diff) and (rmse_diff > args.min_delta)
        mae_improved = np.isfinite(mae_diff) and (mae_diff > args.min_delta)
        r2_improved = np.isfinite(r2_diff) and (r2_diff > args.min_r2_delta)
        if rmse_improved or mae_improved or r2_improved:
            improved.append((c, rmse_diff, mae_diff, r2_diff))

    out_dir = ensure_dir(Path(args.run_root) / make_run_id(prefix="t1_compare_discrete_overlay"))
    out_csv = out_dir / "per_target_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    write_json(
        out_dir / "summary.json",
        {
            "created_utc": utc_now_iso(),
            "base_csv": str(base_path),
            "overlay_csv": str(overlay_path),
            "split": base_split,
            "mask_mode": mask_mode,
            "min_delta": args.min_delta,
            "min_r2_delta": args.min_r2_delta,
            "output_csv": str(out_csv),
        },
    )

    LOG.info("Wrote per-target metrics: %s", out_csv)
    if improved:
        LOG.info("Improved targets (diffs >= thresholds): %s", len(improved))
        for c, rmse_d, mae_d, r2_d in improved:
            LOG.info(
                "  %s: rmse_diff=%.6f mae_diff=%.6f r2_diff=%.6f",
                c,
                rmse_d,
                mae_d,
                r2_d,
            )
    else:
        LOG.info("No targets met improvement thresholds.")


if __name__ == "__main__":
    main()
