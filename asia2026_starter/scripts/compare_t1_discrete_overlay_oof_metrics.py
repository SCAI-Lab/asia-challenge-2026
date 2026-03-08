#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from asia2026.data import load_track
from asia2026.eval import compute_imputed_only_metrics, load_oof_npz
from asia2026.utils import ensure_dir, make_run_id, utc_now_iso, write_json

LOG = logging.getLogger("compare_discrete_overlay_oof")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom == 0.0:
        return float("nan")
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def _compute_per_target(
    y_true: pd.DataFrame,
    X_features: pd.DataFrame,
    pred: np.ndarray,
    target_cols: List[str],
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for j, c in enumerate(target_cols):
        mask = X_features[c].isna().to_numpy()
        y_c = y_true[c].to_numpy()[mask]
        p_c = pred[:, j][mask]
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
    p.add_argument("--base-oof-npz", type=str, required=True)
    p.add_argument("--overlay-oof-npz", type=str, required=True)
    p.add_argument("--min-rmse-delta", type=float, default=0.0)
    p.add_argument("--min-mae-delta", type=float, default=0.0)
    p.add_argument("--min-r2-delta", type=float, default=0.0)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    data = load_track(1, args.data_root)
    Xtr = data.X_train.copy()
    ytr = data.y_train.copy()
    target_cols = data.target_cols
    sensory_cols = data.sensory_target_cols

    base_ids, base_cols, base_pred = load_oof_npz(Path(args.base_oof_npz))
    over_ids, over_cols, over_pred = load_oof_npz(Path(args.overlay_oof_npz))

    if list(base_cols) != list(target_cols):
        raise ValueError("Base OOF target columns do not match track target_cols")
    if list(over_cols) != list(target_cols):
        raise ValueError("Overlay OOF target columns do not match track target_cols")
    if not np.array_equal(base_ids, Xtr["ID"].to_numpy()):
        raise ValueError("Base OOF IDs do not match X_train IDs")
    if not np.array_equal(over_ids, Xtr["ID"].to_numpy()):
        raise ValueError("Overlay OOF IDs do not match X_train IDs")

    LOG.info("Computing per-target imputed-only metrics on train")
    base_metrics = _compute_per_target(ytr, Xtr, base_pred, target_cols)
    over_metrics = _compute_per_target(ytr, Xtr, over_pred, target_cols)

    rows = []
    improved = []
    for c in target_cols:
        b = base_metrics[c]
        o = over_metrics[c]
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
        rmse_improved = np.isfinite(rmse_diff) and (rmse_diff > args.min_rmse_delta)
        mae_improved = np.isfinite(mae_diff) and (mae_diff > args.min_mae_delta)
        r2_improved = np.isfinite(r2_diff) and (r2_diff > args.min_r2_delta)
        if rmse_improved and mae_improved and r2_improved:
            improved.append((c, rmse_diff, mae_diff, r2_diff))

    out_dir = ensure_dir(Path(args.run_root) / make_run_id(prefix="t1_compare_discrete_overlay_oof"))
    out_csv = out_dir / "per_target_metrics.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    base_overall = compute_imputed_only_metrics(ytr, base_pred, Xtr, target_cols, sensory_cols)
    over_overall = compute_imputed_only_metrics(ytr, over_pred, Xtr, target_cols, sensory_cols)

    write_json(
        out_dir / "summary.json",
        {
            "created_utc": utc_now_iso(),
            "base_oof_npz": str(Path(args.base_oof_npz)),
            "overlay_oof_npz": str(Path(args.overlay_oof_npz)),
            "min_rmse_delta": args.min_rmse_delta,
            "min_mae_delta": args.min_mae_delta,
            "min_r2_delta": args.min_r2_delta,
            "output_csv": str(out_csv),
            "overall_base": base_overall.to_dict(),
            "overall_overlay": over_overall.to_dict(),
        },
    )

    LOG.info("Wrote per-target metrics: %s", out_csv)
    if improved:
        LOG.info("Improved targets (diffs > thresholds): %s", len(improved))
        for c, rmse_d, mae_d, r2_d in improved:
            LOG.info("  %s: rmse_diff=%.6f mae_diff=%.6f r2_diff=%.6f", c, rmse_d, mae_d, r2_d)
    else:
        LOG.info("No targets met improvement thresholds.")


if __name__ == "__main__":
    main()
