#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from asia2026.data import load_track

ID_COL = "ID"


def _parse_pred_list(values: List[str]) -> List[Path]:
    out: List[Path] = []
    for val in values:
        for part in val.split(","):
            part = part.strip()
            if part:
                out.append(Path(part))
    return out


def _masked_r2(y_true_vals: np.ndarray, y_pred_vals: np.ndarray) -> float:
    if y_true_vals.size < 2:
        return 0.0
    y_mean = y_true_vals.mean()
    ss_tot = np.sum((y_true_vals - y_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    ss_res = np.sum((y_true_vals - y_pred_vals) ** 2)
    return float(1.0 - ss_res / ss_tot)


def _masked_rmse(y_true_vals: np.ndarray, y_pred_vals: np.ndarray) -> float:
    if y_true_vals.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true_vals - y_pred_vals) ** 2)))


def _masked_mae(y_true_vals: np.ndarray, y_pred_vals: np.ndarray) -> float:
    if y_true_vals.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true_vals - y_pred_vals)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-csv", action="append", required=True, help="CSV path(s); repeat or comma-separate")
    ap.add_argument("--data-root", required=True, help="Path to staged data root")
    ap.add_argument("--track", type=int, default=1, help="Track number (default 1)")
    ap.add_argument("--out-csv", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    pred_paths = _parse_pred_list(args.pred_csv)
    if not pred_paths:
        raise SystemExit("No prediction files provided")

    data = load_track(args.track, args.data_root)
    target_cols = data.target_cols
    sensory_cols = data.sensory_target_cols
    sensory_set = set(sensory_cols)

    xte = data.X_test.copy()
    if ID_COL not in xte.columns:
        raise SystemExit("X_test missing ID column; cannot align predictions.")
    y_true_df = xte[[ID_COL] + target_cols].copy()

    rows = []
    for path in pred_paths:
        if not path.exists():
            raise SystemExit(f"pred_csv not found: {path}")
        pred = pd.read_csv(path)
        if ID_COL not in pred.columns:
            raise SystemExit(f"pred_csv missing {ID_COL}: {path}")

        merged = y_true_df.merge(pred, on=ID_COL, suffixes=("_true", "_pred"))
        y_true_mat = merged[[f"{c}_true" for c in target_cols]]
        y_pred_mat = merged[[f"{c}_pred" for c in target_cols]]

        yt_all = []
        yp_all = []
        yt_sens = []
        yp_sens = []

        for c in target_cols:
            true_vals = y_true_mat[f"{c}_true"].to_numpy(dtype=np.float32)
            pred_vals = y_pred_mat[f"{c}_pred"].to_numpy(dtype=np.float32)
            mask = ~np.isnan(true_vals)
            if mask.any():
                yt_all.append(true_vals[mask])
                yp_all.append(pred_vals[mask])
                if c in sensory_set:
                    yt_sens.append(true_vals[mask])
                    yp_sens.append(pred_vals[mask])

        yt_all = np.concatenate(yt_all) if yt_all else np.array([], dtype=np.float32)
        yp_all = np.concatenate(yp_all) if yp_all else np.array([], dtype=np.float32)
        yt_sens = np.concatenate(yt_sens) if yt_sens else np.array([], dtype=np.float32)
        yp_sens = np.concatenate(yp_sens) if yp_sens else np.array([], dtype=np.float32)

        r2_all = _masked_r2(yt_all, yp_all)
        rmse_all = _masked_rmse(yt_all, yp_all)
        mae_all = _masked_mae(yt_all, yp_all)
        r2_s = _masked_r2(yt_sens, yp_sens)
        rmse_s = _masked_rmse(yt_sens, yp_sens)
        mae_s = _masked_mae(yt_sens, yp_sens)

        metrics = {
            "r2_sensory": r2_s,
            "rmse_sensory": rmse_s,
            "mae_sensory": mae_s,
            "r2_all": r2_all,
            "rmse_all": rmse_all,
            "mae_all": mae_all,
            # We do not have ground-truth for missing test cells; mirror observed metrics.
            "r2_sensory_imputed_only": r2_s,
            "rmse_sensory_imputed_only": rmse_s,
            "mae_sensory_imputed_only": mae_s,
            "r2_all_imputed_only": r2_all,
            "rmse_all_imputed_only": rmse_all,
            "mae_all_imputed_only": mae_all,
            "rmse_imputed_only": rmse_all,
        }

        rows.append({"file": str(path), **metrics})

    df = pd.DataFrame(rows)
    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Wrote: {out}")

    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
