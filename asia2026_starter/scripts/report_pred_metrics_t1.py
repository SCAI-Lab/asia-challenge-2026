#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from asia2026.data import load_track


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Run directory containing predictions_test.csv")
    ap.add_argument("--data-root", required=True, help="Path to staged data root")
    ap.add_argument("--out-csv", default=None, help="Optional CSV output path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    files = [
        run_dir / "predictions_test.csv",
        run_dir / "predictions_test_calib1_row1_ctx16_greedy_t01-0.5_t12-1.5.csv",
        run_dir / "predictions_test_calib1_row1_ctx16_greedy_t01-0.5_t12-1.0.csv",
        run_dir / "predictions_test_calib1_row1_ctx16_prior_time_soft_tau-0.10.csv",
        run_dir / "predictions_test_calib1_row1_ctx16_prior_time_soft_tau-0.20.csv",
    ]

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise SystemExit(f"missing predictions files (first 5 shown): {missing[:5]} (total {len(missing)})")

    data = load_track(1, args.data_root)
    target_cols = data.target_cols
    sensory_cols = data.sensory_target_cols

    xte = data.X_test.copy()
    if "ID" not in xte.columns:
        raise SystemExit("X_test missing ID column; cannot align predictions.")
    y_true_df = xte[["ID"] + target_cols].copy()

    sensory_set = set(sensory_cols)

    def masked_r2(y_true_vals, y_pred_vals):
        if y_true_vals.size < 2:
            return 0.0
        y_mean = y_true_vals.mean()
        ss_tot = np.sum((y_true_vals - y_mean) ** 2)
        if ss_tot == 0:
            return 0.0
        ss_res = np.sum((y_true_vals - y_pred_vals) ** 2)
        return float(1.0 - ss_res / ss_tot)

    def masked_rmse(y_true_vals, y_pred_vals):
        if y_true_vals.size == 0:
            return 0.0
        return float(np.sqrt(np.mean((y_true_vals - y_pred_vals) ** 2)))

    def masked_mae(y_true_vals, y_pred_vals):
        if y_true_vals.size == 0:
            return 0.0
        return float(np.mean(np.abs(y_true_vals - y_pred_vals)))

    rows = []
    for path in files:
        pred = pd.read_csv(path)
        merged = y_true_df.merge(pred, on="ID", suffixes=("_true", "_pred"))
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

        r2_all = masked_r2(yt_all, yp_all)
        rmse_all = masked_rmse(yt_all, yp_all)
        mae_all = masked_mae(yt_all, yp_all)
        r2_s = masked_r2(yt_sens, yp_sens)
        rmse_s = masked_rmse(yt_sens, yp_sens)
        mae_s = masked_mae(yt_sens, yp_sens)

        metrics = {
            "r2_sensory": r2_s,
            "rmse_sensory": rmse_s,
            "mae_sensory": mae_s,
            "r2_all": r2_all,
            "rmse_all": rmse_all,
            "mae_all": mae_all,
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
