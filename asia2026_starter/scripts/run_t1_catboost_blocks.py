#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor
from tqdm import tqdm

from asia2026.data import load_track
from asia2026.eval import (
    compute_imputed_only_breakdown,
    compute_imputed_only_metrics,
    make_time_stratified_folds,
    save_oof_npz,
)
from asia2026.metrics_weighted import compute_wrmse_imputed_only
from asia2026.utils import ensure_dir, make_run_id, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)


def _block_cols(target_cols: List[str], suffix: str) -> List[str]:
    return [c for c in target_cols if c.endswith(suffix)]


def _apply_copy_through(pred: np.ndarray, features: pd.DataFrame, target_cols: List[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    for j, c in enumerate(target_cols):
        if c in features.columns:
            obs = features[c].to_numpy()
            m = ~pd.isna(obs)
            pred[m, j] = obs[m]
    return pred


def _clip_predictions(pred: np.ndarray, target_cols: List[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    for j, c in enumerate(target_cols):
        if c == "anyana":
            pred[:, j] = np.clip(pred[:, j], 0.0, 1.0)
        else:
            pred[:, j] = np.clip(pred[:, j], 0.0, 2.0)
    return pred


def _write_submission(sample_sub: pd.DataFrame, ids: pd.Series, pred: np.ndarray, target_cols: List[str], out_csv: Path) -> None:
    sub = sample_sub.copy()
    sub["ID"] = ids.values
    for j, c in enumerate(target_cols):
        sub[c] = pred[:, j]
    ensure_dir(out_csv.parent)
    sub.to_csv(out_csv, index=False)


def _build_features(data) -> tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    cb_cols = list(dict.fromkeys(data.motor_cols + data.meta_cols + ["time"]))
    Xtr_cb = data.X_train[cb_cols].copy()
    Xte_cb = data.X_test[cb_cols].copy()
    cat_cols = [
        c
        for c in cb_cols
        if pd.api.types.is_object_dtype(Xtr_cb[c]) or pd.api.types.is_categorical_dtype(Xtr_cb[c])
    ]
    if cat_cols:
        for c in cat_cols:
            Xtr_cb[c] = Xtr_cb[c].fillna("MISSING").astype(str)
            Xte_cb[c] = Xte_cb[c].fillna("MISSING").astype(str)
    return Xtr_cb, Xte_cb, cb_cols


def _build_params(device: str) -> Dict[str, object]:
    params: Dict[str, object] = {
        "loss_function": "MultiRMSE",
        "iterations": 4000,
        "learning_rate": 0.03,
        "depth": 8,
        "l2_leaf_reg": 3,
        "random_strength": 1.0,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "eval_metric": "MultiRMSE",
        "od_type": "Iter",
        "od_wait": 200,
        "verbose": False,
    }
    if device == "gpu":
        params["task_type"] = "GPU"
        params["devices"] = "0"
        params["boosting_type"] = "Plain"
    else:
        params["task_type"] = "CPU"
    return params


def _gpu_available() -> bool:
    try:
        from catboost.utils import get_gpu_device_count

        return int(get_gpu_device_count()) > 0
    except Exception:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible is not None and cuda_visible.strip() == "":
            return False
        return True


def _predict_blocks(
    Xtr_cb: pd.DataFrame,
    ytr: pd.DataFrame,
    Xte_cb: pd.DataFrame,
    Xte_features: pd.DataFrame,
    target_cols: List[str],
    blocks: Dict[str, List[str]],
    cat_idx: List[int],
    params: Dict[str, object],
    log_prefix: str,
) -> np.ndarray:
    pred = np.zeros((len(Xte_cb), len(target_cols)), dtype=np.float32)

    for name, cols in tqdm(blocks.items(), desc=f"Blocks{log_prefix}"):
        LOGGER.info("%sTraining block %s (%d targets).", log_prefix, name, len(cols))
        model = CatBoostRegressor(**params)
        y_block = ytr[cols]
        model.fit(Xtr_cb, y_block, cat_features=cat_idx)
        block_pred = model.predict(Xte_cb)
        if block_pred.ndim == 1:
            block_pred = block_pred.reshape(-1, 1)
        for col_idx, col in enumerate(cols):
            out_idx = target_cols.index(col)
            pred[:, out_idx] = block_pred[:, col_idx]
        LOGGER.info("%sFinished block %s.", log_prefix, name)

    if "anyana" in target_cols:
        anyana_idx = target_cols.index("anyana")
        pred[:, anyana_idx] = float(ytr["anyana"].mean())
        LOGGER.info("%sFilled anyana with train mean.", log_prefix)

    pred = _clip_predictions(pred, target_cols)
    pred = _apply_copy_through(pred, Xte_features, target_cols)
    return pred


def run_one(data_root: str, run_root: str, device: str, seed: int, n_splits: int) -> Path:
    if device not in {"gpu", "cpu"}:
        raise ValueError("device must be 'gpu' or 'cpu'")
    if device == "gpu":
        if not _gpu_available():
            raise RuntimeError("CUDA is required for GPU mode. No GPU detected.")

    track = 1
    data = load_track(track, data_root)

    Xtr_cb, Xte_cb, cb_cols = _build_features(data)
    target_cols = data.target_cols

    blocks: Dict[str, List[str]] = {
        "lt_l": _block_cols(target_cols, "ltl"),
        "lt_r": _block_cols(target_cols, "ltr"),
        "pp_l": _block_cols(target_cols, "ppl"),
        "pp_r": _block_cols(target_cols, "ppr"),
    }

    for name, cols in blocks.items():
        if not cols:
            raise ValueError(f"Missing target columns for block: {name}")

    remaining = set(target_cols) - {"anyana"}
    for cols in blocks.values():
        remaining -= set(cols)
    if remaining:
        raise ValueError(f"Unexpected target columns not covered by blocks: {sorted(remaining)}")

    method = f"t1_catboost_blocks_{device}"
    run_id = make_run_id(prefix=method)
    out_dir = ensure_dir(Path(run_root) / run_id)

    cat_cols = [c for c in cb_cols if Xtr_cb[c].dtype == object]
    cat_idx = [cb_cols.index(c) for c in cat_cols]

    LOGGER.info("Device mode: %s", device)
    LOGGER.info("Run output: %s", out_dir)
    LOGGER.info("Built CatBoost features: %d columns.", len(cb_cols))
    LOGGER.info("Targets per block: %s", {k: len(v) for k, v in blocks.items()})

    params = _build_params(device)

    folds = make_time_stratified_folds(data.X_train["time"].to_numpy(), n_splits=n_splits, seed=seed)
    oof = np.zeros((len(data.X_train), len(target_cols)), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc="Folds")):
        LOGGER.info("Fold %d: train=%d val=%d", fold, len(tr_idx), len(va_idx))
        Xtr_f = Xtr_cb.iloc[tr_idx].reset_index(drop=True)
        ytr_f = data.y_train.iloc[tr_idx].reset_index(drop=True)
        Xva_f = Xtr_cb.iloc[va_idx].reset_index(drop=True)
        Xva_feat = data.X_train.iloc[va_idx].reset_index(drop=True)

        pred_va = _predict_blocks(
            Xtr_f,
            ytr_f,
            Xva_f,
            Xva_feat,
            target_cols,
            blocks,
            cat_idx,
            params,
            log_prefix=f" (fold={fold}) ",
        )
        oof[va_idx] = pred_va

    oof_path = out_dir / "oof_predictions_train.npz"
    save_oof_npz(oof_path, ids=data.X_train["ID"].to_numpy(), target_cols=target_cols, preds=oof)

    metrics = compute_imputed_only_metrics(data.y_train, oof, data.X_train, target_cols, data.sensory_target_cols)
    breakdown = compute_imputed_only_breakdown(data.y_train, oof, data.X_train, target_cols, data.sensory_target_cols)
    wrep = compute_wrmse_imputed_only(
        y_true=data.y_train,
        oof_pred=oof,
        X_train_features=data.X_train,
        X_test_features=data.X_test,
        target_cols=target_cols,
    )
    LOGGER.info("OOF imputed-only RMSE (sensory): %.6f", metrics.rmse_sensory_imputed_only)
    LOGGER.info("OOF imputed-only RMSE (all): %.6f", metrics.rmse_all_imputed_only)
    LOGGER.info("OOF WRMSE (imputed-only): %.6f", wrep.wrmse_imputed_only)
    LOGGER.info("OOF WRMSE (all): %.6f", wrep.wrmse_all)

    pred_test = _predict_blocks(
        Xtr_cb,
        data.y_train,
        Xte_cb,
        data.X_test,
        target_cols,
        blocks,
        cat_idx,
        params,
        log_prefix=" (test) ",
    )

    _write_submission(
        data.sample_submission,
        ids=data.X_test["ID"],
        pred=pred_test,
        target_cols=target_cols,
        out_csv=out_dir / "predictions_test.csv",
    )

    write_json(
        out_dir / "run_summary.json",
        {
            "run_id": run_id,
            "finished_utc": utc_now_iso(),
            "track": track,
            "method": method,
            "params": params,
            "feature_cols": cb_cols,
            "blocks": blocks,
            "n_splits": n_splits,
            "metrics": metrics.to_dict(),
            "metrics_breakdown": breakdown,
            "weighted_metrics": {
                "wrmse_imputed_only": wrep.wrmse_imputed_only,
                "wrmse_all": wrep.wrmse_all,
            },
            "artifacts": {
                "submission_csv": "predictions_test.csv",
                "oof_npz": "oof_predictions_train.npz",
                "weighted_oof": "weighted_oof.json",
            },
        },
    )
    write_json(
        out_dir / "weighted_oof.json",
        {
            "wrmse_imputed_only": wrep.wrmse_imputed_only,
            "wrmse_all": wrep.wrmse_all,
            "per_target_contrib": wrep.per_target_contrib,
            "weights": wrep.weights,
            "mse_per_target": wrep.mse_per_target,
            "mse_per_target_all": wrep.mse_per_target_all,
        },
    )
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument("--device", type=str, default="gpu", help="gpu or cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        device=args.device,
        seed=args.seed,
        n_splits=args.n_splits,
    )
    print(f"[asia2026 t1 catboost blocks] done -> {out_dir}")


if __name__ == "__main__":
    main()
