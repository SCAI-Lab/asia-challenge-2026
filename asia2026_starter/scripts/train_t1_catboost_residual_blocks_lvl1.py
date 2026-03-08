#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from asia2026.data import load_track
from asia2026.eval import (
    compute_imputed_only_breakdown,
    compute_imputed_only_metrics,
    load_oof_npz,
    make_time_stratified_folds,
    save_oof_npz,
)
from asia2026.metrics_weighted import compute_wrmse_imputed_only
from asia2026.utils import ensure_dir, make_run_id, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)


def _block_cols(target_cols: List[str], suffix: str) -> List[str]:
    return [c for c in target_cols if c.endswith(suffix)]


def _select_boost_targets(
    *,
    target_cols: List[str],
    sensory_cols: List[str],
    X_ref: pd.DataFrame,
    topk: int,
) -> List[str]:
    if topk <= 0 or topk >= len(sensory_cols):
        return [c for c in target_cols if c in sensory_cols]
    miss = X_ref[sensory_cols].isna().mean().sort_values(ascending=False)
    top = miss.head(topk).index.tolist()
    return [c for c in target_cols if c in top]


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


def _weighted_rmse_imputed_only(
    y_true_df: pd.DataFrame,
    y_pred: np.ndarray,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    target_cols: List[str],
) -> float:
    weights = []
    rmses = []
    for j, col in enumerate(target_cols):
        if col not in X_train.columns or col not in X_test.columns:
            continue
        mask = X_train[col].isna().to_numpy()
        if not mask.any():
            continue
        y_true = y_true_df[col].to_numpy(dtype=np.float32)
        y_pred_col = y_pred[:, j]
        rmse = float(np.sqrt(np.mean((y_true[mask] - y_pred_col[mask]) ** 2)))
        weight = float(X_test[col].isna().mean())
        weights.append(weight)
        rmses.append(rmse)
    weight_sum = float(np.sum(weights)) if weights else 0.0
    if weight_sum == 0.0:
        return 0.0
    return float(np.sum(np.array(weights) * np.array(rmses)) / weight_sum)


def _resolve_device(device: str) -> str:
    if device in {"cpu", "gpu"}:
        return device
    if device != "auto":
        raise ValueError("device must be cpu, gpu, or auto")

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is not None and cuda_visible.strip() == "":
        return "cpu"

    try:
        from catboost.utils import get_gpu_device_count

        if int(get_gpu_device_count()) > 0:
            return "gpu"
    except Exception:
        pass

    return "gpu" if cuda_visible is None or cuda_visible.strip() else "cpu"


def _build_params(device: str, args: argparse.Namespace) -> Dict[str, object]:
    params: Dict[str, object] = {
        "loss_function": "MultiRMSE",
        "eval_metric": "MultiRMSE",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_strength": float(args.random_strength),
        "bootstrap_type": "Bayesian",
        "bagging_temperature": float(args.bagging_temperature),
        "od_type": "Iter",
        "od_wait": int(args.od_wait),
        "random_seed": int(args.seed),
        "verbose": False,
    }
    if device == "gpu":
        params["task_type"] = "GPU"
        params["devices"] = "0"
        params["boosting_type"] = "Plain"
    else:
        params["task_type"] = "CPU"
        params["thread_count"] = int(args.thread_count)
    return params


def _load_oof_aligned(oof_path: Path, train_ids: np.ndarray, target_cols: List[str]) -> np.ndarray:
    oof_ids, oof_cols, oof_preds = load_oof_npz(oof_path)
    oof_cols = list(oof_cols)

    missing = [c for c in target_cols if c not in oof_cols]
    if missing:
        raise ValueError(f"OOF missing target columns: {missing}")

    col_idx = [oof_cols.index(c) for c in target_cols]
    oof_preds = oof_preds[:, col_idx]

    id_map = {str(i): idx for idx, i in enumerate(oof_ids)}
    indices = []
    for i in train_ids:
        key = str(i)
        if key not in id_map:
            raise ValueError(f"Train ID {key} missing from OOF predictions")
        indices.append(id_map[key])

    aligned = oof_preds[np.array(indices, dtype=int)]
    return aligned


def _prepare_base_features(
    df: pd.DataFrame,
    base_cols: List[str],
    fill_values: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, float]]:
    X_base = df[base_cols].copy()
    cat_cols = [
        c
        for c in base_cols
        if pd.api.types.is_object_dtype(X_base[c]) or pd.api.types.is_categorical_dtype(X_base[c])
    ]
    for c in cat_cols:
        X_base[c] = X_base[c].fillna("MISSING").astype(str)
    num_cols = [c for c in base_cols if c not in cat_cols]
    if fill_values is None:
        fill_values = {}
        for c in num_cols:
            if pd.api.types.is_numeric_dtype(X_base[c]):
                med = X_base[c].median()
                if pd.isna(med):
                    med = 0.0
                fill_values[c] = float(med)
    if fill_values and num_cols:
        X_base[num_cols] = X_base[num_cols].fillna(value=fill_values)
    return X_base, cat_cols, fill_values


def _build_block_features(
    X_base: pd.DataFrame,
    X_features: pd.DataFrame,
    oof_block: np.ndarray,
    block_cols: List[str],
    add_missing: bool,
) -> pd.DataFrame:
    X = X_base.copy()
    for j, col in enumerate(block_cols):
        X[f"oof__{col}"] = oof_block[:, j]
    if add_missing:
        for col in block_cols:
            if col in X_features.columns:
                X[f"miss__{col}"] = X_features[col].isna().astype(np.int8)
    return X


def run_one(
    data_root: str,
    run_root: str,
    oof_path: Path,
    device: str,
    seed: int,
    do_cv: bool,
    n_splits: int,
    eval_lambdas: List[float],
    limit_rows: Optional[int],
    add_missing_indicators: bool,
    topk_targets: int,
    topk_source: str,
    args: argparse.Namespace,
) -> Path:
    track = 1
    data = load_track(track, data_root)

    Xtr = data.X_train.copy()
    ytr = data.y_train.copy()

    oof_preds = _load_oof_aligned(oof_path, Xtr["ID"].to_numpy(), data.target_cols)

    if limit_rows is not None:
        Xtr = Xtr.iloc[:limit_rows].reset_index(drop=True)
        ytr = ytr.iloc[:limit_rows].reset_index(drop=True)
        oof_preds = oof_preds[:limit_rows]

    target_cols = data.target_cols
    X_ref = data.X_test if topk_source == "test" else data.X_train
    boost_targets = _select_boost_targets(
        target_cols=target_cols,
        sensory_cols=data.sensory_target_cols,
        X_ref=X_ref,
        topk=topk_targets,
    )
    boost_set = set(boost_targets)

    blocks: Dict[str, List[str]] = {
        "lt_l": [c for c in target_cols if c.endswith("ltl") and c in boost_set],
        "lt_r": [c for c in target_cols if c.endswith("ltr") and c in boost_set],
        "pp_l": [c for c in target_cols if c.endswith("ppl") and c in boost_set],
        "pp_r": [c for c in target_cols if c.endswith("ppr") and c in boost_set],
    }
    skipped_blocks = [name for name, cols in blocks.items() if not cols]
    blocks = {name: cols for name, cols in blocks.items() if cols}
    if not blocks:
        LOGGER.warning("No boost targets selected; residual models will be skipped.")

    method = f"t1_catboost_residual_blocks_{device}"
    run_id = make_run_id(prefix=method)
    out_dir = ensure_dir(Path(run_root) / run_id)
    model_dir = ensure_dir(out_dir / "models")

    base_cols = list(dict.fromkeys(data.motor_cols + data.meta_cols + ["time"]))
    X_base, cat_cols, num_fill = _prepare_base_features(Xtr, base_cols)

    params = _build_params(device, args)

    baseline_metrics = compute_imputed_only_metrics(
        ytr,
        oof_preds,
        Xtr,
        target_cols,
        data.sensory_target_cols,
    ).to_dict()
    baseline_breakdown = compute_imputed_only_breakdown(
        ytr,
        oof_preds,
        Xtr,
        target_cols,
        data.sensory_target_cols,
    )
    baseline_wrmse = compute_wrmse_imputed_only(
        y_true=ytr,
        oof_pred=oof_preds,
        X_train_features=Xtr,
        X_test_features=data.X_test,
        target_cols=target_cols,
    )
    baseline_metrics["wrmse_imputed_only"] = baseline_wrmse.wrmse_imputed_only
    baseline_metrics["wrmse_all"] = baseline_wrmse.wrmse_all

    feature_columns: Dict[str, List[str]] = {}
    model_paths: Dict[str, str] = {}

    cv_metrics: Dict[str, Dict[str, float]] = {}
    if do_cv:
        if len(Xtr) < n_splits:
            LOGGER.warning("Too few rows for %d splits; reducing to %d.", n_splits, len(Xtr))
            n_splits = len(Xtr)
        if n_splits < 2:
            LOGGER.warning("Skipping CV because n_splits < 2.")
            do_cv = False

    oof_residual = np.zeros_like(oof_preds, dtype=np.float32)
    if do_cv:
        folds = make_time_stratified_folds(Xtr["time"].to_numpy(), n_splits=n_splits, seed=seed)
        for fold, (tr_idx, va_idx) in enumerate(folds):
            LOGGER.info("CV fold %d: train=%d val=%d", fold, len(tr_idx), len(va_idx))
            Xtr_base_f = X_base.iloc[tr_idx].reset_index(drop=True)
            Xva_base_f = X_base.iloc[va_idx].reset_index(drop=True)
            Xtr_feat_f = Xtr.iloc[tr_idx].reset_index(drop=True)
            Xva_feat_f = Xtr.iloc[va_idx].reset_index(drop=True)
            ytr_f = ytr.iloc[tr_idx].reset_index(drop=True)

            for name, cols in blocks.items():
                block_idx = [target_cols.index(c) for c in cols]
                oof_block_tr = oof_preds[tr_idx][:, block_idx]
                oof_block_va = oof_preds[va_idx][:, block_idx]
                y_block_tr = ytr_f[cols].to_numpy(dtype=np.float32)
                residual_tr = y_block_tr - oof_block_tr

                Xtr_block = _build_block_features(
                    Xtr_base_f, Xtr_feat_f, oof_block_tr, cols, add_missing_indicators
                )
                Xva_block = _build_block_features(
                    Xva_base_f, Xva_feat_f, oof_block_va, cols, add_missing_indicators
                )
                cat_idx = [Xtr_block.columns.get_loc(c) for c in cat_cols]

                model = CatBoostRegressor(**params)
                model.fit(Xtr_block, residual_tr, cat_features=cat_idx)
                pred_va = model.predict(Xva_block)
                if pred_va.ndim == 1:
                    pred_va = pred_va.reshape(-1, 1)
                for j, col in enumerate(cols):
                    out_idx = target_cols.index(col)
                    miss_mask = Xva_feat_f[col].isna().to_numpy()
                    oof_residual[va_idx, out_idx] = pred_va[:, j] * miss_mask

        save_oof_npz(
            out_dir / "oof_residual_predictions_train.npz",
            ids=Xtr["ID"].to_numpy(),
            target_cols=target_cols,
            preds=oof_residual,
        )

        oof_base = oof_preds.copy()
        oof_base = np.nan_to_num(oof_base, nan=0.0)
        oof_base = _clip_predictions(oof_base, target_cols)
        oof_base = _apply_copy_through(oof_base, Xtr, target_cols)
        baseline_weighted_rmse = _weighted_rmse_imputed_only(
            ytr,
            oof_base,
            Xtr,
            data.X_test,
            target_cols,
        )

        for alpha in eval_lambdas:
            blend = oof_base.copy()
            for name, cols in blocks.items():
                block_idx = [target_cols.index(c) for c in cols]
                imp = Xtr[cols].isna().to_numpy()
                block_base = blend[:, block_idx]
                block_resid = oof_residual[:, block_idx]
                block_base[imp] = block_base[imp] + (alpha * block_resid[imp])
                blend[:, block_idx] = block_base
            blend = np.nan_to_num(blend, nan=0.0)
            blend = _clip_predictions(blend, target_cols)
            blend = _apply_copy_through(blend, Xtr, target_cols)
            metrics = compute_imputed_only_metrics(
                ytr,
                blend,
                Xtr,
                target_cols,
                data.sensory_target_cols,
            ).to_dict()
            weighted_rmse = _weighted_rmse_imputed_only(
                ytr,
                blend,
                Xtr,
                data.X_test,
                target_cols,
            )
            metrics["weighted_rmse_imputed_only"] = weighted_rmse
            wrmse = compute_wrmse_imputed_only(
                y_true=ytr,
                oof_pred=blend,
                X_train_features=Xtr,
                X_test_features=data.X_test,
                target_cols=target_cols,
            )
            metrics["wrmse_imputed_only"] = wrmse.wrmse_imputed_only
            metrics["wrmse_all"] = wrmse.wrmse_all
            cv_metrics[f"lambda_{alpha:.2f}"] = metrics

    for name, cols in blocks.items():
        block_idx = [target_cols.index(c) for c in cols]
        oof_block = oof_preds[:, block_idx]
        y_block = ytr[cols].to_numpy(dtype=np.float32)
        residual = y_block - oof_block

        X_block = _build_block_features(X_base, Xtr, oof_block, cols, add_missing_indicators)
        cat_idx = [X_block.columns.get_loc(c) for c in cat_cols]

        LOGGER.info("Training residual block %s (%d targets).", name, len(cols))
        model = CatBoostRegressor(**params)
        model.fit(X_block, residual, cat_features=cat_idx)

        model_path = model_dir / f"{name}.cbm"
        model.save_model(model_path)

        feature_columns[name] = list(X_block.columns)
        model_paths[name] = str(model_path)

    write_json(
        out_dir / "residual_config.json",
        {
            "run_id": run_id,
            "track": track,
            "method": method,
            "device": device,
            "seed": seed,
            "data_root": data_root,
            "oof_path": str(oof_path),
            "limit_rows": limit_rows,
            "base_cols": base_cols,
            "cat_feature_cols": cat_cols,
            "num_fill_values": num_fill,
            "blocks": blocks,
            "skipped_blocks": skipped_blocks,
            "boost_targets": boost_targets,
            "topk_targets": topk_targets,
            "topk_source": topk_source,
            "target_cols": target_cols,
            "sensory_target_cols": data.sensory_target_cols,
            "add_missing_indicators": add_missing_indicators,
            "feature_columns": feature_columns,
            "params": params,
            "model_paths": model_paths,
        },
    )

    write_json(
        out_dir / "run_summary.json",
        {
            "run_id": run_id,
            "finished_utc": utc_now_iso(),
            "track": track,
            "method": method,
            "device": device,
            "seed": seed,
            "baseline_oof_metrics": baseline_metrics,
            "baseline_oof_breakdown": baseline_breakdown,
            "baseline_weighted_rmse_imputed_only": baseline_weighted_rmse if do_cv else None,
            "baseline_wrmse": {
                "wrmse_imputed_only": baseline_wrmse.wrmse_imputed_only,
                "wrmse_all": baseline_wrmse.wrmse_all,
            },
            "cv_metrics": cv_metrics if do_cv else None,
            "boost_targets": boost_targets,
            "topk_targets": topk_targets,
            "topk_source": topk_source,
            "artifacts": {
                "residual_config": "residual_config.json",
                "models_dir": "models",
                "oof_residual_npz": "oof_residual_predictions_train.npz" if do_cv else None,
            },
        },
    )

    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--oof-path", required=True)
    p.add_argument("--device", default="gpu", choices=["gpu", "cpu", "auto"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do-cv", type=int, default=1)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--eval-lambdas", default="0.1,0.2,0.3")
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--add-missing-indicators", type=int, default=1)
    p.add_argument("--topk-targets", type=int, default=0)
    p.add_argument("--topk-source", type=str, default="test", choices=["train", "test"])

    p.add_argument("--iterations", type=int, default=4000)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--l2-leaf-reg", type=float, default=3.0)
    p.add_argument("--random-strength", type=float, default=1.0)
    p.add_argument("--bagging-temperature", type=float, default=0.5)
    p.add_argument("--od-wait", type=int, default=200)
    p.add_argument("--thread-count", type=int, default=1)

    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    device = _resolve_device(args.device)
    add_missing = bool(args.add_missing_indicators)
    eval_lambdas = [float(x.strip()) for x in args.eval_lambdas.split(",") if x.strip()]

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        oof_path=Path(args.oof_path),
        device=device,
        seed=args.seed,
        do_cv=bool(args.do_cv),
        n_splits=args.n_splits,
        eval_lambdas=eval_lambdas,
        limit_rows=args.limit_rows,
        add_missing_indicators=add_missing,
        topk_targets=int(args.topk_targets),
        topk_source=args.topk_source,
        args=args,
    )
    print(f"[asia2026 t1 catboost residual] done -> {out_dir}")


if __name__ == "__main__":
    main()
