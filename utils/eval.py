from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .utils import ensure_dir, read_json


@dataclass
class ImputedOnlyMetrics:
    rmse_sensory_imputed_only: float
    rmse_all_imputed_only: float
    mae_sensory_imputed_only: float
    mae_all_imputed_only: float
    r2_sensory_imputed_only: float
    r2_all_imputed_only: float
    n_imputed_sensory: int
    n_imputed_all: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "rmse_sensory_imputed_only": self.rmse_sensory_imputed_only,
            "rmse_all_imputed_only": self.rmse_all_imputed_only,
            "mae_sensory_imputed_only": self.mae_sensory_imputed_only,
            "mae_all_imputed_only": self.mae_all_imputed_only,
            "r2_sensory_imputed_only": self.r2_sensory_imputed_only,
            "r2_all_imputed_only": self.r2_all_imputed_only,
            "n_imputed_sensory": float(self.n_imputed_sensory),
            "n_imputed_all": float(self.n_imputed_all),
        }


def _masked_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _masked_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)))


def _masked_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return 0.0
    y_mean = y_true.mean()
    ss_tot = np.sum((y_true - y_mean) ** 2)
    if ss_tot == 0:
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - ss_res / ss_tot)


def _metrics_from_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if mask.ndim > 1:
        mask_flat = mask.ravel()
    else:
        mask_flat = mask
    y_true_flat = y_true.ravel() if y_true.ndim > 1 else y_true
    y_pred_flat = y_pred.ravel() if y_pred.ndim > 1 else y_pred
    y_true_sel = y_true_flat[mask_flat]
    y_pred_sel = y_pred_flat[mask_flat]
    return {
        "rmse": _masked_rmse(y_true_sel, y_pred_sel),
        "mae": _masked_mae(y_true_sel, y_pred_sel),
        "r2": _masked_r2(y_true_sel, y_pred_sel),
        "n": float(mask_flat.sum()),
    }


def compute_imputed_only_metrics(
    y_true_df: pd.DataFrame,
    y_pred: np.ndarray,
    features_df: pd.DataFrame,
    target_cols: List[str],
    sensory_cols: List[str],
) -> ImputedOnlyMetrics:
    if y_pred.shape[1] != len(target_cols):
        raise ValueError("y_pred columns do not match target_cols")

    mask_all = features_df[target_cols].isna().to_numpy()
    y_true = y_true_df[target_cols].to_numpy(dtype=np.float32)

    sensory_set = set(sensory_cols)
    sensory_idx = [i for i, c in enumerate(target_cols) if c in sensory_set]
    if sensory_idx:
        mask_s = mask_all[:, sensory_idx]
        y_true_s = y_true[:, sensory_idx]
        y_pred_s = y_pred[:, sensory_idx]
    else:
        mask_s = np.zeros((y_true.shape[0], 0), dtype=bool)
        y_true_s = np.zeros((y_true.shape[0], 0), dtype=np.float32)
        y_pred_s = np.zeros((y_true.shape[0], 0), dtype=np.float32)

    yt_all = y_true[mask_all]
    yp_all = y_pred[mask_all]
    yt_s = y_true_s[mask_s]
    yp_s = y_pred_s[mask_s]

    return ImputedOnlyMetrics(
        rmse_sensory_imputed_only=_masked_rmse(yt_s, yp_s),
        rmse_all_imputed_only=_masked_rmse(yt_all, yp_all),
        mae_sensory_imputed_only=_masked_mae(yt_s, yp_s),
        mae_all_imputed_only=_masked_mae(yt_all, yp_all),
        r2_sensory_imputed_only=_masked_r2(yt_s, yp_s),
        r2_all_imputed_only=_masked_r2(yt_all, yp_all),
        n_imputed_sensory=int(mask_s.sum()),
        n_imputed_all=int(mask_all.sum()),
    )


def compute_imputed_only_breakdown(
    y_true_df: pd.DataFrame,
    y_pred: np.ndarray,
    features_df: pd.DataFrame,
    target_cols: List[str],
    sensory_cols: List[str],
    time_col: str = "time",
) -> Dict[str, Dict[str, Dict[str, float]]]:
    if y_pred.shape[1] != len(target_cols):
        raise ValueError("y_pred columns do not match target_cols")

    per_target: Dict[str, Dict[str, float]] = {}
    sensory_set = set(sensory_cols)
    y_true_all = y_true_df[target_cols].to_numpy(dtype=np.float32)

    for j, col in enumerate(target_cols):
        mask = features_df[col].isna().to_numpy()
        metrics = _metrics_from_mask(y_true_all[:, j], y_pred[:, j], mask)
        metrics["sensory"] = bool(col in sensory_set)
        per_target[col] = metrics

    per_time: Dict[str, Dict[str, Dict[str, float]]] = {}
    if time_col in features_df.columns:
        time_vals = features_df[time_col].to_numpy()
        sensory_idx = [i for i, c in enumerate(target_cols) if c in sensory_set]
        for t in np.unique(time_vals):
            rows = time_vals == t
            if not np.any(rows):
                continue
            mask_all = features_df.loc[rows, target_cols].isna().to_numpy()
            y_true_t = y_true_all[rows]
            y_pred_t = y_pred[rows]
            metrics_all = _metrics_from_mask(y_true_t, y_pred_t, mask_all)
            if sensory_idx:
                mask_s = mask_all[:, sensory_idx]
                metrics_s = _metrics_from_mask(y_true_t[:, sensory_idx], y_pred_t[:, sensory_idx], mask_s)
            else:
                metrics_s = {"rmse": 0.0, "mae": 0.0, "r2": 0.0, "n": 0.0}
            per_time[str(t)] = {"all": metrics_all, "sensory": metrics_s}

    return {"per_target": per_target, "per_time": per_time}


def save_oof_npz(path: Path, ids: np.ndarray, target_cols: List[str], preds: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.savez_compressed(
        path,
        ids=ids.astype(str),
        target_cols=np.array(target_cols, dtype=object),
        preds=preds.astype(np.float32),
    )


def load_oof_npz(path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    data = np.load(path, allow_pickle=True)
    ids = data["ids"]
    target_cols = data["target_cols"].tolist()
    preds = data["preds"]
    return ids, target_cols, preds


def load_per_target_breakdown(run_dir: Path) -> Dict[str, Dict[str, float]] | None:
    run_summary = run_dir / "run_summary.json"
    if not run_summary.exists():
        return None
    data = read_json(run_summary)
    for key in ("metrics_breakdown", "baseline_oof_breakdown", "oof_breakdown"):
        block = data.get(key)
        if isinstance(block, dict):
            per_target = block.get("per_target")
            if isinstance(per_target, dict) and per_target:
                return per_target
    return None


def make_time_stratified_folds(
    time_values: np.ndarray,
    n_splits: int,
    seed: int = 42,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    time_values = np.asarray(time_values)
    unique_times = np.unique(time_values)

    buckets = []
    for t in unique_times:
        idx = np.where(time_values == t)[0]
        rng.shuffle(idx)
        splits = np.array_split(idx, n_splits)
        buckets.append(splits)

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    for fold in range(n_splits):
        val_parts = [bucket[fold] for bucket in buckets if len(bucket[fold])]
        val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
        train_mask = np.ones(len(time_values), dtype=bool)
        train_mask[val_idx] = False
        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, val_idx))

    return folds
