from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _flatten(a: np.ndarray) -> np.ndarray:
    return np.asarray(a).reshape(-1)


def _imputed_mask(features: pd.DataFrame, target_cols: List[str]) -> np.ndarray:
    masks = []
    for c in target_cols:
        if c in features.columns:
            masks.append(features[c].isna().to_numpy())
        else:
            masks.append(np.zeros(len(features), dtype=bool))
    return np.column_stack(masks)


def _masked_flatten(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if mask is None or not mask.any():
        return np.array([]), np.array([])
    return y_true[mask], y_pred[mask]


def _masked_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _masked_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    mse = mean_squared_error(y_true, y_pred)
    return float(np.sqrt(mse))


def _masked_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    return float(mean_absolute_error(y_true, y_pred))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    sensory_target_cols: List[str],
    features: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Compute metrics that are easy to compare across baselines.

    Kaggle uses **R² (coefficient of determination)** for the imputed sensory scores.
    We report:
      * r2_sensory (sensory targets only)
      * r2_all (all 112 targets, including anyana)
      * rmse_sensory / mae_sensory
      * rmse_all / mae_all
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    col2idx = {c: i for i, c in enumerate(target_cols)}
    sensory_idx = [col2idx[c] for c in sensory_target_cols]

    yt_s = y_true[:, sensory_idx]
    yp_s = y_pred[:, sensory_idx]

    metrics: Dict[str, float] = {}

    # Global/flattened metrics
    metrics["r2_sensory"] = float(r2_score(_flatten(yt_s), _flatten(yp_s)))
    mse_sensory = mean_squared_error(_flatten(yt_s), _flatten(yp_s))
    metrics["rmse_sensory"] = float(np.sqrt(mse_sensory))
    metrics["mae_sensory"] = float(mean_absolute_error(_flatten(yt_s), _flatten(yp_s)))

    metrics["r2_all"] = float(r2_score(_flatten(y_true), _flatten(y_pred)))
    mse_all = mean_squared_error(_flatten(y_true), _flatten(y_pred))
    metrics["rmse_all"] = float(np.sqrt(mse_all))
    metrics["mae_all"] = float(mean_absolute_error(_flatten(y_true), _flatten(y_pred)))

    rmse_imputed_only = float("nan")
    if features is not None:
        mask = _imputed_mask(features, target_cols)
        yt_all, yp_all = _masked_flatten(y_true, y_pred, mask)
        metrics["r2_all_imputed_only"] = _masked_r2(yt_all, yp_all)
        metrics["rmse_all_imputed_only"] = _masked_rmse(yt_all, yp_all)
        metrics["mae_all_imputed_only"] = _masked_mae(yt_all, yp_all)

        mask_s = mask[:, sensory_idx] if mask.size else np.array([])
        yt_s_m, yp_s_m = _masked_flatten(yt_s, yp_s, mask_s)
        metrics["r2_sensory_imputed_only"] = _masked_r2(yt_s_m, yp_s_m)
        metrics["rmse_sensory_imputed_only"] = _masked_rmse(yt_s_m, yp_s_m)
        metrics["mae_sensory_imputed_only"] = _masked_mae(yt_s_m, yp_s_m)
        rmse_imputed_only = metrics["rmse_all_imputed_only"]
    else:
        metrics["r2_all_imputed_only"] = float("nan")
        metrics["rmse_all_imputed_only"] = float("nan")
        metrics["mae_all_imputed_only"] = float("nan")
        metrics["r2_sensory_imputed_only"] = float("nan")
        metrics["rmse_sensory_imputed_only"] = float("nan")
        metrics["mae_sensory_imputed_only"] = float("nan")

    metrics["rmse_imputed_only"] = rmse_imputed_only

    return metrics
