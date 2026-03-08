from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class WeightedOofReport:
    wrmse_imputed_only: float
    wrmse_all: float
    per_target_contrib: Dict[str, float]
    weights: Dict[str, float]
    mse_per_target: Dict[str, float]
    mse_per_target_all: Dict[str, float]


def compute_wrmse_imputed_only(
    *,
    y_true: pd.DataFrame,
    oof_pred: np.ndarray,
    X_train_features: pd.DataFrame,
    X_test_features: pd.DataFrame,
    target_cols: List[str],
) -> WeightedOofReport:
    # weights from TEST missingness
    w: Dict[str, float] = {}
    mse: Dict[str, float] = {}
    mse_all: Dict[str, float] = {}
    contrib: Dict[str, float] = {}
    contrib_all: Dict[str, float] = {}

    y_mat = y_true[target_cols].to_numpy(dtype=np.float32)

    for j, c in enumerate(target_cols):
        if c not in X_train_features.columns or c not in X_test_features.columns:
            continue

        # imputed-only mask defined by TRAIN feature missingness
        m_imp = X_train_features[c].isna().to_numpy()
        if m_imp.sum() == 0:
            continue

        err = (y_mat[m_imp, j] - oof_pred[m_imp, j]).astype(np.float32)
        mse_c = float(np.mean(err * err))
        mse[c] = mse_c

        err_all = (y_mat[:, j] - oof_pred[:, j]).astype(np.float32)
        mse_c_all = float(np.mean(err_all * err_all))
        mse_all[c] = mse_c_all

        w_c = float(X_test_features[c].isna().mean())
        w[c] = w_c
        contrib[c] = w_c * mse_c
        contrib_all[c] = w_c * mse_c_all

    denom = sum(w.values()) if w else 1.0
    wrmse = float(np.sqrt(sum(contrib.values()) / denom)) if contrib else 0.0
    wrmse_all = float(np.sqrt(sum(contrib_all.values()) / denom)) if contrib_all else 0.0
    return WeightedOofReport(
        wrmse_imputed_only=wrmse,
        wrmse_all=wrmse_all,
        per_target_contrib=contrib,
        weights=w,
        mse_per_target=mse,
        mse_per_target_all=mse_all,
    )
