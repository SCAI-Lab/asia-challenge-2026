from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def apply_copy_through(
    X: pd.DataFrame,
    pred: np.ndarray,
    target_cols: List[str],
) -> np.ndarray:
    """If a target column exists in X and is observed (not NaN), force prediction to that value."""
    pred = np.asarray(pred).copy()
    col2idx = {c: i for i, c in enumerate(target_cols)}
    for c in target_cols:
        if c in X.columns:
            idx = col2idx[c]
            obs = X[c].to_numpy()
            m = ~pd.isna(obs)
            pred[m, idx] = obs[m]
    return pred


def _make_numeric_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median", add_indicator=True)


def _nan_to_num(X):
    if sparse.issparse(X):
        X = X.tocsr(copy=True)
        X.data = np.nan_to_num(X.data, nan=0.0, posinf=0.0, neginf=0.0)
        return X
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def baseline_time_mean(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X: pd.DataFrame,
    target_cols: List[str],
    time_col: str = "time",
) -> np.ndarray:
    """Copy-through, otherwise fill with mean(label | time)."""
    # mean per time for each target
    g = y_train.groupby(X_train[time_col])
    means_by_time = g.mean(numeric_only=True)
    global_mean = y_train.mean(numeric_only=True)

    out = np.zeros((len(X), len(target_cols)), dtype=np.float32)
    times = X[time_col].to_numpy()
    for j, c in enumerate(target_cols):
        if c in means_by_time.columns:
            # map time -> mean, fallback global mean
            m = means_by_time[c]
            out[:, j] = np.array([m.get(t, global_mean[c]) for t in times], dtype=np.float32)
        else:
            out[:, j] = float(global_mean.get(c, 0.0))
    return apply_copy_through(X, out, target_cols)


def baseline_strat_time_mean(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X: pd.DataFrame,
    target_cols: List[str],
    strat_cols: List[str],
    time_col: str = "time",
    min_group: int = 20,
) -> np.ndarray:
    """Copy-through, else use mean(label | time + metadata strata) with backoff.

    Backoff:
      * if (time + strata) group has < min_group examples -> use time-only mean
      * if time missing (shouldn't) -> global mean
    """
    keys = [time_col] + strat_cols

    df = X_train[keys].copy()
    # normalize missing categories
    for c in strat_cols:
        if df[c].dtype == object:
            df[c] = df[c].fillna("MISSING")
        else:
            df[c] = df[c].fillna(-1)

    time_means = y_train.groupby(X_train[time_col]).mean(numeric_only=True)
    global_mean = y_train.mean(numeric_only=True)

    merged = pd.concat([df.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    grp = merged.groupby(keys)
    means = grp[target_cols].mean(numeric_only=True)
    counts = grp.size()

    out = np.zeros((len(X), len(target_cols)), dtype=np.float32)
    Xk = X[keys].copy()
    for c in strat_cols:
        if Xk[c].dtype == object:
            Xk[c] = Xk[c].fillna("MISSING")
        else:
            Xk[c] = Xk[c].fillna(-1)

    # row-wise lookup with backoff
    for i in range(len(Xk)):
        key = tuple(Xk.iloc[i].tolist())
        time_val = Xk.iloc[i][time_col]
        use_strata = False
        if key in counts.index:
            use_strata = int(counts.loc[key]) >= min_group
        for j, t in enumerate(target_cols):
            if use_strata and t in means.columns:
                out[i, j] = float(means.loc[key, t])
            else:
                if time_val in time_means.index and t in time_means.columns:
                    out[i, j] = float(time_means.loc[time_val, t])
                else:
                    out[i, j] = float(global_mean.get(t, 0.0))

    return apply_copy_through(X, out, target_cols)


def baseline_knn15(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X: pd.DataFrame,
    target_cols: List[str],
    always_missing_targets: List[str],
    knn_feature_cols: List[str],
    time_fallback: np.ndarray,
    n_neighbors: int = 25,
) -> np.ndarray:
    """Start from a fallback prediction (e.g., time-mean), then replace the 15 always-missing targets
    using KNN regression on motor + metadata.
    """
    out = np.asarray(time_fallback).copy()

    if not always_missing_targets:
        return apply_copy_through(X, out, target_cols)

    # Build preprocessing for KNN
    Xtr = X_train[knn_feature_cols].copy()
    Xte = X[knn_feature_cols].copy()
    cat_cols = [c for c in knn_feature_cols if Xtr[c].dtype == object]
    num_cols = [c for c in knn_feature_cols if c not in cat_cols]

    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                ("oh", OneHotEncoder(handle_unknown="ignore")),
            ]), cat_cols),
            ("num", Pipeline([
                ("imp", _make_numeric_imputer()),
                ("sc", StandardScaler()),
            ]), num_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    ytr = y_train[always_missing_targets].to_numpy(dtype=np.float32)

    model = Pipeline(
        steps=[
            ("pre", pre),
            ("nanfix", FunctionTransformer(_nan_to_num, validate=False)),
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance", n_jobs=-1)),
        ]
    )
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte).astype(np.float32)

    col2idx = {c: i for i, c in enumerate(target_cols)}
    for j, c in enumerate(always_missing_targets):
        out[:, col2idx[c]] = yhat[:, j]

    return apply_copy_through(X, out, target_cols)
