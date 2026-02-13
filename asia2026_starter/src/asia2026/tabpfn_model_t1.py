from __future__ import annotations

import inspect
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabpfn import TabPFNRegressor


def _make_numeric_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median", add_indicator=True)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Encode string columns with one-hot; median-impute + scale numeric columns."""
    cat_cols = [c for c in X.columns if X[c].dtype == object]
    num_cols = [c for c in X.columns if c not in cat_cols]

    return ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imp", SimpleImputer(strategy="constant", fill_value="MISSING")),
                    ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                cat_cols,
            ),
            (
                "num",
                Pipeline([
                    ("imp", _make_numeric_imputer()),
                    ("sc", StandardScaler()),
                ]),
                num_cols,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.0,
    )


def _init_tabpfn(device: str) -> TabPFNRegressor:
    # Force safer, deterministic config if the version supports it.
    extra = {
        "n_ensemble_configurations": 1,
        "n_estimators": 1,
        "n_jobs": 1,
    }
    sig = inspect.signature(TabPFNRegressor)
    kwargs = {k: v for k, v in extra.items() if k in sig.parameters}
    return TabPFNRegressor(device=device, ignore_pretraining_limits=True, **kwargs)


def _safe_reduce(Xtr: np.ndarray, Xte: np.ndarray, seed: int, do_pca: bool) -> tuple[np.ndarray, np.ndarray]:
    # Drop constant columns first.
    vt = VarianceThreshold()
    Xtr_v = vt.fit_transform(Xtr)
    Xte_v = vt.transform(Xte)

    if not do_pca:
        return Xtr_v, Xte_v

    # Use randomized PCA to avoid ARPACK.
    n_features = Xtr_v.shape[1]
    n_samples = Xtr_v.shape[0]
    n_components = min(256, n_features, max(1, n_samples - 1))
    if n_components < 1:
        return Xtr_v, Xte_v

    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    Xtr_p = pca.fit_transform(Xtr_v)
    Xte_p = pca.transform(Xte_v)
    return Xtr_p, Xte_p


def tabpfn_predict_multioutput_t1(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X: pd.DataFrame,
    target_cols: List[str],
    copy_through_cols: Optional[List[str]] = None,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Track-1 safe TabPFN with reduced ensemble, single-threaded preprocessing."""
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Force single-threaded behavior to reduce ARPACK instability.
    os.environ.setdefault("TABPFN_NUM_WORKERS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    pre = build_preprocessor(X_train)
    Xtr_all = np.asarray(pre.fit_transform(X_train), dtype=np.float32)
    Xte_all = np.asarray(pre.transform(X), dtype=np.float32)
    Xtr_all = np.nan_to_num(Xtr_all, nan=0.0, posinf=0.0, neginf=0.0)
    Xte_all = np.nan_to_num(Xte_all, nan=0.0, posinf=0.0, neginf=0.0)

    feat_names = list(pre.get_feature_names_out())
    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    n = Xte_all.shape[0]
    out = np.zeros((n, len(target_cols)), dtype=np.float32)

    if max_train_samples is not None and Xtr_all.shape[0] > max_train_samples:
        idx = rng.choice(Xtr_all.shape[0], size=max_train_samples, replace=False)
        Xtr_fit = Xtr_all[idx]
        y_fit_full = y_train.iloc[idx]
    else:
        Xtr_fit = Xtr_all
        y_fit_full = y_train

    for j, col in enumerate(target_cols):
        y = y_fit_full[col].to_numpy(dtype=np.float32)
        if copy_through_cols and col in copy_through_cols:
            out[:, j] = np.nan
            continue

        leak_idx = name_to_idx.get(f"num__{col}")
        if leak_idx is not None:
            Xtr_use = Xtr_fit.copy()
            Xte_use = Xte_all.copy()
            Xtr_use[:, leak_idx] = np.nan
            Xte_use[:, leak_idx] = np.nan
            Xtr_use = np.nan_to_num(Xtr_use, nan=0.0, posinf=0.0, neginf=0.0)
            Xte_use = np.nan_to_num(Xte_use, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            Xtr_use = Xtr_fit
            Xte_use = Xte_all

        try:
            Xtr_safe, Xte_safe = _safe_reduce(Xtr_use, Xte_use, seed=seed, do_pca=False)
            model = _init_tabpfn(device)
            model.fit(Xtr_safe, y)
            out[:, j] = model.predict(Xte_safe).astype(np.float32)
        except Exception as e:
            # Retry with PCA reduction if preprocessing fails (e.g., ARPACK).
            Xtr_safe, Xte_safe = _safe_reduce(Xtr_use, Xte_use, seed=seed, do_pca=True)
            model = _init_tabpfn(device)
            model.fit(Xtr_safe, y)
            out[:, j] = model.predict(Xte_safe).astype(np.float32)

    return out
