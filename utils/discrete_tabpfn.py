from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabpfn import TabPFNClassifier, TabPFNRegressor


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


def _expected_from_proba_fixed(
    proba: np.ndarray,
    classes: np.ndarray,
    target_classes: np.ndarray,
) -> np.ndarray:
    target_classes = target_classes.astype(np.float32)
    p_full = np.zeros((proba.shape[0], len(target_classes)), dtype=np.float32)
    class_to_idx = {int(c): i for i, c in enumerate(target_classes)}
    for col, cls in enumerate(classes):
        idx = class_to_idx.get(int(cls))
        if idx is not None:
            p_full[:, idx] = proba[:, col]
    return p_full @ target_classes


def tabpfn_predict_multioutput_discrete(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X: pd.DataFrame,
    target_cols: List[str],
    sensory_target_cols: List[str],
    copy_through_cols: Optional[List[str]] = None,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """TabPFN classifier-as-regressor for sensory targets and anyana; regressor for others.

    Notes:
      * We set the target column feature itself to NaN during fitting/prediction to avoid trivial leakage.
      * You should still apply copy-through afterwards for observed targets.
    """
    rng = np.random.default_rng(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Preprocess once (same feature space for every target)
    pre = build_preprocessor(X_train)
    Xtr_all = np.asarray(pre.fit_transform(X_train), dtype=np.float32)
    Xte_all = np.asarray(pre.transform(X), dtype=np.float32)
    # Ensure no NaNs remain after preprocessing; TabPFN SVD can fail on NaNs.
    Xtr_all = np.nan_to_num(Xtr_all, nan=0.0, posinf=0.0, neginf=0.0)
    Xte_all = np.nan_to_num(Xte_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Map original numeric columns -> transformed column indices so we can
    # mask the target's own feature (avoid trivial leakage when it happens to be observed).
    feat_names = list(pre.get_feature_names_out())
    name_to_idx = {n: i for i, n in enumerate(feat_names)}

    n = Xte_all.shape[0]
    out = np.zeros((n, len(target_cols)), dtype=np.float32)

    # Optional subsampling for speed (especially on Track 1 with 1694 rows)
    if max_train_samples is not None and Xtr_all.shape[0] > max_train_samples:
        idx = rng.choice(Xtr_all.shape[0], size=max_train_samples, replace=False)
        Xtr_fit = Xtr_all[idx]
        y_fit_full = y_train.iloc[idx]
    else:
        Xtr_fit = Xtr_all
        y_fit_full = y_train

    sensory_targets = set(sensory_target_cols)
    sensory_classes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    anyana_classes = np.array([0.0, 1.0], dtype=np.float32)
    for j, col in enumerate(target_cols):
        is_sensory = col in sensory_targets
        is_anyana = col == "anyana"

        # Skip columns that are always provided in X (e.g., anyana) if requested
        if copy_through_cols and col in copy_through_cols:
            out[:, j] = np.nan
            continue

        # Mask the target column feature itself if present in numeric passthrough.
        # ColumnTransformer names numeric passthrough features as: "num__<colname>".
        leak_idx = name_to_idx.get(f"num__{col}")
        if leak_idx is not None:
            Xtr_use = Xtr_fit.copy()
            Xte_use = Xte_all.copy()
            Xtr_use[:, leak_idx] = np.nan
            Xte_use[:, leak_idx] = np.nan
            # Re-zero any NaNs introduced by leakage masking.
            Xtr_use = np.nan_to_num(Xtr_use, nan=0.0, posinf=0.0, neginf=0.0)
            Xte_use = np.nan_to_num(Xte_use, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            Xtr_use = Xtr_fit
            Xte_use = Xte_all

        if is_anyana:
            y = y_fit_full[col].to_numpy()
            y_int = np.rint(y).astype(np.int64)
            model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
            model.fit(Xtr_use, y_int)
            proba = model.predict_proba(Xte_use)
            out[:, j] = _expected_from_proba_fixed(
                proba,
                model.classes_,
                anyana_classes,
            ).astype(np.float32)
        elif is_sensory:
            y = y_fit_full[col].to_numpy()
            y_int = np.rint(y).astype(np.int64)
            model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
            model.fit(Xtr_use, y_int)
            proba = model.predict_proba(Xte_use)
            out[:, j] = _expected_from_proba_fixed(
                proba,
                model.classes_,
                sensory_classes,
            ).astype(np.float32)
        else:
            y = y_fit_full[col].to_numpy(dtype=np.float32)
            model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
            model.fit(Xtr_use, y)
            out[:, j] = model.predict(Xte_use).astype(np.float32)

    return out
