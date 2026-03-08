from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

from asia2026.sci_dermatomes import DERMS_28, SUFFIXES_4

DERM_ALIASES = {
    "s4_5": "s45",
    "s4-5": "s45",
    "s4s5": "s45",
    "s4/5": "s45",
}


def _norm_derm(d: str) -> str:
    d = d.lower()
    return DERM_ALIASES.get(d, d)


def build_sens_grid(df: pd.DataFrame) -> np.ndarray:
    n = len(df)
    X = np.full((n, len(DERMS_28), len(SUFFIXES_4)), np.nan, dtype=np.float32)
    derm2i = {d: i for i, d in enumerate(DERMS_28)}
    suf2i = {s: i for i, s in enumerate(SUFFIXES_4)}

    for col in df.columns:
        col_lower = col.lower()
        for suf in SUFFIXES_4:
            if col_lower.endswith(suf):
                derm = _norm_derm(col_lower[: -len(suf)])
                di = derm2i.get(derm)
                if di is None:
                    break
                si = suf2i[suf]
                X[:, di, si] = df[col].to_numpy(dtype=np.float32)
                break
    return X


def preprocess_static(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    cols: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    cols = list(cols)
    Xtr = Xtr[cols].copy()
    Xte = Xte[cols].copy()

    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num_cols = [c for c in cols if c not in cat_cols]

    for c in num_cols:
        med = float(Xtr[c].median()) if not Xtr[c].isna().all() else 0.0
        Xtr[c] = Xtr[c].fillna(med)
        Xte[c] = Xte[c].fillna(med)
        mu = float(Xtr[c].mean())
        sd = float(Xtr[c].std())
        if sd == 0.0 or np.isnan(sd):
            sd = 1.0
        Xtr[c] = (Xtr[c] - mu) / sd
        Xte[c] = (Xte[c] - mu) / sd

    for c in cat_cols:
        Xtr[c] = Xtr[c].fillna("MISSING")
        Xte[c] = Xte[c].fillna("MISSING")

    combined = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=cat_cols, dtype=np.float32)

    n_tr = len(Xtr)
    return (
        combined.iloc[:n_tr].to_numpy(dtype=np.float32),
        combined.iloc[n_tr:].to_numpy(dtype=np.float32),
    )


def broadcast_static(static: np.ndarray) -> np.ndarray:
    return np.repeat(static[:, None, :], repeats=len(DERMS_28), axis=1)


def build_saits_X(X_sens: np.ndarray, X_static: np.ndarray) -> np.ndarray:
    X_stat = X_static if X_static.ndim == 3 else broadcast_static(X_static)
    return np.concatenate([X_sens, X_stat], axis=2)


def build_target_mapping(target_cols: Iterable[str]) -> List[Tuple[int, int] | None]:
    derm_idx = {d: i for i, d in enumerate(DERMS_28)}
    mapping: List[Tuple[int, int] | None] = []
    for col in target_cols:
        if col == "anyana":
            mapping.append(None)
            continue
        matched = False
        for s_idx, suf in enumerate(SUFFIXES_4):
            col_lower = col.lower()
            if col_lower.endswith(suf):
                derm = _norm_derm(col_lower[: -len(suf)])
                if derm not in derm_idx:
                    raise ValueError(f"Unknown dermatome for column {col}")
                mapping.append((derm_idx[derm], s_idx))
                matched = True
                break
        if not matched:
            raise ValueError(f"Unsupported target column: {col}")
    return mapping


def grid_to_targets(
    pred_grid: np.ndarray,
    mapping: Iterable[Tuple[int, int] | None],
    anyana: float | np.ndarray,
) -> np.ndarray:
    mapping = list(mapping)
    out = np.zeros((pred_grid.shape[0], len(mapping)), dtype=np.float32)
    if isinstance(anyana, (int, float)):
        anyana = float(anyana)
        anyana_arr = np.full(pred_grid.shape[0], anyana, dtype=np.float32)
    else:
        anyana_arr = np.asarray(anyana, dtype=np.float32).reshape(-1)
    for j, m in enumerate(mapping):
        if m is None:
            out[:, j] = anyana_arr
        else:
            d_idx, s_idx = m
            out[:, j] = pred_grid[:, d_idx, s_idx]
    return out


def apply_copy_through(pred: np.ndarray, features: pd.DataFrame, target_cols: Iterable[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    for j, c in enumerate(target_cols):
        if c in features.columns:
            obs = features[c].to_numpy()
            m = ~pd.isna(obs)
            pred[m, j] = obs[m]
    return pred


def clip_predictions(pred: np.ndarray, target_cols: Iterable[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    for j, c in enumerate(target_cols):
        if c == "anyana":
            pred[:, j] = np.clip(pred[:, j], 0.0, 1.0)
        else:
            pred[:, j] = np.clip(pred[:, j], 0.0, 2.0)
    return pred
