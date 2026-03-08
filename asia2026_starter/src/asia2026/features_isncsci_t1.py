from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Canonical 28 dermatomes in ISNCSCI order (avoid lexicographic bugs like t10 < t2)
DERMS_28: List[str] = [
    "c2", "c3", "c4", "c5", "c6", "c7", "c8",
    "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12",
    "l1", "l2", "l3", "l4", "l5",
    "s1", "s2", "s3", "s45",
]
SUFFIXES_4: List[str] = ["ltl", "ltr", "ppl", "ppr"]  # LT/PP x Left/Right

_DERM_ALIASES: Dict[str, str] = {
    "s4_5": "s45",
    "s4-5": "s45",
    "s4/5": "s45",
    "s4s5": "s45",
    "s4_5 ": "s45",
}
_DERM2I = {d: i for i, d in enumerate(DERMS_28)}
_SUF2I = {s: i for i, s in enumerate(SUFFIXES_4)}


def _norm_derm(raw: str) -> str:
    s = raw.strip().lower()
    s = _DERM_ALIASES.get(s, s)
    return s


def _parse_sens_col(col: str) -> Tuple[int, int] | None:
    cl = col.lower()
    for suf in SUFFIXES_4:
        if cl.endswith(suf):
            derm = _norm_derm(cl[:-len(suf)])
            di = _DERM2I.get(derm)
            if di is None:
                return None
            return di, _SUF2I[suf]
    return None


def _build_grid_and_mask(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    grid = np.full((n, 28, 4), np.nan, dtype=np.float32)
    mask = np.zeros((n, 28, 4), dtype=np.float32)

    for col in df.columns:
        parsed = _parse_sens_col(col)
        if parsed is None:
            continue
        di, si = parsed
        vals = df[col].to_numpy(dtype=np.float32)
        m = ~pd.isna(vals)
        grid[:, di, si] = vals
        mask[:, di, si] = m.astype(np.float32)

    return grid, mask


def _safe_mean_std(vals: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # vals: [N,28], m: [N,28] in {0,1}
    cnt = m.sum(axis=1)
    cnt_safe = np.where(cnt > 0, cnt, 1.0)
    s1 = np.nan_to_num(vals, nan=0.0) * m
    mean = s1.sum(axis=1) / cnt_safe

    s2 = np.nan_to_num(vals * vals, nan=0.0) * m
    var = s2.sum(axis=1) / cnt_safe - mean * mean
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def _sensory_level_proxy(grid: np.ndarray, mask: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
    # ISNCSCI sensory level: most caudal intact dermatome for LT and PP (score 2).
    # We compute proxy using only observed pairs on each side.
    # side="L": LT=ltl (0), PP=ppl (2); side="R": LT=ltr (1), PP=ppr (3)
    if side == "L":
        lt_i, pp_i = 0, 2
    else:
        lt_i, pp_i = 1, 3

    lt = grid[:, :, lt_i]
    pp = grid[:, :, pp_i]
    m = (mask[:, :, lt_i] > 0.5) & (mask[:, :, pp_i] > 0.5)

    # impairment where observed and either <2
    impair = m & ((lt < 2.0) | (pp < 2.0))
    first_imp = np.full((grid.shape[0],), -1, dtype=np.int32)
    for i in range(28):
        hit = (first_imp < 0) & impair[:, i]
        first_imp[hit] = i

    # last intact index where both observed and both == 2, but only above first impairment if exists
    last_intact = np.full((grid.shape[0],), -1, dtype=np.int32)
    intact = m & (lt == 2.0) & (pp == 2.0)
    for i in range(28):
        last_intact[intact[:, i]] = i

    # sensory level proxy: immediately above first impairment if impairment exists and there is intact above
    level = np.where(first_imp > 0, first_imp - 1, last_intact)
    # if first_imp == 0, keep last_intact (could be -1)
    level = np.where(first_imp == 0, last_intact, level)
    return level.astype(np.int32), first_imp.astype(np.int32)


def add_isncsci_features_t1(X: pd.DataFrame, *, motor_cols: List[str], meta_cols: List[str]) -> pd.DataFrame:
    """
    Deterministic domain features for Track 1.
    Works off whatever sensory columns exist in X (suffix parse), plus motor/meta/time if present.
    Returns a NEW dataframe with extra numeric columns.
    """
    X = X.copy()
    grid, mask = _build_grid_and_mask(X)

    # Coverage / stop-depth (captures E-ISNCSCI stopping behavior)
    idx = np.arange(28, dtype=np.int32)[None, :]
    for si, suf in enumerate(SUFFIXES_4):
        m = mask[:, :, si]
        obs_count = m.sum(axis=1)
        X[f"feat__obs_count_{suf}"] = obs_count.astype(np.float32)
        X[f"feat__obs_frac_{suf}"] = (obs_count / 28.0).astype(np.float32)
        deepest = np.where(m > 0.5, idx, -1).max(axis=1)
        X[f"feat__deepest_obs_idx_{suf}"] = deepest.astype(np.float32)

        vals = grid[:, :, si]
        # counts by value among observed
        X[f"feat__n0_{suf}"] = (((vals == 0.0) & (m > 0.5)).sum(axis=1)).astype(np.float32)
        X[f"feat__n1_{suf}"] = (((vals == 1.0) & (m > 0.5)).sum(axis=1)).astype(np.float32)
        X[f"feat__n2_{suf}"] = (((vals == 2.0) & (m > 0.5)).sum(axis=1)).astype(np.float32)
        mean, std = _safe_mean_std(vals, m)
        X[f"feat__mean_{suf}"] = mean
        X[f"feat__std_{suf}"] = std

    # overall sensory coverage
    m_all = mask.sum(axis=(1, 2))
    X["feat__sens_obs_total"] = m_all.astype(np.float32)
    X["feat__sens_obs_frac_total"] = (m_all / (28.0 * 4.0)).astype(np.float32)

    # Sensory level proxies (left/right)
    lvl_L, first_imp_L = _sensory_level_proxy(grid, mask, "L")
    lvl_R, first_imp_R = _sensory_level_proxy(grid, mask, "R")
    X["feat__sens_level_L_idx"] = lvl_L.astype(np.float32)
    X["feat__sens_level_R_idx"] = lvl_R.astype(np.float32)
    X["feat__first_impair_L_idx"] = first_imp_L.astype(np.float32)
    X["feat__first_impair_R_idx"] = first_imp_R.astype(np.float32)

    # Symmetry + modality coupling (cheap, robust)
    def _mean_absdiff(si_a: int, si_b: int, name: str) -> None:
        a = grid[:, :, si_a]
        b = grid[:, :, si_b]
        m = (mask[:, :, si_a] > 0.5) & (mask[:, :, si_b] > 0.5)
        cnt = m.sum(axis=1)
        cnt_safe = np.where(cnt > 0, cnt, 1.0)
        diff = np.abs(a - b) * m
        X[name] = (diff.sum(axis=1) / cnt_safe).astype(np.float32)

    _mean_absdiff(0, 1, "feat__lr_absdiff_lt_mean")
    _mean_absdiff(2, 3, "feat__lr_absdiff_pp_mean")
    _mean_absdiff(0, 2, "feat__lt_pp_absdiff_left_mean")
    _mean_absdiff(1, 3, "feat__lt_pp_absdiff_right_mean")

    # Sacral sparing proxies (from observed S4-5 sensory)
    def _s45_present(cols: List[str]) -> np.ndarray:
        present = np.zeros((len(X),), dtype=bool)
        for c in cols:
            if c in X.columns:
                v = X[c].to_numpy(dtype=np.float32)
                present |= (~pd.isna(v)) & (v > 0.0)
        return present.astype(np.float32)

    X["feat__s45_lt_present"] = _s45_present(["s45ltl", "s45ltr"])
    X["feat__s45_pp_present"] = _s45_present(["s45ppl", "s45ppr"])
    X["feat__sacral_sparing_proxy"] = np.maximum(
        X["feat__s45_lt_present"].to_numpy(),
        X["feat__s45_pp_present"].to_numpy(),
    ).astype(np.float32)

    # Motor aggregates (UEMS/LEMS-style totals are standard)
    motor_cols_in = [c for c in motor_cols if c in X.columns]
    if motor_cols_in:
        M = X[motor_cols_in].to_numpy(dtype=np.float32)
        m = ~np.isnan(M)
        cnt = m.sum(axis=1).astype(np.float32)
        cnt_safe = np.where(cnt > 0, cnt, 1.0)
        M0 = np.nan_to_num(M, nan=0.0)

        X["feat__motor_obs_count"] = cnt
        X["feat__motor_missing_frac"] = (1.0 - cnt / float(len(motor_cols_in))).astype(np.float32)
        X["feat__motor_sum"] = (M0.sum(axis=1)).astype(np.float32)
        X["feat__motor_mean"] = (M0.sum(axis=1) / cnt_safe).astype(np.float32)
        X["feat__motor_ge3_count"] = ((M0 >= 3.0) & m).sum(axis=1).astype(np.float32)
        X["feat__motor_ge4_count"] = ((M0 >= 4.0) & m).sum(axis=1).astype(np.float32)
    else:
        X["feat__motor_obs_count"] = 0.0
        X["feat__motor_missing_frac"] = 1.0
        X["feat__motor_sum"] = 0.0
        X["feat__motor_mean"] = 0.0
        X["feat__motor_ge3_count"] = 0.0
        X["feat__motor_ge4_count"] = 0.0

    # Time transforms (if present)
    if "time" in X.columns and pd.api.types.is_numeric_dtype(X["time"]):
        t = X["time"].to_numpy(dtype=np.float32)
        X["feat__time_log1p"] = np.log1p(np.maximum(t, 0.0)).astype(np.float32)
        # coarse bins (numeric) to capture stage effects without high-cardinality categories
        X["feat__time_bin"] = np.digitize(t, bins=[1, 7, 30, 90, 180, 365]).astype(np.float32)

    # Basic meta passthrough counts
    meta_cols_in = [c for c in meta_cols if c in X.columns]
    if meta_cols_in:
        meta_miss = X[meta_cols_in].isna().mean(axis=1).to_numpy(dtype=np.float32)
        X["feat__meta_missing_frac"] = meta_miss
    else:
        X["feat__meta_missing_frac"] = 1.0

    return X
