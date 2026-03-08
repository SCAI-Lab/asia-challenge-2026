from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

SUFFIXES_4 = ["ltl", "ltr", "ppl", "ppr"]


def _parse_target(col: str) -> Tuple[str, str] | None:
    cl = col.lower()
    if cl == "anyana":
        return None
    for suf in SUFFIXES_4:
        if cl.endswith(suf):
            derm = cl[:-len(suf)]
            return derm, suf
    return None


def apply_constraints_t1(
    pred: np.ndarray,
    X_features: pd.DataFrame,
    target_cols: List[str],
) -> np.ndarray:
    """
    Conservative constraints:
    1) Above first observed impairment on each side, clamp imputed sensory predictions upward toward intact.
    2) Mild symmetry shrink for L/R when both sides missing at a dermatome/channel and disagreement is extreme.
    """
    pred = np.asarray(pred).copy()

    # Access derived proxies (present after features are added)
    first_L = X_features.get("feat__first_impair_L_idx", pd.Series([-1] * len(X_features))).to_numpy(dtype=np.int32)
    first_R = X_features.get("feat__first_impair_R_idx", pd.Series([-1] * len(X_features))).to_numpy(dtype=np.int32)

    derm_order = [
        "c2", "c3", "c4", "c5", "c6", "c7", "c8",
        "t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9", "t10", "t11", "t12",
        "l1", "l2", "l3", "l4", "l5", "s1", "s2", "s3", "s45",
    ]
    derm2i = {d: i for i, d in enumerate(derm_order)}

    cols_info = []
    for j, c in enumerate(target_cols):
        parsed = _parse_target(c)
        if parsed is None:
            continue
        derm_raw, suf = parsed
        derm = (
            derm_raw.replace("s4_5", "s45")
            .replace("s4-5", "s45")
            .replace("s4/5", "s45")
            .replace("s4s5", "s45")
        )
        di = derm2i.get(derm)
        if di is None:
            continue
        side = "L" if suf in ("ltl", "ppl") else "R"
        mod = "LT" if suf in ("ltl", "ltr") else "PP"
        cols_info.append((j, c, di, side, mod))

    # 1) Above-first-impair clamp: only if first impairment exists and is not at top.
    for (j, c, di, side, _mod) in cols_info:
        if c not in X_features.columns:
            continue
        obs = X_features[c].to_numpy()
        miss = pd.isna(obs)
        if not miss.any():
            continue

        first = first_L if side == "L" else first_R
        apply = miss & (first >= 1) & (di < first)
        if apply.any():
            pred[apply, j] = np.maximum(pred[apply, j], 1.8)

    # 2) Symmetry shrink when both sides missing and disagreement extreme (>1.5)
    key_to_pair = {}
    for (j, c, di, side, mod) in cols_info:
        k = (di, mod)
        key_to_pair.setdefault(k, {})[side] = (j, c)
    for (_di, _mod), sides in key_to_pair.items():
        if "L" not in sides or "R" not in sides:
            continue
        jL, cL = sides["L"]
        jR, cR = sides["R"]
        if cL not in X_features.columns or cR not in X_features.columns:
            continue
        missL = pd.isna(X_features[cL].to_numpy())
        missR = pd.isna(X_features[cR].to_numpy())
        both = missL & missR
        if not both.any():
            continue
        diff = np.abs(pred[:, jL] - pred[:, jR])
        extreme = both & (diff > 1.5)
        if extreme.any():
            m = 0.5 * (pred[extreme, jL] + pred[extreme, jR])
            pred[extreme, jL] = m
            pred[extreme, jR] = m

    return pred
