#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.data import load_track
from utils.utils import ensure_dir, make_run_id, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)

TRACK2_ROOT = Path(__file__).resolve().parents[2]
FILES_DIR = TRACK2_ROOT / "files"
DEFAULT_BASELINE = FILES_DIR / "t2_tabpfn_25_discrete_bag5__20260214_014329__job57139435__on5h0rmx.csv"
DEFAULT_DATA_ROOT = TRACK2_ROOT / "data"
DEFAULT_RUN_ROOT = TRACK2_ROOT / "runs"


def _parse_level(col: str) -> Tuple[str, str] | None:
    for suffix in ("ltl", "ltr", "ppl", "ppr"):
        if col.endswith(suffix):
            return col[: -len(suffix)], suffix
    return None


def _compute_p_equal(a: pd.Series, b: pd.Series) -> float:
    mask = ~(a.isna() | b.isna())
    if not mask.any():
        return float("nan")
    return float((a[mask].to_numpy() == b[mask].to_numpy()).mean())


def _clip_predictions(pred: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    pred = pred.copy()
    for col in target_cols:
        if col == "anyana":
            pred[col] = pred[col].clip(0.0, 1.0)
        else:
            pred[col] = pred[col].clip(0.0, 2.0)
    return pred


def _apply_copy_through(pred: pd.DataFrame, features: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    pred = pred.copy()
    for col in target_cols:
        if col not in features.columns:
            continue
        obs = features[col]
        m = ~obs.isna()
        if m.any():
            pred.loc[m, col] = obs[m]
    return pred


def _build_pairs(target_cols: List[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    levels = sorted({lvl for col in target_cols if (parsed := _parse_level(col)) for lvl, _ in [parsed]})
    lr_pairs: List[Tuple[str, str]] = []
    ltpp_pairs: List[Tuple[str, str]] = []
    for lvl in levels:
        ltl, ltr = f"{lvl}ltl", f"{lvl}ltr"
        ppl, ppr = f"{lvl}ppl", f"{lvl}ppr"
        if ltl in target_cols and ltr in target_cols:
            lr_pairs.append((ltl, ltr))
        if ppl in target_cols and ppr in target_cols:
            lr_pairs.append((ppl, ppr))
        if ltl in target_cols and ppl in target_cols:
            ltpp_pairs.append((ltl, ppl))
        if ltr in target_cols and ppr in target_cols:
            ltpp_pairs.append((ltr, ppr))
    return lr_pairs, ltpp_pairs


def _apply_pairwise_shrink(
    pred: pd.DataFrame,
    features_test: pd.DataFrame,
    labels_train: pd.DataFrame,
    target_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], List[Tuple[str, str]], List[Tuple[str, str]]]:
    pred = pred.copy()
    miss_rate = features_test[target_cols].isna().mean()
    lr_pairs, ltpp_pairs = _build_pairs(target_cols)

    w_lr_map: Dict[str, float] = {}
    applied_lr: List[Tuple[str, str]] = []
    for left, right in lr_pairs:
        p_equal = _compute_p_equal(labels_train[left], labels_train[right])
        w_lr = 0.0 if np.isnan(p_equal) else float(np.clip((p_equal - 0.85) / 0.15, 0.0, 1.0) * 0.25)
        w_lr_map[f"{left}|{right}"] = w_lr
        if w_lr == 0.0:
            continue
        if miss_rate[left] < 0.85 or miss_rate[right] < 0.85:
            continue
        mask = features_test[left].isna() & features_test[right].isna()
        if not mask.any():
            continue
        avg = 0.5 * (pred.loc[mask, left] + pred.loc[mask, right])
        pred.loc[mask, left] = (1.0 - w_lr) * pred.loc[mask, left] + w_lr * avg
        pred.loc[mask, right] = (1.0 - w_lr) * pred.loc[mask, right] + w_lr * avg
        applied_lr.append((left, right))

    w_ltpp_map: Dict[str, float] = {}
    applied_ltpp: List[Tuple[str, str]] = []
    for lt_col, pp_col in ltpp_pairs:
        p_equal = _compute_p_equal(labels_train[lt_col], labels_train[pp_col])
        w_ltpp = 0.0 if np.isnan(p_equal) else float(np.clip((p_equal - 0.82) / 0.18, 0.0, 1.0) * 0.15)
        w_ltpp_map[f"{lt_col}|{pp_col}"] = w_ltpp
        if w_ltpp == 0.0:
            continue
        if miss_rate[lt_col] < 0.85 or miss_rate[pp_col] < 0.85:
            continue
        mask = features_test[lt_col].isna() & features_test[pp_col].isna()
        if not mask.any():
            continue
        avg = 0.5 * (pred.loc[mask, lt_col] + pred.loc[mask, pp_col])
        pred.loc[mask, lt_col] = (1.0 - w_ltpp) * pred.loc[mask, lt_col] + w_ltpp * avg
        pred.loc[mask, pp_col] = (1.0 - w_ltpp) * pred.loc[mask, pp_col] + w_ltpp * avg
        applied_ltpp.append((lt_col, pp_col))

    return pred, w_lr_map, w_ltpp_map, applied_lr, applied_ltpp


def _ensure_no_nans(pred: pd.DataFrame, base: pd.DataFrame, target_cols: List[str]) -> Tuple[pd.DataFrame, int]:
    pred = pred.copy()
    nan_before = int(pred[target_cols].isna().sum().sum())
    if nan_before == 0:
        return pred, 0
    pred[target_cols] = pred[target_cols].fillna(base[target_cols])
    remaining = pred[target_cols].isna()
    if remaining.any().any():
        for col in target_cols:
            if remaining[col].any():
                col_median = float(base[col].median(skipna=True))
                fill_val = col_median if np.isfinite(col_median) else 0.0
                pred.loc[remaining[col], col] = fill_val
    nan_after = int(pred[target_cols].isna().sum().sum())
    return pred, nan_after


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = ap.parse_args()

    data = load_track(track=2, data_root=args.data_root)
    target_cols = data.target_cols

    base = pd.read_csv(args.base_csv)
    if "ID" not in base.columns:
        raise ValueError(f"Missing ID column in {args.base_csv}")
    missing_cols = [c for c in target_cols if c not in base.columns]
    if missing_cols:
        raise ValueError(f"Base csv missing columns: {missing_cols}")

    features_test = data.X_test.set_index("ID")
    base = base.set_index("ID")
    base = base.loc[features_test.index]
    pred = base[target_cols].copy()

    pred, w_lr_map, w_ltpp_map, applied_lr, applied_ltpp = _apply_pairwise_shrink(
        pred=pred,
        features_test=features_test,
        labels_train=data.y_train,
        target_cols=target_cols,
    )

    pred = _clip_predictions(pred, target_cols)
    pred = _apply_copy_through(pred, features_test, target_cols)
    pred, nan_after = _ensure_no_nans(pred, base, target_cols)
    if nan_after:
        raise RuntimeError(f"NaNs remain after correction: {nan_after}")

    method = "t2_hedge_pairwise_shrink"
    run_id = make_run_id(prefix=method)
    out_dir = ensure_dir(Path(args.run_root) / run_id)
    out_csv = out_dir / "predictions_test.csv"

    out = data.sample_submission.copy()
    out["ID"] = features_test.index.values
    for col in target_cols:
        out[col] = pred[col].values
    out.to_csv(out_csv, index=False)

    labeled_csv = FILES_DIR / f"predictions_test__{method}__{run_id}.csv"
    ensure_dir(labeled_csv.parent)
    out.to_csv(labeled_csv, index=False)

    run_summary = {
        "method": method,
        "created_at": utc_now_iso(),
        "base_csv": str(args.base_csv),
        "submission_csv": str(out_csv),
        "labeled_csv": str(labeled_csv),
        "lr_weight_map": w_lr_map,
        "ltpp_weight_map": w_ltpp_map,
        "applied_lr_pairs": applied_lr,
        "applied_ltpp_pairs": applied_ltpp,
        "nan_after_correction": nan_after,
        "notes": "Track2 pairwise shrink with gating on test missingness >=0.85 + copy-through.",
    }
    write_json(out_dir / "run_summary.json", run_summary)
    LOGGER.info("Wrote %s", out_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
