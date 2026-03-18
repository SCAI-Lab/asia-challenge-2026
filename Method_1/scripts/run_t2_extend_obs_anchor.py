#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from utils.utils import ensure_dir, make_run_id, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)

TRACK2_ROOT = Path(__file__).resolve().parents[2]
FILES_DIR = TRACK2_ROOT / "files"
DEFAULT_BASELINE = FILES_DIR / "anchor_correction__t2_anchor_correction__20260315_222231__jobnojid__4chrmkaj.csv"
DEFAULT_FEATURES_TEST = TRACK2_ROOT / "data/features_test_2.csv"
DEFAULT_FEATURES_TRAIN = TRACK2_ROOT / "data/features_train_2.csv"
DEFAULT_LABELS_TRAIN = TRACK2_ROOT / "data/labels_train_2.csv"
DEFAULT_RUN_ROOT = TRACK2_ROOT / "runs"

RULES: List[Tuple[str, str, float]] = [
    ("c5ltl", "c5ltr", 0.10),
    ("c6ltl", "c6ltr", 0.16),
    ("c6ppr", "c6ltr", 0.12),
    ("c2ppl", "c2ltr", 0.18),
    ("c3ppl", "c3ltr", 0.14),
    ("c4ppl", "c4ltr", 0.12),
    ("c6ppl", "c6ltr", 0.08),
    ("c7ppl", "c7ltr", 0.10),
    ("c8ppl", "c8ltr", 0.16),
    ("t1ppl", "t1ltr", 0.16),
    ("t2ppl", "t2ltr", 0.16),
    ("t3ppl", "t3ltr", 0.08),
]

MARGIN = 0.02


def _clip_submission(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if c == "ID":
            continue
        if c == "anyana":
            out[c] = out[c].clip(0.0, 1.0)
        else:
            out[c] = out[c].clip(0.0, 2.0)
    return out


def _apply_copy_through(df_pred: pd.DataFrame, features_test: pd.DataFrame) -> pd.DataFrame:
    out = df_pred.copy()
    for c in out.columns:
        if c == "ID":
            continue
        if c in features_test.columns:
            obs = features_test[c]
            m = obs.notna()
            if m.any():
                out.loc[m, c] = obs.loc[m].astype(float).values
    return out


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
    ap.add_argument("--base-cv", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--features-test", type=Path, default=DEFAULT_FEATURES_TEST)
    ap.add_argument("--features-train", type=Path, default=DEFAULT_FEATURES_TRAIN)
    ap.add_argument("--labels-train", type=Path, default=DEFAULT_LABELS_TRAIN)
    ap.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = ap.parse_args()

    pred = pd.read_csv(args.base_cv).sort_values("ID").reset_index(drop=True)
    Xte = pd.read_csv(args.features_test).sort_values("ID").reset_index(drop=True)
    Xtr = pd.read_csv(args.features_train).sort_values("ID").reset_index(drop=True)
    ytr = pd.read_csv(args.labels_train).sort_values("ID").reset_index(drop=True)

    base = pred.copy()
    pred = _clip_submission(_apply_copy_through(pred, Xte))

    maps = {}
    for target, source, _ in RULES:
        if target not in ytr.columns or source not in Xtr.columns:
            continue
        d = pd.DataFrame({"y": ytr[target], "x": Xtr[source]}).dropna()
        maps[(target, source)] = d.groupby("x")["y"].mean().to_dict()

    applied_rules: List[Tuple[str, str, float]] = []
    for target, source, w in RULES:
        if target not in pred.columns or source not in Xte.columns:
            continue
        mapping = maps.get((target, source))
        if not mapping:
            continue

        m = Xte[target].isna() & Xte[source].notna() & (Xte[source] >= 1)
        if not m.any():
            continue

        src = Xte.loc[m, source].astype(float)
        anchor = src.map(mapping).astype(float)
        cur = pred.loc[m, target].astype(float)

        need = cur < (anchor - MARGIN)
        if need.any():
            idx = cur.index[need]
            src_need = src.loc[idx]
            anchor_need = anchor.loc[idx]
            cur_need = cur.loc[idx]
            eff_w = np.where(src_need.to_numpy() >= 2.0, w, 0.5 * w)
            pred.loc[idx, target] = cur_need.to_numpy() + eff_w * (
                anchor_need.to_numpy() - cur_need.to_numpy()
            )
            applied_rules.append((target, source, w))

    pred = _clip_submission(_apply_copy_through(pred, Xte))
    target_cols = [c for c in pred.columns if c != "ID"]
    pred, nan_after = _ensure_no_nans(pred, base, target_cols)
    if nan_after:
        raise RuntimeError(f"NaNs remain after correction: {nan_after}")

    method = "t2_extend_obs_anchor"
    run_id = make_run_id(prefix=method)
    out_dir = ensure_dir(Path(args.run_root) / run_id)
    out_csv = out_dir / "predictions_test.csv"
    pred.to_csv(out_csv, index=False)

    labeled_csv = FILES_DIR / f"predictions_test__{method}__{run_id}.csv"
    ensure_dir(labeled_csv.parent)
    pred.to_csv(labeled_csv, index=False)

    run_summary = {
        "method": method,
        "created_at": utc_now_iso(),
        "base_csv": str(args.base_cv),
        "features_test": str(args.features_test),
        "features_train": str(args.features_train),
        "labels_train": str(args.labels_train),
        "submission_csv": str(out_csv),
        "labeled_csv": str(labeled_csv),
        "applied_rules": applied_rules,
        "nan_after_correction": nan_after,
        "notes": "Extend observed anchors with upward-only correction and margin.",
    }
    write_json(out_dir / "run_summary.json", run_summary)
    LOGGER.info("Wrote %s", out_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
