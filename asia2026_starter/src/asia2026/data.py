from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd


SENSORY_SUFFIXES = ("ltl", "ltr", "ppl", "ppr")


def _is_sensory_col(col: str) -> bool:
    return col.endswith(SENSORY_SUFFIXES)


def _infer_motor_cols(feature_cols: List[str], target_cols: List[str]) -> List[str]:
    """Motor cols are the non-ID, non-target, non-metadata cols (mostly 20 key muscles).

    We keep `vaccd`, `time`, `anyana` as non-motor special columns.
    """
    specials = {"ID", "time", "vaccd", "anyana"}
    return [c for c in feature_cols if c not in specials and (c not in target_cols) and not _is_sensory_col(c)]


@dataclass
class TrackData:
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_test: pd.DataFrame
    sample_submission: pd.DataFrame
    target_cols: List[str]
    sensory_target_cols: List[str]
    always_missing_targets: List[str]
    motor_cols: List[str]
    meta_cols: List[str]


def load_track(track: int, data_root: str) -> TrackData:
    """Load and merge features + metadata for the selected track."""
    root = Path(data_root)
    if track == 1:
        tdir = root / "track1"
        feats_tr = pd.read_csv(tdir / "features_train_1.csv")
        feats_te = pd.read_csv(tdir / "features_test_1.csv")
        meta_tr = pd.read_csv(tdir / "metadata_train_1.csv")
        meta_te = pd.read_csv(tdir / "metadata_test_1.csv")
        y_tr = pd.read_csv(tdir / "labels_train_1.csv")
        sub = pd.read_csv(tdir / "labels_test_1_dummy.csv")
    elif track == 2:
        tdir = root / "track2"
        feats_tr = pd.read_csv(tdir / "features_train_2.csv")
        feats_te = pd.read_csv(tdir / "features_test_2.csv")
        meta_tr = pd.read_csv(tdir / "metadata_train_2.csv")
        meta_te = pd.read_csv(tdir / "metadata_test_2.csv")
        y_tr = pd.read_csv(tdir / "labels_train_2.csv")
        sub = pd.read_csv(tdir / "labels_test_2_dummy.csv")
    else:
        raise ValueError(f"Unknown track: {track}")

    target_cols = [c for c in y_tr.columns if c != "ID"]
    sensory_target_cols = [c for c in target_cols if c != "anyana"]

    # merge
    X_train = feats_tr.merge(meta_tr, on="ID", how="left")
    X_test = feats_te.merge(meta_te, on="ID", how="left")
    y_train = y_tr.set_index("ID").loc[X_train["ID"], target_cols].reset_index(drop=True)

    # Identify always-missing targets in the *features* (these exist as targets).
    # These are the ones you asked to treat specially with KNN.
    miss_rate = feats_tr[target_cols].isna().mean()
    always_missing_targets = [c for c, r in miss_rate.items() if r >= 0.999999]

    meta_cols = [c for c in meta_tr.columns if c != "ID"]
    motor_cols = _infer_motor_cols(list(feats_tr.columns), target_cols)

    return TrackData(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        sample_submission=sub,
        target_cols=target_cols,
        sensory_target_cols=sensory_target_cols,
        always_missing_targets=always_missing_targets,
        motor_cols=motor_cols,
        meta_cols=meta_cols,
    )
