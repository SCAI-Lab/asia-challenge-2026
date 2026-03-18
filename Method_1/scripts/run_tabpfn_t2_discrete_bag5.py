#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tabpfn import TabPFNClassifier, TabPFNRegressor

from utils.data import load_track
from utils.metrics import compute_metrics
from utils.utils import RunConfig, ensure_dir, make_run_id, utc_now_iso, write_json

BAG_SPLITS = 5
BAG_SEED = 42
DEFAULT_N_ESTIMATORS = 8
TRACK2_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = TRACK2_ROOT / "data"
DEFAULT_RUN_ROOT = TRACK2_ROOT / "runs"
LOGGER = None


def _set_single_thread_env() -> None:
    os.environ.setdefault("TABPFN_NUM_WORKERS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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


def _init_tabpfn_classifier(device: str) -> TabPFNClassifier:
    extra = {
        "n_jobs": 1,
        "n_estimators": DEFAULT_N_ESTIMATORS,
    }
    sig = inspect.signature(TabPFNClassifier)
    kwargs = {k: v for k, v in extra.items() if k in sig.parameters}
    return TabPFNClassifier(device=device, ignore_pretraining_limits=True, **kwargs)


def _init_tabpfn_regressor(device: str) -> TabPFNRegressor:
    extra = {
        "n_jobs": 1,
        "n_estimators": DEFAULT_N_ESTIMATORS,
    }
    sig = inspect.signature(TabPFNRegressor)
    kwargs = {k: v for k, v in extra.items() if k in sig.parameters}
    return TabPFNRegressor(device=device, ignore_pretraining_limits=True, **kwargs)


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


def tabpfn_predict_multioutput_discrete_bag(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X: pd.DataFrame,
    target_cols: List[str],
    sensory_target_cols: List[str],
    copy_through_cols: Optional[List[str]] = None,
    max_train_samples: Optional[int] = None,
    seed: int = 42,
    progress_desc: Optional[str] = None,
) -> np.ndarray:
    """TabPFN classifier-as-regressor for sensory targets and anyana; regressor for others."""
    rng = np.random.default_rng(seed)
    device = "cuda"
    _set_single_thread_env()

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

    sensory_targets = set(sensory_target_cols)
    sensory_classes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    anyana_classes = np.array([0.0, 1.0], dtype=np.float32)
    target_iter = tqdm(
        list(enumerate(target_cols)),
        desc=progress_desc or "targets",
        leave=False,
    )
    for j, col in target_iter:
        is_sensory = col in sensory_targets
        is_anyana = col == "anyana"
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

        if is_anyana:
            y = y_fit_full[col].to_numpy()
            y_int = np.rint(y).astype(np.int64)
            model = _init_tabpfn_classifier(device)
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
            model = _init_tabpfn_classifier(device)
            model.fit(Xtr_use, y_int)
            proba = model.predict_proba(Xte_use)
            out[:, j] = _expected_from_proba_fixed(
                proba,
                model.classes_,
                sensory_classes,
            ).astype(np.float32)
        else:
            y = y_fit_full[col].to_numpy(dtype=np.float32)
            model = _init_tabpfn_regressor(device)
            model.fit(Xtr_use, y)
            out[:, j] = model.predict(Xte_use).astype(np.float32)

    return out


def _clip_predictions(pred: np.ndarray, target_cols: List[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    for j, c in enumerate(target_cols):
        if c == "anyana":
            pred[:, j] = np.clip(pred[:, j], 0.0, 1.0)
        else:
            pred[:, j] = np.clip(pred[:, j], 0.0, 2.0)
    return pred


def _apply_copy_through(pred: np.ndarray, features: pd.DataFrame, target_cols: List[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    for j, c in enumerate(target_cols):
        if c in features.columns:
            obs = features[c].to_numpy()
            m = ~pd.isna(obs)
            pred[m, j] = obs[m]
    return pred


def _imputed_mask(features: pd.DataFrame, target_cols: List[str]) -> np.ndarray:
    masks = []
    for c in target_cols:
        if c in features.columns:
            masks.append(features[c].isna().to_numpy())
        else:
            masks.append(np.zeros(len(features), dtype=bool))
    return np.column_stack(masks)


def _write_submission(sample_sub: pd.DataFrame, ids: pd.Series, pred: np.ndarray, target_cols: List[str], out_csv: Path) -> None:
    sub = sample_sub.copy()
    sub["ID"] = ids.values
    for j, c in enumerate(target_cols):
        sub[c] = pred[:, j]
    ensure_dir(out_csv.parent)
    sub.to_csv(out_csv, index=False)


def run_one(
    data_root: str,
    run_root: str,
    seed: int,
    limit_rows: Optional[int] = None,
    limit_targets: Optional[int] = None,
) -> Path:
    LOGGER.info("Loading track %s data from %s", 2, data_root)
    _set_single_thread_env()
    track = 2
    data = load_track(track, data_root)

    Xtr = data.X_train.copy()
    ytr = data.y_train.copy()
    Xte = data.X_test.copy()

    if limit_rows is not None:
        Xtr = Xtr.iloc[:limit_rows].reset_index(drop=True)
        ytr = ytr.iloc[:limit_rows].reset_index(drop=True)

    target_cols = data.target_cols
    sensory_cols = data.sensory_target_cols
    if limit_targets is not None:
        target_cols = target_cols[:limit_targets]
        sensory_cols = [c for c in sensory_cols if c in target_cols]

    LOGGER.info("Preparing run directory")
    method = "tabpfn_25_discrete_bag5"
    run_id = make_run_id(prefix=f"t2_{method}")
    out_dir = ensure_dir(Path(run_root) / run_id)

    cfg = RunConfig(
        track=track,
        method=method,
        data_root=data_root,
        run_root=run_root,
        seed=seed,
        do_cv=True,
        n_splits=BAG_SPLITS,
        limit_rows=limit_rows,
        limit_targets=limit_targets,
        notes="t2 tabpfn discrete bag5: classifier-as-regressor for sensory targets",
    )
    write_json(out_dir / "config.json", {"run_id": run_id, "created_utc": utc_now_iso(), **cfg.to_dict()})

    def predict(_Xtr: pd.DataFrame, _ytr: pd.DataFrame, _X: pd.DataFrame, apply_copy_through: bool, desc: str) -> np.ndarray:
        copy_cols = [c for c in target_cols if c in _X.columns and not _X[c].isna().any()]
        pred = tabpfn_predict_multioutput_discrete_bag(
            _Xtr,
            _ytr[target_cols],
            _X,
            target_cols=target_cols,
            sensory_target_cols=sensory_cols,
            copy_through_cols=copy_cols,
            max_train_samples=None,
            seed=seed,
            progress_desc=desc,
        )
        pred = np.nan_to_num(pred, nan=0.0)
        if apply_copy_through:
            pred = _apply_copy_through(pred, _X, target_cols)
        return pred

    kf = KFold(n_splits=BAG_SPLITS, shuffle=True, random_state=BAG_SEED)
    oof = np.zeros((len(Xtr), len(target_cols)), dtype=np.float32)
    oof_mask = np.zeros(len(Xtr), dtype=bool)
    test_sum = np.zeros((len(Xte), len(target_cols)), dtype=np.float32)
    fold_metrics = []

    LOGGER.info("Starting %s-fold CV", BAG_SPLITS)
    fold_iter = tqdm(list(enumerate(kf.split(Xtr))), desc="folds", leave=True)
    for fold, (tr_idx, va_idx) in fold_iter:
        LOGGER.info("Fold %s/%s", fold + 1, BAG_SPLITS)
        Xtr_f = Xtr.iloc[tr_idx].reset_index(drop=True)
        ytr_f = ytr.iloc[tr_idx].reset_index(drop=True)
        Xva_f = Xtr.iloc[va_idx].reset_index(drop=True)
        yva_f = ytr.iloc[va_idx].reset_index(drop=True)

        LOGGER.info("Fold %s: predicting validation", fold)
        pred_va = predict(Xtr_f, ytr_f, Xva_f, apply_copy_through=True, desc=f"fold{fold}-val")
        pred_va = _clip_predictions(pred_va, target_cols)
        oof[va_idx] = pred_va
        oof_mask[va_idx] = True
        m = compute_metrics(
            yva_f[target_cols].to_numpy(),
            pred_va,
            target_cols,
            sensory_cols,
            features=Xva_f,
        )
        m["fold"] = fold
        fold_metrics.append(m)

        LOGGER.info("Fold %s: predicting test", fold)
        pred_te = predict(Xtr_f, ytr_f, Xte, apply_copy_through=False, desc=f"fold{fold}-test")
        pred_te = _clip_predictions(pred_te, target_cols)
        test_sum += pred_te

        pred_te_out = _apply_copy_through(pred_te, Xte, target_cols)
        pred_te_out = _clip_predictions(pred_te_out, target_cols)
        _write_submission(
            data.sample_submission,
            ids=Xte["ID"],
            pred=pred_te_out,
            target_cols=target_cols,
            out_csv=out_dir / f"predictions_test_fold{fold}.csv",
        )

    LOGGER.info("Computing OOF metrics and summaries")
    oof_filled_rows = np.isfinite(oof).all(axis=1)
    if not oof_mask.all() or not oof_filled_rows.all():
        missing = int((~oof_filled_rows).sum())
        raise RuntimeError(f"Missing OOF predictions for {missing} rows")

    imputed_mask = _imputed_mask(Xtr, target_cols)
    sensory_idx = [i for i, c in enumerate(target_cols) if c in set(sensory_cols)]
    imputed_all = int(imputed_mask.sum())
    imputed_sensory = int(imputed_mask[:, sensory_idx].sum()) if sensory_idx else 0
    print(f"[t2_discrete_bag5] oof filled rows: {int(oof_filled_rows.sum())}/{len(oof_filled_rows)}")
    print(f"[t2_discrete_bag5] imputed cells (all targets): {imputed_all}")
    print(f"[t2_discrete_bag5] imputed cells (sensory targets): {imputed_sensory}")

    overall_cv = compute_metrics(
        ytr[target_cols].to_numpy(),
        oof,
        target_cols,
        sensory_cols,
        features=Xtr,
    )
    write_json(out_dir / "cv_metrics.json", {"overall": overall_cv, "folds": fold_metrics})

    LOGGER.info("Writing averaged test predictions")
    test_avg = test_sum / float(BAG_SPLITS)
    test_avg = _apply_copy_through(test_avg, Xte, target_cols)
    test_avg = _clip_predictions(test_avg, target_cols)

    _write_submission(
        data.sample_submission,
        ids=Xte["ID"],
        pred=test_avg,
        target_cols=target_cols,
        out_csv=out_dir / "predictions_test.csv",
    )

    write_json(
        out_dir / "run_summary.json",
        {
            "run_id": run_id,
            "finished_utc": utc_now_iso(),
            "track": track,
            "method": method,
            "cv_overall": overall_cv,
            "artifacts": {
                "submission_csv": "predictions_test.csv",
                "cv_metrics_json": "cv_metrics.json",
            },
        },
    )
    LOGGER.info("Run complete: %s", out_dir)
    return out_dir


def main() -> None:
    import logging

    global LOGGER
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    LOGGER = logging.getLogger("t2_discrete_bag5")

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--limit-targets", type=int, default=None)
    args = p.parse_args()

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        seed=args.seed,
        limit_rows=args.limit_rows,
        limit_targets=args.limit_targets,
    )
    print(f"[asia2026 t2 discrete bag5] done -> {out_dir}")


if __name__ == "__main__":
    main()
