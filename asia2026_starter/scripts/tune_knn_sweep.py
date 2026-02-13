#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from asia2026.baselines import apply_copy_through, baseline_time_mean
from asia2026.data import load_track
from asia2026.metrics import compute_metrics
from asia2026.utils import RunConfig, ensure_dir, make_run_id, utc_now_iso, write_json


def _clip_predictions(pred: np.ndarray, target_cols: List[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    for j, c in enumerate(target_cols):
        if c == "anyana":
            pred[:, j] = np.clip(pred[:, j], 0.0, 1.0)
        else:
            pred[:, j] = np.clip(pred[:, j], 0.0, 2.0)
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


def _write_submission(sample_sub: pd.DataFrame, ids: pd.Series, pred: np.ndarray, target_cols: List[str], out_csv: Path) -> None:
    sub = sample_sub.copy()
    sub["ID"] = ids.values
    for j, c in enumerate(target_cols):
        sub[c] = pred[:, j]
    ensure_dir(out_csv.parent)
    sub.to_csv(out_csv, index=False)


def _baseline_knn(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X: pd.DataFrame,
    target_cols: List[str],
    always_missing_targets: List[str],
    knn_feature_cols: List[str],
    time_fallback: np.ndarray,
    n_neighbors: int,
    weights: str,
) -> np.ndarray:
    out = np.asarray(time_fallback).copy()

    if not always_missing_targets:
        return apply_copy_through(X, out, target_cols)

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
            ("knn", KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights, n_jobs=-1)),
        ]
    )
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte).astype(np.float32)

    col2idx = {c: i for i, c in enumerate(target_cols)}
    for j, c in enumerate(always_missing_targets):
        out[:, col2idx[c]] = yhat[:, j]

    return apply_copy_through(X, out, target_cols)


def run_one(
    track: int,
    data_root: str,
    run_root: str,
    seed: int,
    do_cv: bool,
    n_splits: int,
    n_neighbors: int,
    weights: str,
    limit_rows: Optional[int] = None,
    limit_targets: Optional[int] = None,
) -> Path:
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

    method = f"knn_sweep_k{n_neighbors}_w{weights}"
    run_id = make_run_id(prefix=f"t{track}_{method}")
    out_dir = ensure_dir(Path(run_root) / run_id)

    cfg = RunConfig(
        track=track,
        method=method,
        data_root=data_root,
        run_root=run_root,
        seed=seed,
        do_cv=bool(do_cv),
        n_splits=n_splits,
        limit_rows=limit_rows,
        limit_targets=limit_targets,
        notes=f"k={n_neighbors}, weights={weights}",
    )
    write_json(out_dir / "config.json", {"run_id": run_id, "created_utc": utc_now_iso(), **cfg.to_dict()})

    def predict(_Xtr: pd.DataFrame, _ytr: pd.DataFrame, _X: pd.DataFrame) -> np.ndarray:
        fallback = baseline_time_mean(_Xtr, _ytr, _X, target_cols)
        knn_cols = []
        knn_cols += [c for c in data.motor_cols if c in _Xtr.columns]
        if track == 2:
            knn_cols += [c for c in _Xtr.columns if c.startswith("w1_") and c[3:] in data.motor_cols]
        knn_cols += [c for c in data.meta_cols if c in _Xtr.columns]
        knn_cols = list(dict.fromkeys(knn_cols))
        always15 = [c for c in data.always_missing_targets if c in target_cols]
        return _baseline_knn(
            _Xtr,
            _ytr,
            _X,
            target_cols=target_cols,
            always_missing_targets=always15,
            knn_feature_cols=knn_cols,
            time_fallback=fallback,
            n_neighbors=n_neighbors,
            weights=weights,
        )

    overall_cv = None
    if do_cv:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros((len(Xtr), len(target_cols)), dtype=np.float32)
        fold_metrics = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
            Xtr_f = Xtr.iloc[tr_idx].reset_index(drop=True)
            ytr_f = ytr.iloc[tr_idx].reset_index(drop=True)
            Xva_f = Xtr.iloc[va_idx].reset_index(drop=True)
            yva_f = ytr.iloc[va_idx].reset_index(drop=True)

            pred_va = predict(Xtr_f, ytr_f, Xva_f)
            pred_va = _clip_predictions(pred_va, target_cols)
            oof[va_idx] = pred_va
            m = compute_metrics(
                yva_f[target_cols].to_numpy(),
                pred_va,
                target_cols,
                sensory_cols,
                features=Xva_f,
            )
            m["fold"] = fold
            fold_metrics.append(m)

        overall_cv = compute_metrics(
            ytr[target_cols].to_numpy(),
            oof,
            target_cols,
            sensory_cols,
            features=Xtr,
        )
        write_json(out_dir / "cv_metrics.json", {"overall": overall_cv, "folds": fold_metrics})

    pred_test = predict(Xtr, ytr, Xte)
    pred_test = _clip_predictions(pred_test, target_cols)

    _write_submission(
        data.sample_submission,
        ids=Xte["ID"],
        pred=pred_test,
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
                "cv_metrics_json": "cv_metrics.json" if do_cv else None,
            },
        },
    )
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--track", type=int, required=True, choices=[1, 2])
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do-cv", type=int, default=1)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--limit-targets", type=int, default=None)
    p.add_argument("--k-list", type=str, default="1,3,5,7,11,15,25")
    p.add_argument("--weights", type=str, default="uniform,distance")
    args = p.parse_args()

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    weights_list = [x.strip() for x in args.weights.split(",") if x.strip()]

    for k in k_list:
        for w in weights_list:
            out_dir = run_one(
                track=args.track,
                data_root=args.data_root,
                run_root=args.run_root,
                seed=args.seed,
                do_cv=bool(args.do_cv),
                n_splits=args.n_splits,
                n_neighbors=k,
                weights=w,
                limit_rows=args.limit_rows,
                limit_targets=args.limit_targets,
            )
            print(f"[tune_knn_sweep] done -> {out_dir}")


if __name__ == "__main__":
    main()
