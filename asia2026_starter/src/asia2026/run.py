from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from asia2026.baselines import (
    baseline_knn15,
    baseline_strat_time_mean,
    baseline_time_mean,
)
from asia2026.data import load_track
from asia2026.metrics import compute_metrics
from asia2026.tabpfn_model import tabpfn_predict_multioutput
from asia2026.utils import RunConfig, ensure_dir, make_run_id, utc_now_iso, write_json


def _clip_predictions(pred: np.ndarray, target_cols: List[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    # sensory scores are [0,2]; motor are not in labels; anyana is [0,1]
    for j, c in enumerate(target_cols):
        if c == "anyana":
            pred[:, j] = np.clip(pred[:, j], 0.0, 1.0)
        else:
            pred[:, j] = np.clip(pred[:, j], 0.0, 2.0)
    return pred


def _write_submission(sample_sub: pd.DataFrame, ids: pd.Series, pred: np.ndarray, target_cols: List[str], out_csv: Path) -> None:
    sub = sample_sub.copy()
    sub["ID"] = ids.values
    for j, c in enumerate(target_cols):
        sub[c] = pred[:, j]
    ensure_dir(out_csv.parent)
    sub.to_csv(out_csv, index=False)


def run_one(
    track: int,
    method: str,
    data_root: str,
    run_root: str,
    seed: int,
    do_cv: bool,
    n_splits: int,
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
    )
    write_json(out_dir / "config.json", {"run_id": run_id, "created_utc": utc_now_iso(), **cfg.to_dict()})

    # --- choose method ---
    def predict(_Xtr: pd.DataFrame, _ytr: pd.DataFrame, _X: pd.DataFrame) -> np.ndarray:
        if method == "baseline_time_mean":
            return baseline_time_mean(_Xtr, _ytr, _X, target_cols)
        if method == "baseline_strat_time_mean":
            strat_cols = ["bmi_category", "age_category", "sexcd", "tx1_r"]
            strat_cols = [c for c in strat_cols if c in _Xtr.columns]
            return baseline_strat_time_mean(_Xtr, _ytr, _X, target_cols, strat_cols=strat_cols)
        if method == "baseline_knn15":
            # fallback = time-mean
            fallback = baseline_time_mean(_Xtr, _ytr, _X, target_cols)
            # KNN on motor + metadata
            knn_cols = []
            # motor
            knn_cols += [c for c in data.motor_cols if c in _Xtr.columns]
            # include baseline motor for Track 2 (prefixed w1_)
            if track == 2:
                knn_cols += [c for c in _Xtr.columns if c.startswith("w1_") and c[3:] in data.motor_cols]
            # metadata
            knn_cols += [c for c in data.meta_cols if c in _Xtr.columns]
            knn_cols = list(dict.fromkeys(knn_cols))
            always15 = [c for c in data.always_missing_targets if c in target_cols]
            return baseline_knn15(
                _Xtr,
                _ytr,
                _X,
                target_cols=target_cols,
                always_missing_targets=always15,
                knn_feature_cols=knn_cols,
                time_fallback=fallback,
            )
        if method == "tabpfn_25":
            # columns that are fully observed in X can be skipped (we'll copy-through anyway)
            copy_cols = [c for c in target_cols if c in _X.columns and not _X[c].isna().any()]
            pred = tabpfn_predict_multioutput(
                _Xtr,
                _ytr[target_cols],
                _X,
                target_cols=target_cols,
                copy_through_cols=copy_cols,
                max_train_samples=None,
                seed=seed,
            )
            # if we skipped some columns, they are NaN in pred; fill with 0 before copy-through
            pred = np.nan_to_num(pred, nan=0.0)
            # enforce copy-through for observed
            for j, c in enumerate(target_cols):
                if c in _X.columns:
                    obs = _X[c].to_numpy()
                    m = ~pd.isna(obs)
                    pred[m, j] = obs[m]
            return pred
        raise ValueError(f"Unknown method: {method}")

    # --- CV (optional) ---
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

    # --- train-on-full -> predict test ---
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
    p.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["baseline_time_mean", "baseline_strat_time_mean", "baseline_knn15", "tabpfn_25"],
    )
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do-cv", type=int, default=1)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--limit-targets", type=int, default=None)
    args = p.parse_args()

    out_dir = run_one(
        track=args.track,
        method=args.method,
        data_root=args.data_root,
        run_root=args.run_root,
        seed=args.seed,
        do_cv=bool(args.do_cv),
        n_splits=args.n_splits,
        limit_rows=args.limit_rows,
        limit_targets=args.limit_targets,
    )
    print(f"[asia2026] done -> {out_dir}")


if __name__ == "__main__":
    main()
