#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from asia2026.data import load_track
from asia2026.metrics import compute_metrics
from asia2026.tabpfn_model_t1_discrete import tabpfn_predict_multioutput_t1_discrete
from asia2026.utils import RunConfig, ensure_dir, make_run_id, utc_now_iso, write_json

BAG_SPLITS = 5
BAG_SEED = 42


def _set_single_thread_env() -> None:
    os.environ.setdefault("TABPFN_NUM_WORKERS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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
    _set_single_thread_env()
    track = 1
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

    method = "tabpfn_25_t1safe_discrete_bag5"
    run_id = make_run_id(prefix=f"t1_{method}")
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
        notes="t1-safe tabpfn discrete bag5: classifier-as-regressor for sensory targets",
    )
    write_json(out_dir / "config.json", {"run_id": run_id, "created_utc": utc_now_iso(), **cfg.to_dict()})

    def predict(_Xtr: pd.DataFrame, _ytr: pd.DataFrame, _X: pd.DataFrame, apply_copy_through: bool) -> np.ndarray:
        copy_cols = [c for c in target_cols if c in _X.columns and not _X[c].isna().any()]
        pred = tabpfn_predict_multioutput_t1_discrete(
            _Xtr,
            _ytr[target_cols],
            _X,
            target_cols=target_cols,
            sensory_target_cols=sensory_cols,
            copy_through_cols=copy_cols,
            max_train_samples=None,
            seed=seed,
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

    for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
        Xtr_f = Xtr.iloc[tr_idx].reset_index(drop=True)
        ytr_f = ytr.iloc[tr_idx].reset_index(drop=True)
        Xva_f = Xtr.iloc[va_idx].reset_index(drop=True)
        yva_f = ytr.iloc[va_idx].reset_index(drop=True)

        pred_va = predict(Xtr_f, ytr_f, Xva_f, apply_copy_through=True)
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

        pred_te = predict(Xtr_f, ytr_f, Xte, apply_copy_through=False)
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

    oof_filled_rows = np.isfinite(oof).all(axis=1)
    if not oof_mask.all() or not oof_filled_rows.all():
        missing = int((~oof_filled_rows).sum())
        raise RuntimeError(f"Missing OOF predictions for {missing} rows")

    imputed_mask = _imputed_mask(Xtr, target_cols)
    sensory_idx = [i for i, c in enumerate(target_cols) if c in set(sensory_cols)]
    imputed_all = int(imputed_mask.sum())
    imputed_sensory = int(imputed_mask[:, sensory_idx].sum()) if sensory_idx else 0
    print(f"[t1safe_discrete_bag5] oof filled rows: {int(oof_filled_rows.sum())}/{len(oof_filled_rows)}")
    print(f"[t1safe_discrete_bag5] imputed cells (all targets): {imputed_all}")
    print(f"[t1safe_discrete_bag5] imputed cells (sensory targets): {imputed_sensory}")

    overall_cv = compute_metrics(
        ytr[target_cols].to_numpy(),
        oof,
        target_cols,
        sensory_cols,
        features=Xtr,
    )
    write_json(out_dir / "cv_metrics.json", {"overall": overall_cv, "folds": fold_metrics})

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
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
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
    print(f"[asia2026 t1 safe discrete bag5] done -> {out_dir}")


if __name__ == "__main__":
    main()
