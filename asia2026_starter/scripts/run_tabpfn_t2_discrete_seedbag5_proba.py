#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from tabpfn import TabPFNClassifier

from asia2026.data import load_track
from asia2026.eval import save_oof_npz
from asia2026.metrics import compute_metrics
from asia2026.metrics_weighted import compute_wrmse_imputed_only
from asia2026.tabpfn_model_discrete import build_preprocessor
from asia2026.utils import RunConfig, ensure_dir, make_run_id, utc_now_iso, write_json

SEEDS = [11, 22, 33, 44, 55]
CV_SPLITS = 5
CV_SEED = 42
DEFAULT_N_ESTIMATORS = 8
LOGGER = logging.getLogger(__name__)


def _set_single_thread_env() -> None:
    os.environ.setdefault("TABPFN_NUM_WORKERS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _init_tabpfn_classifier(device: str) -> TabPFNClassifier:
    extra = {
        "n_jobs": 1,
        "n_estimators": DEFAULT_N_ESTIMATORS,
    }
    sig = inspect.signature(TabPFNClassifier)
    kwargs = {k: v for k, v in extra.items() if k in sig.parameters}
    return TabPFNClassifier(device=device, ignore_pretraining_limits=True, **kwargs)


def _proba_to_fixed(proba: np.ndarray, classes: np.ndarray, target_classes: np.ndarray) -> np.ndarray:
    target_classes = target_classes.astype(np.int64)
    p_full = np.zeros((proba.shape[0], len(target_classes)), dtype=np.float32)
    class_to_idx = {int(c): i for i, c in enumerate(target_classes)}
    for col, cls in enumerate(classes):
        idx = class_to_idx.get(int(cls))
        if idx is not None:
            p_full[:, idx] = proba[:, col]
    return p_full


def _apply_copy_through(pred: np.ndarray, features: pd.DataFrame, target_cols: List[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    for j, c in enumerate(target_cols):
        if c in features.columns:
            obs = features[c].to_numpy()
            m = ~pd.isna(obs)
            pred[m, j] = obs[m]
    return pred


def _clip_predictions(pred: np.ndarray, target_cols: List[str]) -> np.ndarray:
    pred = np.asarray(pred).copy()
    for j, c in enumerate(target_cols):
        if c == "anyana":
            pred[:, j] = np.clip(pred[:, j], 0.0, 1.0)
        else:
            pred[:, j] = np.clip(pred[:, j], 0.0, 2.0)
    return pred


def _write_submission(
    sample_sub: pd.DataFrame,
    ids: pd.Series,
    pred: np.ndarray,
    target_cols: List[str],
    out_csv: Path,
) -> None:
    sub = sample_sub.copy()
    sub["ID"] = ids.values
    for j, c in enumerate(target_cols):
        sub[c] = pred[:, j]
    ensure_dir(out_csv.parent)
    sub.to_csv(out_csv, index=False)


def _preprocess(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    pre = build_preprocessor(Xtr)
    Xtr_all = np.asarray(pre.fit_transform(Xtr), dtype=np.float32)
    Xte_all = np.asarray(pre.transform(Xte), dtype=np.float32)
    Xtr_all = np.nan_to_num(Xtr_all, nan=0.0, posinf=0.0, neginf=0.0)
    Xte_all = np.nan_to_num(Xte_all, nan=0.0, posinf=0.0, neginf=0.0)
    feat_names = list(pre.get_feature_names_out())
    name_to_idx = {n: i for i, n in enumerate(feat_names)}
    return Xtr_all, Xte_all, name_to_idx


def _predict_seedbag_prob(
    Xtr: pd.DataFrame,
    ytr: pd.DataFrame,
    Xte: pd.DataFrame,
    target_cols: List[str],
    sensory_cols: List[str],
    seeds: List[int],
    log_prefix: str,
) -> np.ndarray:
    Xtr_all, Xte_all, name_to_idx = _preprocess(Xtr, Xte)

    sensory_targets = set(sensory_cols)
    sensory_classes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    anyana_classes = np.array([0.0, 1.0], dtype=np.float32)

    p_sum: Dict[str, np.ndarray] = {}
    for col in target_cols:
        n_classes = 2 if col == "anyana" else 3
        p_sum[col] = np.zeros((len(Xte_all), n_classes), dtype=np.float32)

    copy_cols = [c for c in target_cols if c in Xte.columns and not Xte[c].isna().any()]
    LOGGER.info("%sCopy-through targets: %s", log_prefix, len(copy_cols))

    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for col in target_cols:
            if col in copy_cols:
                continue

            is_anyana = col == "anyana"
            is_sensory = col in sensory_targets
            if not is_anyana and not is_sensory:
                raise ValueError(f"Unexpected non-sensory target in t2 discrete seedbag: {col}")

            leak_idx = name_to_idx.get(f"num__{col}")
            if leak_idx is not None:
                Xtr_use = Xtr_all.copy()
                Xte_use = Xte_all.copy()
                Xtr_use[:, leak_idx] = np.nan
                Xte_use[:, leak_idx] = np.nan
                Xtr_use = np.nan_to_num(Xtr_use, nan=0.0, posinf=0.0, neginf=0.0)
                Xte_use = np.nan_to_num(Xte_use, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                Xtr_use = Xtr_all
                Xte_use = Xte_all

            y = ytr[col].to_numpy()
            y_int = np.rint(y).astype(np.int64)
            model = _init_tabpfn_classifier("cuda")
            model.fit(Xtr_use, y_int)
            proba = model.predict_proba(Xte_use)

            target_classes = anyana_classes if is_anyana else sensory_classes
            p_sum[col] += _proba_to_fixed(proba, model.classes_, target_classes)

    out = np.zeros((len(Xte_all), len(target_cols)), dtype=np.float32)
    for j, col in enumerate(target_cols):
        if col in copy_cols:
            continue
        target_classes = anyana_classes if col == "anyana" else sensory_classes
        p_avg = p_sum[col] / float(len(seeds))
        out[:, j] = p_avg @ target_classes

    return out


def run_one(
    data_root: str,
    run_root: str,
    seed: int,
    do_cv: bool,
    n_splits: int,
    limit_rows: Optional[int] = None,
    limit_targets: Optional[int] = None,
    limit_test_rows: Optional[int] = None,
) -> Path:
    _set_single_thread_env()
    track = 2
    data = load_track(track, data_root)

    Xtr = data.X_train.copy()
    ytr = data.y_train.copy()
    Xte = data.X_test.copy()

    if limit_rows is not None:
        Xtr = Xtr.iloc[:limit_rows].reset_index(drop=True)
        ytr = ytr.iloc[:limit_rows].reset_index(drop=True)
    if limit_test_rows is not None:
        Xte = Xte.iloc[:limit_test_rows].reset_index(drop=True)

    target_cols = data.target_cols
    sensory_cols = data.sensory_target_cols
    if limit_targets is not None:
        target_cols = target_cols[:limit_targets]
        sensory_cols = [c for c in sensory_cols if c in target_cols]

    method = "t2_discrete_seedbag5_proba"
    run_id = make_run_id(prefix=method)
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
        notes="t2 tabpfn discrete seedbag5: full-train seed ensemble with proba averaging",
    )
    write_json(out_dir / "config.json", {"run_id": run_id, "created_utc": utc_now_iso(), **cfg.to_dict()})

    overall_cv = None
    weighted_cv = None
    if do_cv:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=CV_SEED)
        oof = np.zeros((len(Xtr), len(target_cols)), dtype=np.float32)
        fold_metrics = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(Xtr)):
            Xtr_f = Xtr.iloc[tr_idx].reset_index(drop=True)
            ytr_f = ytr.iloc[tr_idx].reset_index(drop=True)
            Xva_f = Xtr.iloc[va_idx].reset_index(drop=True)
            yva_f = ytr.iloc[va_idx].reset_index(drop=True)

            pred_va = _predict_seedbag_prob(
                Xtr_f,
                ytr_f,
                Xva_f,
                target_cols,
                sensory_cols,
                SEEDS,
                log_prefix=f" (fold={fold}) ",
            )
            pred_va = _apply_copy_through(pred_va, Xva_f, target_cols)
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
        weighted_cv = compute_wrmse_imputed_only(
            y_true=ytr,
            oof_pred=oof,
            X_train_features=Xtr,
            X_test_features=Xte,
            target_cols=target_cols,
        )
        write_json(
            out_dir / "weighted_oof.json",
            {
                "wrmse_imputed_only": weighted_cv.wrmse_imputed_only,
                "wrmse_all": weighted_cv.wrmse_all,
                "per_target_contrib": weighted_cv.per_target_contrib,
                "weights": weighted_cv.weights,
                "mse_per_target": weighted_cv.mse_per_target,
                "mse_per_target_all": weighted_cv.mse_per_target_all,
            },
        )
        write_json(
            out_dir / "cv_metrics.json",
            {
                "overall": overall_cv,
                "weighted_overall": {
                    "wrmse_imputed_only": weighted_cv.wrmse_imputed_only,
                    "wrmse_all": weighted_cv.wrmse_all,
                },
                "folds": fold_metrics,
            },
        )
        save_oof_npz(out_dir / "oof_predictions_train.npz", ids=Xtr["ID"].to_numpy(), target_cols=target_cols, preds=oof)

    pred_test = _predict_seedbag_prob(
        Xtr,
        ytr,
        Xte,
        target_cols,
        sensory_cols,
        SEEDS,
        log_prefix=" (test) ",
    )
    pred_test = _apply_copy_through(pred_test, Xte, target_cols)
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
            "seeds": SEEDS,
            "cv_overall": overall_cv,
            "wrmse_imputed_only": weighted_cv.wrmse_imputed_only if weighted_cv else None,
            "wrmse_all": weighted_cv.wrmse_all if weighted_cv else None,
            "artifacts": {
                "submission_csv": "predictions_test.csv",
                "cv_metrics_json": "cv_metrics.json" if do_cv else None,
                "oof_npz": "oof_predictions_train.npz" if do_cv else None,
                "weighted_oof_json": "weighted_oof.json" if do_cv else None,
            },
        },
    )
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do-cv", type=int, default=1)
    p.add_argument("--n-splits", type=int, default=CV_SPLITS)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--limit-targets", type=int, default=None)
    p.add_argument("--limit-test-rows", type=int, default=None)
    args = p.parse_args()

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        seed=args.seed,
        do_cv=bool(args.do_cv),
        n_splits=args.n_splits,
        limit_rows=args.limit_rows,
        limit_targets=args.limit_targets,
        limit_test_rows=args.limit_test_rows,
    )
    print(f"[asia2026 t2 discrete seedbag5 proba] done -> {out_dir}")


if __name__ == "__main__":
    main()
