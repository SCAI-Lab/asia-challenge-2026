#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from asia2026.data import load_track
from asia2026.eval import (
    compute_imputed_only_breakdown,
    compute_imputed_only_metrics,
    make_time_stratified_folds,
    save_oof_npz,
)
from asia2026.features_isncsci_t1 import add_isncsci_features_t1
from asia2026.metrics_weighted import compute_wrmse_imputed_only
from asia2026.post_constraints_t1 import apply_constraints_t1
from asia2026.tabpfn_model_t1_discrete import (
    _init_tabpfn_classifier,
    _safe_reduce,
    build_preprocessor,
)
from asia2026.utils import ensure_dir, make_run_id, utc_now_iso, write_json

SEEDS = [11, 22, 33, 44, 55]
LOGGER = logging.getLogger(__name__)


def _set_single_thread_env() -> None:
    os.environ.setdefault("TABPFN_NUM_WORKERS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


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
    Xtr_base: pd.DataFrame,
    ytr: pd.DataFrame,
    Xte_base: pd.DataFrame,
    target_cols: List[str],
    sensory_cols: List[str],
    seeds: List[int],
    motor_cols: List[str],
    meta_cols: List[str],
    log_prefix: str,
) -> np.ndarray:
    sensory_targets = set(sensory_cols)
    sensory_classes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    anyana_classes = np.array([0.0, 1.0], dtype=np.float32)

    copy_cols = [c for c in target_cols if c in Xte_base.columns and not Xte_base[c].isna().any()]
    LOGGER.info("%sCopy-through targets: %s", log_prefix, len(copy_cols))

    p_sum: Dict[str, np.ndarray] = {}
    for col in target_cols:
        n_classes = 2 if col == "anyana" else 3
        p_sum[col] = np.zeros((len(Xte_base), n_classes), dtype=np.float32)

    for col in tqdm(target_cols, desc=f"Targets{log_prefix}"):
        if col in copy_cols:
            continue

        is_sensory = col in sensory_targets
        is_anyana = col == "anyana"
        if not is_sensory and not is_anyana:
            raise ValueError(f"Unexpected non-sensory target in discrete seedbag: {col}")

        Xtr_masked = Xtr_base.copy()
        Xte_masked = Xte_base.copy()
        if col in Xtr_masked.columns:
            Xtr_masked[col] = np.nan
        if col in Xte_masked.columns:
            Xte_masked[col] = np.nan

        Xtr_aug = add_isncsci_features_t1(Xtr_masked, motor_cols=motor_cols, meta_cols=meta_cols)
        Xte_aug = add_isncsci_features_t1(Xte_masked, motor_cols=motor_cols, meta_cols=meta_cols)
        Xtr_all, Xte_all, name_to_idx = _preprocess(Xtr_aug, Xte_aug)

        leak_idx = name_to_idx.get(f"num__{col}")

        for seed in tqdm(seeds, desc=f"Seeds target={col}{log_prefix}", leave=False):
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

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

            try:
                Xtr_safe, Xte_safe = _safe_reduce(Xtr_use, Xte_use, seed=seed, do_pca=False)
                model.fit(Xtr_safe, y_int)
                proba = model.predict_proba(Xte_safe)
            except Exception:
                Xtr_safe, Xte_safe = _safe_reduce(Xtr_use, Xte_use, seed=seed, do_pca=True)
                model.fit(Xtr_safe, y_int)
                proba = model.predict_proba(Xte_safe)

            target_classes = anyana_classes if is_anyana else sensory_classes
            p_sum[col] += _proba_to_fixed(proba, model.classes_, target_classes)

    out = np.zeros((len(Xte_base), len(target_cols)), dtype=np.float32)
    for j, col in enumerate(target_cols):
        if col in copy_cols:
            continue
        target_classes = anyana_classes if col == "anyana" else sensory_classes
        p_avg = p_sum[col] / float(len(seeds))
        out[:, j] = p_avg @ target_classes

    return out


def _mask_reconstruct(
    Xtr_base: pd.DataFrame,
    ytr: pd.DataFrame,
    target_cols: List[str],
    sensory_cols: List[str],
    mask_frac: float,
    seed: int,
    motor_cols: List[str],
    meta_cols: List[str],
) -> dict:
    rng = np.random.default_rng(seed)
    X_masked = Xtr_base.copy()
    mask_artificial = np.zeros((len(Xtr_base), len(target_cols)), dtype=bool)

    for j, col in enumerate(target_cols):
        if col not in sensory_cols:
            continue
        if col not in X_masked.columns:
            continue
        obs = ~X_masked[col].isna().to_numpy()
        rand = rng.random(len(X_masked))
        m = (rand < mask_frac) & obs
        if m.any():
            X_masked.loc[m, col] = np.nan
            mask_artificial[:, j] = m

    pred = _predict_seedbag_prob(
        Xtr_base,
        ytr,
        X_masked,
        target_cols,
        sensory_cols,
        SEEDS,
        motor_cols,
        meta_cols,
        log_prefix=" (mask-recon) ",
    )
    X_masked_feat = add_isncsci_features_t1(X_masked, motor_cols=motor_cols, meta_cols=meta_cols)
    pred = apply_constraints_t1(pred, X_masked_feat, target_cols)
    pred = _clip_predictions(pred, target_cols)
    pred = _apply_copy_through(pred, X_masked_feat, target_cols)

    true_vals = ytr[target_cols].to_numpy(dtype=np.float32)
    if mask_artificial.sum() == 0:
        rmse = 0.0
        mae = 0.0
    else:
        rmse = float(np.sqrt(np.mean((true_vals[mask_artificial] - pred[mask_artificial]) ** 2)))
        mae = float(np.mean(np.abs(true_vals[mask_artificial] - pred[mask_artificial])))

    return {
        "mask_frac": mask_frac,
        "rmse_masked_only": rmse,
        "mae_masked_only": mae,
        "n_masked": int(mask_artificial.sum()),
    }


def run_one(
    data_root: str,
    run_root: str,
    seed: int,
    n_splits: int,
    limit_rows: Optional[int] = None,
    limit_targets: Optional[int] = None,
    limit_test_rows: Optional[int] = None,
    mask_frac: float = 0.1,
) -> Path:
    _set_single_thread_env()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 1 seedbag5 proba. No GPU detected.")

    track = 1
    data = load_track(track, data_root)

    Xtr_base = data.X_train.copy()
    ytr = data.y_train.copy()
    Xte_base = data.X_test.copy()

    if limit_rows is not None:
        Xtr_base = Xtr_base.iloc[:limit_rows].reset_index(drop=True)
        ytr = ytr.iloc[:limit_rows].reset_index(drop=True)
    if limit_test_rows is not None:
        Xte_base = Xte_base.iloc[:limit_test_rows].reset_index(drop=True)

    Xtr_feat = add_isncsci_features_t1(Xtr_base, motor_cols=data.motor_cols, meta_cols=data.meta_cols)
    Xte_feat = add_isncsci_features_t1(Xte_base, motor_cols=data.motor_cols, meta_cols=data.meta_cols)

    target_cols = data.target_cols
    sensory_cols = data.sensory_target_cols
    if limit_targets is not None:
        target_cols = target_cols[:limit_targets]
        sensory_cols = [c for c in sensory_cols if c in target_cols]

    method = "t1_discrete_seedbag5_proba_isncsci"
    run_id = make_run_id(prefix=method)
    out_dir = ensure_dir(Path(run_root) / run_id)

    LOGGER.info("Targets: %d (sensory=%d).", len(target_cols), len(sensory_cols))
    LOGGER.info("Run output: %s", out_dir)

    folds = make_time_stratified_folds(Xtr_feat["time"].to_numpy(), n_splits=n_splits, seed=seed)
    oof = np.zeros((len(Xtr_feat), len(target_cols)), dtype=np.float32)
    fold_metrics = []

    for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc="Folds")):
        Xtr_f_base = Xtr_base.iloc[tr_idx].reset_index(drop=True)
        ytr_f = ytr.iloc[tr_idx].reset_index(drop=True)
        Xva_f_base = Xtr_base.iloc[va_idx].reset_index(drop=True)
        yva_f = ytr.iloc[va_idx].reset_index(drop=True)
        Xva_f_feat = Xtr_feat.iloc[va_idx].reset_index(drop=True)

        LOGGER.info("Fold %d: train=%d val=%d", fold, len(Xtr_f_base), len(Xva_f_base))
        pred_va = _predict_seedbag_prob(
            Xtr_f_base,
            ytr_f,
            Xva_f_base,
            target_cols,
            sensory_cols,
            SEEDS,
            data.motor_cols,
            data.meta_cols,
            log_prefix=f" (fold={fold}) ",
        )
        pred_va = apply_constraints_t1(pred_va, Xva_f_feat, target_cols)
        pred_va = _clip_predictions(pred_va, target_cols)
        pred_va = _apply_copy_through(pred_va, Xva_f_feat, target_cols)
        oof[va_idx] = pred_va

        fold_imputed = compute_imputed_only_metrics(yva_f, pred_va, Xva_f_feat, target_cols, sensory_cols)
        fold_wrmse = compute_wrmse_imputed_only(
            y_true=yva_f,
            oof_pred=pred_va,
            X_train_features=Xva_f_feat,
            X_test_features=Xte_feat,
            target_cols=target_cols,
        )
        fold_metrics.append(
            {
                "fold": fold,
                "n_train": int(len(Xtr_f_base)),
                "n_val": int(len(Xva_f_base)),
                "imputed_only": fold_imputed.to_dict(),
                "wrmse_imputed_only": fold_wrmse.wrmse_imputed_only,
                "wrmse_all": fold_wrmse.wrmse_all,
            }
        )

    oof_path = out_dir / "oof_predictions_train.npz"
    save_oof_npz(oof_path, ids=Xtr_base["ID"].to_numpy(), target_cols=target_cols, preds=oof)

    metrics = compute_imputed_only_metrics(ytr, oof, Xtr_feat, target_cols, sensory_cols)
    breakdown = compute_imputed_only_breakdown(ytr, oof, Xtr_feat, target_cols, sensory_cols)
    wrep = compute_wrmse_imputed_only(
        y_true=ytr,
        oof_pred=oof,
        X_train_features=Xtr_feat,
        X_test_features=Xte_feat,
        target_cols=target_cols,
    )
    LOGGER.info("OOF imputed-only RMSE (sensory): %.6f", metrics.rmse_sensory_imputed_only)
    LOGGER.info("OOF imputed-only RMSE (all): %.6f", metrics.rmse_all_imputed_only)
    LOGGER.info("OOF WRMSE (imputed-only): %.6f", wrep.wrmse_imputed_only)
    LOGGER.info("OOF WRMSE (all): %.6f", wrep.wrmse_all)

    overall_cv = metrics.to_dict()
    overall_cv["wrmse_imputed_only"] = wrep.wrmse_imputed_only
    overall_cv["wrmse_all"] = wrep.wrmse_all

    write_json(
        out_dir / "weighted_oof.json",
        {
            "wrmse_imputed_only": wrep.wrmse_imputed_only,
            "wrmse_all": wrep.wrmse_all,
            "per_target_contrib": wrep.per_target_contrib,
            "weights": wrep.weights,
            "mse_per_target": wrep.mse_per_target,
            "mse_per_target_all": wrep.mse_per_target_all,
        },
    )
    write_json(
        out_dir / "cv_metrics.json",
        {
            "folds": fold_metrics,
            "overall": overall_cv,
        },
    )

    pred_test = _predict_seedbag_prob(
        Xtr_base,
        ytr,
        Xte_base,
        target_cols,
        sensory_cols,
        SEEDS,
        data.motor_cols,
        data.meta_cols,
        log_prefix=" (test) ",
    )
    pred_test = apply_constraints_t1(pred_test, Xte_feat, target_cols)
    pred_test = _clip_predictions(pred_test, target_cols)
    pred_test = _apply_copy_through(pred_test, Xte_feat, target_cols)

    _write_submission(
        data.sample_submission,
        ids=Xte_feat["ID"],
        pred=pred_test,
        target_cols=target_cols,
        out_csv=out_dir / "predictions_test.csv",
    )

    mask_metrics = _mask_reconstruct(
        Xtr_base,
        ytr,
        target_cols,
        sensory_cols,
        mask_frac=mask_frac,
        seed=seed,
        motor_cols=data.motor_cols,
        meta_cols=data.meta_cols,
    )

    write_json(
        out_dir / "run_summary.json",
        {
            "run_id": run_id,
            "finished_utc": utc_now_iso(),
            "track": track,
            "method": method,
            "seeds": SEEDS,
            "n_splits": n_splits,
            "mask_frac": mask_frac,
            "feature_augmented": True,
            "constraints_applied": True,
            "metrics": metrics.to_dict(),
            "wrmse_imputed_only": wrep.wrmse_imputed_only,
            "wrmse_all": wrep.wrmse_all,
            "metrics_breakdown": breakdown,
            "mask_reconstruct_metrics": mask_metrics,
            "cv_folds": fold_metrics,
            "artifacts": {
                "submission_csv": "predictions_test.csv",
                "oof_npz": "oof_predictions_train.npz",
                "weighted_oof_json": "weighted_oof.json",
                "cv_metrics_json": "cv_metrics.json",
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
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--limit-test-rows", type=int, default=None)
    p.add_argument("--limit-targets", type=int, default=None)
    p.add_argument("--mask-frac", type=float, default=0.1)
    args = p.parse_args()

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        seed=args.seed,
        n_splits=args.n_splits,
        limit_rows=args.limit_rows,
        limit_targets=args.limit_targets,
        limit_test_rows=args.limit_test_rows,
        mask_frac=args.mask_frac,
    )
    print(f"[asia2026 t1 seedbag5 proba + isncsci] done -> {out_dir}")


if __name__ == "__main__":
    main()
