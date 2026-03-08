#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

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


def _write_submission(sample_sub: pd.DataFrame, ids: pd.Series, pred: np.ndarray, target_cols: List[str], out_csv: Path) -> None:
    sub = sample_sub.copy()
    sub["ID"] = ids.values
    for j, c in enumerate(target_cols):
        sub[c] = pred[:, j]
    ensure_dir(out_csv.parent)
    sub.to_csv(out_csv, index=False)


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in items:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _prepare_covariates(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    cat_cols = [c for c in Xtr.columns if Xtr[c].dtype == object]
    num_cols = [c for c in Xtr.columns if c not in cat_cols]

    Xtr_num = Xtr[num_cols].copy()
    Xte_num = Xte[num_cols].copy()
    med = Xtr_num.median(numeric_only=True)
    Xtr_num = Xtr_num.fillna(med)
    Xte_num = Xte_num.fillna(med)

    if cat_cols:
        Xtr_cat = Xtr[cat_cols].copy().fillna("MISSING")
        Xte_cat = Xte[cat_cols].copy().fillna("MISSING")
        all_cat = pd.concat([Xtr_cat, Xte_cat], axis=0, ignore_index=True)
        all_dum = pd.get_dummies(all_cat, columns=cat_cols, dummy_na=False)
        Xtr_dum = all_dum.iloc[: len(Xtr_cat)].reset_index(drop=True)
        Xte_dum = all_dum.iloc[len(Xtr_cat) :].reset_index(drop=True)
        Xtr_proc = pd.concat([Xtr_num.reset_index(drop=True), Xtr_dum], axis=1)
        Xte_proc = pd.concat([Xte_num.reset_index(drop=True), Xte_dum], axis=1)
    else:
        Xtr_proc = Xtr_num.reset_index(drop=True)
        Xte_proc = Xte_num.reset_index(drop=True)

    return Xtr_proc.to_numpy(dtype=np.float32), Xte_proc.to_numpy(dtype=np.float32)


def _impute_chunks(imputer, X: np.ndarray, row_chunk: int) -> np.ndarray:
    out = np.empty_like(X)
    with torch.no_grad():
        for start in range(0, len(X), row_chunk):
            end = min(start + row_chunk, len(X))
            block = X[start:end].copy()
            out[start:end] = imputer.impute(block)
    return out


def _impute_with_retry(imputer, X: np.ndarray, row_chunk: int) -> np.ndarray:
    try:
        return _impute_chunks(imputer, X, row_chunk)
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "out of memory" in msg:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if row_chunk > 64:
                print("[tabimpute] CUDA OOM, retrying with row_chunk=64")
                return _impute_chunks(imputer, X, 64)
            if torch.cuda.is_available():
                print("[tabimpute] CUDA OOM at row_chunk=64, retrying on CPU")
                imputer_cpu = imputer.__class__(device="cpu")
                return _impute_chunks(imputer_cpu, X, 64)
        raise
    except Exception as exc:
        if torch.cuda.is_available():
            print(f"[tabimpute] impute failed on CUDA ({exc}); retrying on CPU")
            imputer_cpu = imputer.__class__(device="cpu")
            return _impute_chunks(imputer_cpu, X, min(64, row_chunk))
        raise


def _build_matrix(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    covariate_cols: List[str],
    target_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    covariate_cols = [c for c in covariate_cols if c in Xtr.columns]
    covariate_cols = _unique_keep_order([c for c in covariate_cols if c not in target_cols])

    Xtr_cov = Xtr[covariate_cols].copy()
    Xte_cov = Xte[covariate_cols].copy()
    Xtr_cov_np, Xte_cov_np = _prepare_covariates(Xtr_cov, Xte_cov)

    Xtr_targets = Xtr.reindex(columns=target_cols).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    Xte_targets = Xte.reindex(columns=target_cols).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

    Xtr_mat = np.concatenate([Xtr_cov_np, Xtr_targets], axis=1)
    Xte_mat = np.concatenate([Xte_cov_np, Xte_targets], axis=1)
    return Xtr_mat, Xte_mat, Xtr_targets, Xte_targets


def run_one(
    data_root: str,
    run_root: str,
    seed: int,
    mode: str,
    row_chunk: int,
    limit_rows: Optional[int] = None,
    limit_targets: Optional[int] = None,
) -> Path:
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

    baseline_cols = [c for c in Xtr.columns if c.startswith("w1_")]
    followup_motor = [c for c in data.motor_cols if not c.startswith("w1_")]
    covariate_cols = _unique_keep_order(baseline_cols + followup_motor + ["time", "vaccd"] + data.meta_cols)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if mode == "tabimpute":
        from tabimpute.interface import ImputePFN

        imputer = ImputePFN(device="cuda" if torch.cuda.is_available() else "cpu")
        method = "tabimpute"
    elif mode == "tabimpute_plus":
        from tabimpute.interface import MCTabPFNEnsemble

        imputer = MCTabPFNEnsemble(device="cuda" if torch.cuda.is_available() else "cpu")
        method = "tabimpute_plus"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    run_id = make_run_id(prefix=f"t2_{method}")
    out_dir = ensure_dir(Path(run_root) / run_id)

    cfg = RunConfig(
        track=track,
        method=method,
        data_root=data_root,
        run_root=run_root,
        seed=seed,
        do_cv=False,
        n_splits=0,
        limit_rows=limit_rows,
        limit_targets=limit_targets,
        notes=f"tabimpute t2: mode={mode}, row_chunk={row_chunk}",
    )
    write_json(out_dir / "config.json", {"run_id": run_id, "created_utc": utc_now_iso(), **cfg.to_dict()})

    Xtr_mat, Xte_mat, Xtr_targets, Xte_targets = _build_matrix(Xtr, Xte, covariate_cols, target_cols)
    cov_dim = Xtr_mat.shape[1] - len(target_cols)

    imputed_tr = _impute_with_retry(imputer, Xtr_mat, row_chunk)
    pred_tr = imputed_tr[:, cov_dim:]
    obs_mask = ~np.isnan(Xtr_targets)
    pred_tr[obs_mask] = Xtr_targets[obs_mask]
    pred_tr = _clip_predictions(pred_tr, target_cols)

    train_metrics = compute_metrics(
        ytr[target_cols].to_numpy(dtype=np.float32),
        pred_tr,
        target_cols,
        sensory_cols,
        features=Xtr,
    )
    write_json(out_dir / "train_metrics.json", train_metrics)

    imputed_te = _impute_with_retry(imputer, Xte_mat, row_chunk)
    pred_te = imputed_te[:, cov_dim:]
    obs_mask_te = ~np.isnan(Xte_targets)
    pred_te[obs_mask_te] = Xte_targets[obs_mask_te]
    pred_te = _clip_predictions(pred_te, target_cols)

    _write_submission(
        data.sample_submission,
        ids=Xte["ID"],
        pred=pred_te,
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
            "train_metrics": train_metrics,
            "artifacts": {
                "submission_csv": "predictions_test.csv",
                "train_metrics_json": "train_metrics.json",
            },
        },
    )
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, choices=["tabimpute", "tabimpute_plus"], default="tabimpute")
    p.add_argument("--row-chunk", type=int, default=128)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--limit-targets", type=int, default=None)
    args = p.parse_args()

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        seed=args.seed,
        mode=args.mode,
        row_chunk=args.row_chunk,
        limit_rows=args.limit_rows,
        limit_targets=args.limit_targets,
    )
    print(f"[asia2026 t2 tabimpute] done -> {out_dir}")


if __name__ == "__main__":
    main()
