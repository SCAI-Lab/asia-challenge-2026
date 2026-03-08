#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from asia2026.data import load_track
from asia2026.eval import save_oof_npz
from asia2026.metrics import compute_metrics
from asia2026.tabpfn_model_t1_discrete import tabpfn_predict_multioutput_t1_discrete
from asia2026.utils import RunConfig, ensure_dir, make_run_id, utc_now_iso, write_json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

LOG = logging.getLogger("t1_safe_discrete_oof")

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


def run_one(
    data_root: str,
    run_root: str,
    seed: int,
    do_cv: bool,
    n_splits: int,
    limit_rows: Optional[int] = None,
    limit_targets: Optional[int] = None,
) -> Path:
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

    method = "tabpfn_25_t1safe_discrete_oof"
    run_id = make_run_id(prefix=f"t1_{method}")
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
        notes="t1-safe tabpfn discrete: classifier-as-regressor for sensory targets (OOF saved)",
    )
    write_json(out_dir / "config.json", {"run_id": run_id, "created_utc": utc_now_iso(), **cfg.to_dict()})
    LOG.info("Run dir: %s", out_dir)
    LOG.info("Config: seed=%s do_cv=%s n_splits=%s", seed, bool(do_cv), n_splits)

    def predict(_Xtr: pd.DataFrame, _ytr: pd.DataFrame, _X: pd.DataFrame) -> np.ndarray:
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
        for j, c in enumerate(target_cols):
            if c in _X.columns:
                obs = _X[c].to_numpy()
                m = ~pd.isna(obs)
                pred[m, j] = obs[m]
        return pred

    overall_cv = None
    if do_cv:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros((len(Xtr), len(target_cols)), dtype=np.float32)
        fold_metrics = []
        fold_iter = kf.split(Xtr)
        if tqdm is not None:
            fold_iter = tqdm(fold_iter, total=n_splits, desc="CV folds", unit="fold")
        for fold, (tr_idx, va_idx) in enumerate(fold_iter):
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
        save_oof_npz(
            out_dir / "oof_predictions_train.npz",
            ids=Xtr["ID"].to_numpy(),
            target_cols=target_cols,
            preds=oof,
        )
        write_json(out_dir / "cv_metrics.json", {"overall": overall_cv, "folds": fold_metrics})
        LOG.info("CV done; OOF saved.")

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
                "oof_npz": "oof_predictions_train.npz" if do_cv else None,
                "submission_csv": "predictions_test.csv",
                "cv_metrics_json": "cv_metrics.json" if do_cv else None,
            },
        },
    )
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--do-cv", type=int, default=1)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--limit-targets", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if tqdm is None:
        LOG.warning("tqdm not installed; progress bars disabled.")

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        seed=args.seed,
        do_cv=bool(args.do_cv),
        n_splits=args.n_splits,
        limit_rows=args.limit_rows,
        limit_targets=args.limit_targets,
    )
    print(f"[asia2026 t1 safe discrete oof] done -> {out_dir}")


if __name__ == "__main__":
    main()
