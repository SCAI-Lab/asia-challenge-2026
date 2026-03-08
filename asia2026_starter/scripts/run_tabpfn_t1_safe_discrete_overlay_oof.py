#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional, Sequence

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
except ImportError:  # pragma: no cover - fallback if tqdm is missing
    tqdm = None

LOG = logging.getLogger("t1_safe_discrete_overlay")


def _configure_hf_cache() -> None:
    asia_root = os.environ.get("ASIA2026_ROOT")
    if asia_root:
        hf_home = os.environ.setdefault("HF_HOME", f"{asia_root}/hf_home")
        os.environ.setdefault("HF_HUB_CACHE", f"{hf_home}/hub")
        os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HF_HUB_CACHE"])
        xdg_cache = os.environ.setdefault("XDG_CACHE_HOME", f"{asia_root}/xdg_cache")
        os.environ.setdefault("TABPFN_CACHE_DIR", f"{xdg_cache}/tabpfn")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")


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


def _parse_seeds(seeds: Optional[Sequence[int]]) -> List[int]:
    if not seeds:
        return [11, 22]
    return list(seeds)


def _select_overlay_targets(
    X_ref: pd.DataFrame,
    target_cols: List[str],
    topk: int,
) -> List[str]:
    missing_counts = X_ref[target_cols].isna().sum().sort_values(ascending=False)
    always_missing = missing_counts[missing_counts == len(X_ref)].index.tolist()
    top_missing = missing_counts.head(topk).index.tolist()
    seen = set()
    ordered = []
    for c in always_missing + top_missing:
        if c not in seen:
            ordered.append(c)
            seen.add(c)
    return ordered


def run_one(
    data_root: str,
    run_root: str,
    baseline_csv: str,
    seeds: Sequence[int],
    target_topk: int,
    do_cv: bool,
    n_splits: int,
    limit_rows: Optional[int] = None,
    limit_targets: Optional[int] = None,
) -> Path:
    _configure_hf_cache()
    track = 1
    LOG.info("Loading data for track=%s from %s", track, data_root)
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

    method = "tabpfn_t1safe_discrete_overlay_oof"
    run_id = make_run_id(prefix=f"t1_{method}")
    out_dir = ensure_dir(Path(run_root) / run_id)
    LOG.info("Run dir: %s", out_dir)

    cfg = RunConfig(
        track=track,
        method=method,
        data_root=data_root,
        run_root=run_root,
        seed=int(seeds[0]) if seeds else 11,
        do_cv=bool(do_cv),
        n_splits=n_splits if do_cv else 0,
        limit_rows=limit_rows,
        limit_targets=limit_targets,
        notes="t1-safe tabpfn discrete overlay: partial seed ensemble on top-missing targets (OOF saved)",
    )
    write_json(
        out_dir / "config.json",
        {
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            **cfg.to_dict(),
            "baseline_csv": baseline_csv,
            "seeds": list(seeds),
            "target_topk": target_topk,
        },
    )
    LOG.info("Config: seeds=%s target_topk=%s baseline_csv=%s", list(seeds), target_topk, baseline_csv)

    def predict_full(_Xtr: pd.DataFrame, _ytr: pd.DataFrame, _X: pd.DataFrame, seed: int) -> np.ndarray:
        LOG.info("Predict full: n_train=%s n_pred=%s seed=%s", len(_Xtr), len(_X), seed)
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

    def _avg_metrics(metrics_list: List[dict]) -> dict:
        if not metrics_list:
            return {}
        sums = {}
        counts = {}
        for m in metrics_list:
            for k, v in m.items():
                if isinstance(v, (int, float, np.floating)):
                    sums[k] = sums.get(k, 0.0) + float(v)
                    counts[k] = counts.get(k, 0) + 1
        return {k: sums[k] / counts[k] for k in sums}

    overall_cv = None
    oof_avg = None
    seedwise_cv = None
    if do_cv:
        LOG.info("Starting CV: n_splits=%s seeds=%s", n_splits, list(seeds))
        seedwise_cv = []
        oof_seed_preds = []
        seed_iter = seeds
        if tqdm is not None:
            seed_iter = tqdm(seeds, desc="CV seeds", unit="seed")
        for cv_seed in seed_iter:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=int(cv_seed))
            oof = np.zeros((len(Xtr), len(target_cols)), dtype=np.float32)
            fold_metrics = []
            fold_iter = kf.split(Xtr)
            if tqdm is not None:
                fold_iter = tqdm(fold_iter, total=n_splits, desc=f"CV folds (seed={cv_seed})", unit="fold")
            for fold, (tr_idx, va_idx) in enumerate(fold_iter):
                Xtr_f = Xtr.iloc[tr_idx].reset_index(drop=True)
                ytr_f = ytr.iloc[tr_idx].reset_index(drop=True)
                Xva_f = Xtr.iloc[va_idx].reset_index(drop=True)
                yva_f = ytr.iloc[va_idx].reset_index(drop=True)

                pred_va = predict_full(Xtr_f, ytr_f, Xva_f, seed=int(cv_seed))
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

            overall_seed = compute_metrics(
                ytr[target_cols].to_numpy(),
                oof,
                target_cols,
                sensory_cols,
                features=Xtr,
            )
            save_oof_npz(
                out_dir / f"oof_predictions_train_seed{int(cv_seed)}.npz",
                ids=Xtr["ID"].to_numpy(),
                target_cols=target_cols,
                preds=oof,
            )
            oof_seed_preds.append(oof)
            seedwise_cv.append(
                {
                    "seed": int(cv_seed),
                    "overall": overall_seed,
                    "folds": fold_metrics,
                }
            )

        overall_cv = _avg_metrics([s["overall"] for s in seedwise_cv])
        if oof_seed_preds:
            oof_avg = np.mean(np.stack(oof_seed_preds, axis=0), axis=0)
            save_oof_npz(
                out_dir / "oof_predictions_train_avg.npz",
                ids=Xtr["ID"].to_numpy(),
                target_cols=target_cols,
                preds=oof_avg,
            )
            overall_from_avg = compute_metrics(
                ytr[target_cols].to_numpy(),
                oof_avg,
                target_cols,
                sensory_cols,
                features=Xtr,
            )
        else:
            overall_from_avg = None
        write_json(
            out_dir / "cv_metrics.json",
            {
                "overall_avg": overall_cv,
                "overall_from_oof_avg": overall_from_avg,
                "seedwise": seedwise_cv,
            },
        )
        LOG.info("CV done; metrics written.")

    Xref = Xte if all(c in Xte.columns for c in target_cols) else Xtr
    overlay_targets = _select_overlay_targets(Xref, target_cols, target_topk)
    overlay_sensory = [c for c in sensory_cols if c in overlay_targets]
    LOG.info("Overlay targets: %s (count=%s)", overlay_targets, len(overlay_targets))
    LOG.info("Overlay targets joined: %s", ",".join(overlay_targets))

    baseline_df = pd.read_csv(baseline_csv)
    if "ID" not in baseline_df.columns:
        raise ValueError(f"baseline file missing ID column: {baseline_csv}")
    baseline_df = baseline_df.set_index("ID").loc[Xte["ID"]].copy()
    LOG.info("Baseline loaded from %s", baseline_csv)

    n_test = len(Xte)
    p_sum = np.zeros((n_test, len(overlay_targets)), dtype=np.float32)

    def predict_overlay(seed: int) -> np.ndarray:
        LOG.info("Predict overlay: seed=%s", seed)
        copy_cols = [c for c in overlay_targets if c in Xte.columns and not Xte[c].isna().any()]
        pred = tabpfn_predict_multioutput_t1_discrete(
            Xtr,
            ytr[overlay_targets],
            Xte,
            target_cols=overlay_targets,
            sensory_target_cols=overlay_sensory,
            copy_through_cols=copy_cols,
            max_train_samples=None,
            seed=seed,
        )
        pred = np.nan_to_num(pred, nan=0.0)
        for j, c in enumerate(overlay_targets):
            if c in Xte.columns:
                obs = Xte[c].to_numpy()
                m = ~pd.isna(obs)
                pred[m, j] = obs[m]
        return pred

    seed_iter = seeds
    if tqdm is not None:
        seed_iter = tqdm(seeds, desc="Overlay seeds", unit="seed")
    for seed in seed_iter:
        p_sum += predict_overlay(seed)

    p_avg = p_sum / max(1, len(seeds))
    p_avg = _clip_predictions(p_avg, overlay_targets)

    for j, c in enumerate(overlay_targets):
        baseline_df[c] = p_avg[:, j]

    Xte_idx = Xte.set_index("ID").loc[baseline_df.index]
    for c in target_cols:
        if c in Xte_idx.columns:
            obs = Xte_idx[c]
            m = ~obs.isna()
            baseline_df.loc[m, c] = obs[m]

    out_csv = out_dir / "predictions_test_overlay.csv"
    LOG.info("Writing overlay submission: %s", out_csv)
    _write_submission(
        data.sample_submission,
        ids=pd.Series(baseline_df.index, name="ID"),
        pred=baseline_df[target_cols].to_numpy(),
        target_cols=target_cols,
        out_csv=out_csv,
    )

    write_json(
        out_dir / "run_summary.json",
        {
            "run_id": run_id,
            "finished_utc": utc_now_iso(),
            "track": track,
            "method": method,
            "cv_overall": overall_cv,
            "cv_seedwise_avg": overall_cv,
            "seeds": list(seeds),
            "target_topk": target_topk,
            "overlay_targets": overlay_targets,
            "artifacts": {
                "oof_avg_npz": "oof_predictions_train_avg.npz" if do_cv else None,
                "submission_csv": "predictions_test_overlay.csv",
                "baseline_csv": baseline_csv,
                "cv_metrics_json": "cv_metrics.json" if do_cv else None,
            },
        },
    )
    LOG.info("Run summary written.")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument(
        "--baseline-csv",
        type=str,
        default="/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter/Task1_Discrete.csv",
    )
    p.add_argument("--target-topk", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    p.add_argument("--do-cv", type=int, default=0)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--limit-rows", type=int, default=None)
    p.add_argument("--limit-targets", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if tqdm is None:
        LOG.warning("tqdm not installed; progress bars disabled.")

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        baseline_csv=args.baseline_csv,
        seeds=_parse_seeds(args.seeds),
        target_topk=args.target_topk,
        do_cv=bool(args.do_cv),
        n_splits=args.n_splits,
        limit_rows=args.limit_rows,
        limit_targets=args.limit_targets,
    )
    print(f"[asia2026 t1 safe discrete overlay oof] done -> {out_dir}")


if __name__ == "__main__":
    main()
