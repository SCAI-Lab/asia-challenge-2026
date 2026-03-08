#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import logging
import os
from pathlib import Path
from typing import List
import time

import numpy as np
import pandas as pd
import torch

from asia2026.data import load_track
from asia2026.utils import RunConfig, ensure_dir, make_run_id, utc_now_iso, write_json

from tqdm import tqdm

LOG = logging.getLogger("t1_autotabpfn_overlay_T24")


def _import_autotabpfn_classifier():
    try:
        from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier

        return AutoTabPFNClassifier
    except ImportError as exc:
        raise ImportError(
            "AutoTabPFNClassifier not found. Install via: pip install \"tabpfn-extensions[post_hoc_ensembles]\""
        ) from exc


def _select_overlay_targets(X_ref: pd.DataFrame, target_cols: List[str], topk: int) -> List[str]:
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


def _select_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _init_autotabpfn(
    max_time: int,
    presets: str,
    num_cpus: int,
    ignore_pretraining_limits: bool,
    n_ensemble_models: int,
    fast_debug: bool,
    device: str,
    phe_init_args: dict | None,
    dynamic_stacking: bool,
):
    AutoTabPFNClassifier = _import_autotabpfn_classifier()
    phe_fit_args = {
        "num_cpus": num_cpus,
        "dynamic_stacking": bool(dynamic_stacking),
    }
    if fast_debug:
        phe_fit_args.update(
            {
                "num_bag_folds": 0,
                "fit_weighted_ensemble": False,
            }
        )
    kwargs = {
        "device": device,
        "n_estimators": 8,
        "n_ensemble_models": int(n_ensemble_models),
        "ignore_pretraining_limits": bool(ignore_pretraining_limits),
        "phe_fit_args": phe_fit_args,
        "phe_init_args": phe_init_args,
    }
    sig = inspect.signature(AutoTabPFNClassifier)
    if "max_time" in sig.parameters:
        kwargs["max_time"] = max_time
    if "presets" in sig.parameters:
        kwargs["presets"] = presets
    return AutoTabPFNClassifier(**kwargs)


def _fit_autotabpfn(
    model,
    X: np.ndarray,
    y: np.ndarray,
    max_time: int,
    presets: str,
    n_jobs: int,
    raise_on_no_models_fitted: bool,
) -> None:
    kwargs = {}
    sig = inspect.signature(model.fit)
    if "max_time" in sig.parameters:
        kwargs["max_time"] = max_time
    if "presets" in sig.parameters:
        kwargs["presets"] = presets
    if "n_jobs" in sig.parameters:
        kwargs["n_jobs"] = n_jobs
    if "raise_on_no_models_fitted" in sig.parameters:
        kwargs["raise_on_no_models_fitted"] = raise_on_no_models_fitted
    model.fit(X, y, **kwargs)


def _clip_predictions(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in target_cols:
        if c == "anyana":
            out[c] = out[c].clip(0.0, 1.0)
        else:
            out[c] = out[c].clip(0.0, 2.0)
    return out


def _coerce_categoricals(Xtr: pd.DataFrame, Xte: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    Xtr = Xtr.copy()
    Xte = Xte.copy()
    cat_cols = [
        c
        for c in Xtr.columns
        if pd.api.types.is_object_dtype(Xtr[c])
        or pd.api.types.is_string_dtype(Xtr[c])
        or isinstance(Xtr[c].dtype, pd.CategoricalDtype)
    ]
    for c in cat_cols:
        tr = Xtr[c].astype("category")
        te = Xte[c].astype("category")
        cats = pd.Index(tr.cat.categories).union(te.cat.categories)
        Xtr[c] = tr.cat.set_categories(cats)
        Xte[c] = te.cat.set_categories(cats)
    return Xtr, Xte


def run_one(
    data_root: str,
    run_root: str,
    baseline_csv: str,
    target_topk: int,
    limit_overlay_targets: int | None,
    limit_train_rows: int | None,
    max_time: int,
    presets: str,
    n_jobs: int,
    ignore_pretraining_limits: bool,
    n_ensemble_models: int,
    fast_debug: bool,
    autogluon_verbosity: int,
    raise_on_no_models_fitted: bool,
    dynamic_stacking: bool,
) -> Path:
    track = 1
    data = load_track(track, data_root)

    Xtr = data.X_train.copy()
    ytr = data.y_train.copy()
    Xte = data.X_test.copy()

    if limit_train_rows is not None:
        Xtr = Xtr.iloc[: int(limit_train_rows)].reset_index(drop=True)
        ytr = ytr.iloc[: int(limit_train_rows)].reset_index(drop=True)
    target_cols = data.target_cols
    sensory_cols = set(data.sensory_target_cols)
    device = _select_device()
    if device == "cpu":
        # Allow CPU fallback for large datasets.
        if not os.environ.get("TABPFN_ALLOW_CPU_LARGE_DATASET"):
            os.environ["TABPFN_ALLOW_CPU_LARGE_DATASET"] = "1"

    method = "t1_autotabpfn_overlay_T24"
    run_id = make_run_id(prefix=method)
    out_dir = ensure_dir(Path(run_root) / run_id)

    cfg = RunConfig(
        track=track,
        method=method,
        data_root=data_root,
        run_root=run_root,
        seed=0,
        do_cv=False,
        n_splits=0,
        notes="AutoTabPFN overlay on top-missing targets (T24) using baseline submission",
    )
    write_json(
        out_dir / "config.json",
        {
            "run_id": run_id,
            "created_utc": utc_now_iso(),
            **cfg.to_dict(),
            "baseline_csv": baseline_csv,
            "target_topk": target_topk,
            "limit_overlay_targets": limit_overlay_targets,
            "limit_train_rows": limit_train_rows,
            "max_time": max_time,
            "presets": presets,
            "n_jobs": n_jobs,
            "ignore_pretraining_limits": bool(ignore_pretraining_limits),
            "n_ensemble_models": int(n_ensemble_models),
            "fast_debug": bool(fast_debug),
            "device": device,
            "autogluon_verbosity": int(autogluon_verbosity),
            "raise_on_no_models_fitted": bool(raise_on_no_models_fitted),
            "dynamic_stacking": bool(dynamic_stacking),
        },
    )

    baseline_df = pd.read_csv(baseline_csv)
    if "ID" not in baseline_df.columns:
        raise ValueError(f"baseline file missing ID column: {baseline_csv}")
    missing_cols = [c for c in target_cols if c not in baseline_df.columns]
    if missing_cols:
        raise ValueError(f"baseline file missing target columns (first 10): {missing_cols[:10]}")
    baseline_df = baseline_df.set_index("ID").loc[Xte["ID"]].copy()

    Xref = Xte if all(c in Xte.columns for c in target_cols) else Xtr
    overlay_targets = _select_overlay_targets(Xref, target_cols, target_topk)
    if limit_overlay_targets is not None:
        overlay_targets = overlay_targets[: int(limit_overlay_targets)]
    LOG.info("Overlay targets: %s (count=%s)", overlay_targets, len(overlay_targets))

    Xtr_base, Xte_base = _coerce_categoricals(Xtr, Xte)

    sensory_classes = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    anyana_classes = np.array([0.0, 1.0], dtype=np.float32)

    target_iter = tqdm(overlay_targets, desc="AutoTabPFN targets", unit="target")

    for idx, c in enumerate(target_iter, start=1):
        t0 = time.time()
        LOG.info("AutoTabPFN target %s/%s: %s", idx, len(overlay_targets), c)
        Xtr_c = Xtr_base.copy()
        Xte_c = Xte_base.copy()
        if c in Xtr_c.columns:
            Xtr_c[c] = np.nan
        if c in Xte_c.columns:
            Xte_c[c] = np.nan

        y = ytr[c].to_numpy()
        y_int = np.rint(y).astype(np.int64)

        phe_init_args = {
            "verbosity": int(autogluon_verbosity),
            "path": str(out_dir / f"autogluon_{c}"),
        }
        model = _init_autotabpfn(
            max_time=max_time,
            presets=presets,
            num_cpus=n_jobs,
            ignore_pretraining_limits=ignore_pretraining_limits,
            n_ensemble_models=n_ensemble_models,
            fast_debug=fast_debug,
            device=device,
            phe_init_args=phe_init_args,
            dynamic_stacking=dynamic_stacking,
        )
        _fit_autotabpfn(
            model,
            Xtr_c,
            y_int,
            max_time=max_time,
            presets=presets,
            n_jobs=n_jobs,
            raise_on_no_models_fitted=raise_on_no_models_fitted,
        )
        fit_s = time.time() - t0
        proba = np.asarray(model.predict_proba(Xte_c), dtype=np.float32)
        total_s = time.time() - t0
        LOG.info("AutoTabPFN target done: %s (fit=%.1fs total=%.1fs)", c, fit_s, total_s)
        classes = np.asarray(model.classes_)

        if c == "anyana":
            pred = _expected_from_proba_fixed(proba, classes, anyana_classes)
        elif c in sensory_cols:
            pred = _expected_from_proba_fixed(proba, classes, sensory_classes)
        else:
            pred = _expected_from_proba_fixed(proba, classes, sensory_classes)

        if c in Xte.columns:
            missing_mask = Xte[c].isna().to_numpy()
        else:
            missing_mask = np.ones(len(Xte), dtype=bool)
        if missing_mask.any():
            baseline_df.loc[missing_mask, c] = pred[missing_mask]

    for c in target_cols:
        if c in Xte.columns:
            obs = Xte[c].to_numpy()
            m = ~pd.isna(obs)
            baseline_df.loc[m, c] = obs[m]

    baseline_df = _clip_predictions(baseline_df, target_cols)

    out_csv = out_dir / "predictions_test_autotabpfn_overlay_T24.csv"
    out_df = baseline_df.reset_index()
    out_df.to_csv(out_csv, index=False)

    write_json(
        out_dir / "run_summary.json",
        {
            "run_id": run_id,
            "finished_utc": utc_now_iso(),
            "track": track,
            "method": method,
            "baseline_csv": baseline_csv,
            "target_topk": target_topk,
            "overlay_targets": overlay_targets,
            "max_time": max_time,
            "presets": presets,
            "n_jobs": n_jobs,
            "raise_on_no_models_fitted": bool(raise_on_no_models_fitted),
            "dynamic_stacking": bool(dynamic_stacking),
            "artifacts": {
                "submission_csv": out_csv.name,
            },
        },
    )
    LOG.info("Run summary written: %s", out_dir / "run_summary.json")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--run-root", type=str, required=True)
    p.add_argument(
        "--baseline-csv",
        type=str,
        default=(
            "/cluster/scratch/ppurkayastha/Acads/SCAI/asia2026_starter/runs/"
            "t1_discrete_seedbag5_proba__20260222_120028__job58116257__zkw773g3/"
            "predictions_test.csv"
        ),
    )
    p.add_argument("--target-topk", type=int, default=24)
    p.add_argument("--limit-overlay-targets", type=int, default=None)
    p.add_argument("--limit-train-rows", type=int, default=None)
    p.add_argument("--max-time", type=int, default=180)
    p.add_argument("--presets", type=str, default="medium_quality")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--ignore-pretraining-limits", type=int, default=0)
    p.add_argument("--n-ensemble-models", type=int, default=20)
    p.add_argument("--fast-debug", type=int, default=0)
    p.add_argument("--autogluon-verbosity", type=int, default=2)
    p.add_argument("--raise-on-no-models-fitted", type=int, default=0)
    p.add_argument("--dynamic-stacking", type=int, default=1)
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # tqdm is required for live progress/ETA.

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        baseline_csv=args.baseline_csv,
        target_topk=int(args.target_topk),
        limit_overlay_targets=args.limit_overlay_targets,
        limit_train_rows=args.limit_train_rows,
        max_time=int(args.max_time),
        presets=args.presets,
        n_jobs=int(args.n_jobs),
        ignore_pretraining_limits=bool(args.ignore_pretraining_limits),
        n_ensemble_models=int(args.n_ensemble_models),
        fast_debug=bool(args.fast_debug),
        autogluon_verbosity=int(args.autogluon_verbosity),
        raise_on_no_models_fitted=bool(args.raise_on_no_models_fitted),
        dynamic_stacking=bool(args.dynamic_stacking),
    )
    print(f"[t1_autotabpfn_overlay_T24] done -> {out_dir}")


if __name__ == "__main__":
    main()
