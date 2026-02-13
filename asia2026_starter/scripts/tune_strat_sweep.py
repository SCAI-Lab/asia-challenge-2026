#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from asia2026.baselines import baseline_strat_time_mean, baseline_time_mean
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


def run_one(
    track: int,
    data_root: str,
    run_root: str,
    seed: int,
    do_cv: bool,
    n_splits: int,
    strat_cols: List[str],
    min_group: int,
    limit_rows: Optional[int] = None,
    limit_targets: Optional[int] = None,
    label: str = "",
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

    # Ensure bins exist if requested.
    if "age_bin" in strat_cols and "age_bin" not in Xtr.columns and "age" in Xtr.columns:
        bins = [0, 25, 35, 45, 55, 65, 120]
        Xtr["age_bin"] = pd.cut(Xtr["age"], bins=bins, right=False, include_lowest=True)
        Xte["age_bin"] = pd.cut(Xte["age"], bins=bins, right=False, include_lowest=True)
    if "bmi_bin" in strat_cols and "bmi_bin" not in Xtr.columns and "bmi" in Xtr.columns:
        bins = [0, 18.5, 25, 30, 35, 60]
        Xtr["bmi_bin"] = pd.cut(Xtr["bmi"], bins=bins, right=False, include_lowest=True)
        Xte["bmi_bin"] = pd.cut(Xte["bmi"], bins=bins, right=False, include_lowest=True)

    # baseline_strat_time_mean already adds time internally; drop it if present.
    strat_cols = [c for c in strat_cols if c != "time" and c in Xtr.columns]
    method = f"strat_sweep_{label}_m{min_group}" if label else f"strat_sweep_m{min_group}"
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
        notes=f"strat_cols={strat_cols}, min_group={min_group}",
    )
    write_json(out_dir / "config.json", {"run_id": run_id, "created_utc": utc_now_iso(), **cfg.to_dict()})

    def predict(_Xtr: pd.DataFrame, _ytr: pd.DataFrame, _X: pd.DataFrame) -> np.ndarray:
        if not strat_cols:
            return baseline_time_mean(_Xtr, _ytr, _X, target_cols)
        return baseline_strat_time_mean(
            _Xtr,
            _ytr,
            _X,
            target_cols,
            strat_cols=strat_cols,
            min_group=min_group,
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
    p.add_argument("--min-groups", type=str, default="")
    args = p.parse_args()

    if args.min_groups.strip():
        min_groups = [int(x.strip()) for x in args.min_groups.split(",") if x.strip()]
    else:
        min_groups = [50, 100] if args.track == 1 else [30, 60]

    # Strata candidates: sex + age bins + BMI bins; time is handled separately.
    strata_sets = [
        ("time_sex", ["time", "sexcd"]),
        ("time_sex_agebin", ["time", "sexcd", "age_bin"]),
        ("time_sex_bmibin", ["time", "sexcd", "bmi_bin"]),
        ("time_sex_agebin_bmibin", ["time", "sexcd", "age_bin", "bmi_bin"]),
    ]

    for label, cols in strata_sets:
        for mg in min_groups:
            out_dir = run_one(
                track=args.track,
                data_root=args.data_root,
                run_root=args.run_root,
                seed=args.seed,
                do_cv=bool(args.do_cv),
                n_splits=args.n_splits,
                strat_cols=cols,
                min_group=mg,
                limit_rows=args.limit_rows,
                limit_targets=args.limit_targets,
                label=label,
            )
            print(f"[tune_strat_sweep] done -> {out_dir}")


if __name__ == "__main__":
    main()
