#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import os
import pandas as pd
from tqdm import tqdm

from asia2026.data import load_track
from asia2026.eval import compute_imputed_only_breakdown, compute_imputed_only_metrics, load_oof_npz
from asia2026.utils import ensure_dir, utc_now_iso, write_json

ID_COL = "ID"


def _parse_alphas(raw: str) -> List[float]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise ValueError("No alphas parsed")
    return out


def _load_pred(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ID_COL not in df.columns:
        raise ValueError(f"Missing {ID_COL} in {path}")
    return df


def _align_by_id(df: pd.DataFrame, ids: np.ndarray, name: str) -> pd.DataFrame:
    df_idx = df.set_index(ID_COL)
    missing = set(ids) - set(df_idx.index)
    if missing:
        raise ValueError(f"{name} missing {len(missing)} IDs (first 5: {list(missing)[:5]})")
    return df_idx.loc[ids].reset_index()


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


def _write_blend(out_dir: Path, name: str, ids: np.ndarray, target_cols: List[str], pred: np.ndarray) -> None:
    df = pd.DataFrame({ID_COL: ids})
    for j, c in enumerate(target_cols):
        df[c] = pred[:, j]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / name, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tabpfn-run", required=True, help="Run dir name under runs/ for TabPFN seedbag")
    p.add_argument("--catboost-run", required=True, help="Run dir name under runs/ for CatBoost blocks")
    p.add_argument("--saits-run", default=None, help="Optional run dir name under runs/ for SAITS")
    p.add_argument("--data-root", required=True, help="Path to staged data root")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--alphas", default="0.60,0.70,0.80,0.90")
    p.add_argument("--saits-alphas", default="0.70")
    args = p.parse_args()

    runs_root = Path(os.environ.get("ASIA2026_RUNS_DIR", "runs"))
    tabpfn_dir = runs_root / args.tabpfn_run
    catboost_dir = runs_root / args.catboost_run

    tabpfn_oof_path = tabpfn_dir / "oof_predictions_train.npz"
    catboost_oof_path = catboost_dir / "oof_predictions_train.npz"

    if not tabpfn_oof_path.exists():
        raise SystemExit(f"Missing OOF file: {tabpfn_oof_path}")
    if not catboost_oof_path.exists():
        raise SystemExit(f"Missing OOF file: {catboost_oof_path}")

    tab_ids, tab_cols, tab_oof = load_oof_npz(tabpfn_oof_path)
    cat_ids, cat_cols, cat_oof = load_oof_npz(catboost_oof_path)

    if not np.array_equal(tab_ids, cat_ids):
        raise SystemExit("OOF ID mismatch between TabPFN and CatBoost")
    if tab_cols != cat_cols:
        raise SystemExit("OOF target columns mismatch between TabPFN and CatBoost")

    data = load_track(1, args.data_root)
    target_cols = data.target_cols

    if target_cols != tab_cols:
        raise SystemExit("Target columns differ from OOF file columns")

    alphas = _parse_alphas(args.alphas)
    out_dir = Path(args.out_dir)

    metrics_rows = []
    for alpha in tqdm(alphas, desc="Alphas"):
        blend_oof = alpha * tab_oof + (1.0 - alpha) * cat_oof
        blend_oof = _clip_predictions(blend_oof, target_cols)
        blend_oof = _apply_copy_through(blend_oof, data.X_train, target_cols)

        metrics = compute_imputed_only_metrics(data.y_train, blend_oof, data.X_train, target_cols, data.sensory_target_cols)
        metrics_rows.append({"alpha": alpha, **metrics.to_dict()})

    metrics_df = pd.DataFrame(metrics_rows)
    ensure_dir(out_dir)
    metrics_path = out_dir / "blend_oof_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    best_row = metrics_df.sort_values("rmse_all_imputed_only", ascending=True).iloc[0]
    best_alpha = float(best_row["alpha"])
    best_blend = best_alpha * tab_oof + (1.0 - best_alpha) * cat_oof
    best_blend = _clip_predictions(best_blend, target_cols)
    best_blend = _apply_copy_through(best_blend, data.X_train, target_cols)
    breakdown = compute_imputed_only_breakdown(data.y_train, best_blend, data.X_train, target_cols, data.sensory_target_cols)

    tabpfn_pred = _load_pred(tabpfn_dir / "predictions_test.csv")
    catboost_pred = _load_pred(catboost_dir / "predictions_test.csv")

    ids = tabpfn_pred[ID_COL].to_numpy()
    catboost_pred = _align_by_id(catboost_pred, ids, "catboost")

    tabpfn_vals = tabpfn_pred[target_cols].to_numpy(dtype=np.float32)
    catboost_vals = catboost_pred[target_cols].to_numpy(dtype=np.float32)

    for alpha in alphas:
        blend = alpha * tabpfn_vals + (1.0 - alpha) * catboost_vals
        blend = _clip_predictions(blend, target_cols)
        blend = _apply_copy_through(blend, data.X_test, target_cols)
        name = f"blend_a{int(round(alpha * 100)):03d}.csv"
        _write_blend(out_dir, name, ids, target_cols, blend)

    summary = {
        "finished_utc": utc_now_iso(),
        "method": "t1_blend_stage3",
        "tabpfn_run": args.tabpfn_run,
        "catboost_run": args.catboost_run,
        "alphas": alphas,
        "best_alpha": best_alpha,
        "best_rmse_all_imputed_only": float(best_row["rmse_all_imputed_only"]),
        "best_rmse_sensory_imputed_only": float(best_row["rmse_sensory_imputed_only"]),
        "metrics_breakdown": breakdown,
        "metrics_csv": str(metrics_path),
    }

    if args.saits_run:
        saits_dir = runs_root / args.saits_run
        saits_oof_path = saits_dir / "oof_predictions_train.npz"
        if not saits_oof_path.exists():
            raise SystemExit(f"Missing SAITS OOF file: {saits_oof_path}")
        saits_ids, saits_cols, saits_oof = load_oof_npz(saits_oof_path)
        if not np.array_equal(tab_ids, saits_ids):
            raise SystemExit("OOF ID mismatch between TabPFN and SAITS")
        if saits_cols != tab_cols:
            raise SystemExit("OOF target columns mismatch between TabPFN and SAITS")

        saits_alphas = _parse_alphas(args.saits_alphas)
        saits_metrics = []
        for alpha in tqdm(saits_alphas, desc="SAITS Alphas"):
            blend_oof = alpha * tab_oof + (1.0 - alpha) * saits_oof
            blend_oof = _clip_predictions(blend_oof, target_cols)
            blend_oof = _apply_copy_through(blend_oof, data.X_train, target_cols)
            metrics = compute_imputed_only_metrics(
                data.y_train, blend_oof, data.X_train, target_cols, data.sensory_target_cols
            )
            saits_metrics.append({"alpha": alpha, **metrics.to_dict()})

        saits_metrics_df = pd.DataFrame(saits_metrics)
        saits_metrics_path = out_dir / "blend_saits_oof_metrics.csv"
        saits_metrics_df.to_csv(saits_metrics_path, index=False)

        saits_best = saits_metrics_df.sort_values("rmse_all_imputed_only", ascending=True).iloc[0]
        saits_best_alpha = float(saits_best["alpha"])
        saits_best_blend = saits_best_alpha * tab_oof + (1.0 - saits_best_alpha) * saits_oof
        saits_best_blend = _clip_predictions(saits_best_blend, target_cols)
        saits_best_blend = _apply_copy_through(saits_best_blend, data.X_train, target_cols)
        saits_breakdown = compute_imputed_only_breakdown(
            data.y_train, saits_best_blend, data.X_train, target_cols, data.sensory_target_cols
        )

        saits_pred = _load_pred(saits_dir / "predictions_test.csv")
        saits_pred = _align_by_id(saits_pred, ids, "saits")
        saits_vals = saits_pred[target_cols].to_numpy(dtype=np.float32)

        for alpha in saits_alphas:
            blend = alpha * tabpfn_vals + (1.0 - alpha) * saits_vals
            blend = _clip_predictions(blend, target_cols)
            blend = _apply_copy_through(blend, data.X_test, target_cols)
            name = f"blend_saits_a{int(round(alpha * 100)):03d}.csv"
            _write_blend(out_dir, name, ids, target_cols, blend)

        summary["saits_blend"] = {
            "saits_run": args.saits_run,
            "saits_alphas": saits_alphas,
            "best_alpha": saits_best_alpha,
            "best_rmse_all_imputed_only": float(saits_best["rmse_all_imputed_only"]),
            "best_rmse_sensory_imputed_only": float(saits_best["rmse_sensory_imputed_only"]),
            "metrics_breakdown": saits_breakdown,
            "metrics_csv": str(saits_metrics_path),
        }

    write_json(
        out_dir / "run_summary.json",
        summary,
    )

    print(f"Wrote metrics: {metrics_path}")
    print(f"Best alpha: {best_alpha}")


if __name__ == "__main__":
    main()
