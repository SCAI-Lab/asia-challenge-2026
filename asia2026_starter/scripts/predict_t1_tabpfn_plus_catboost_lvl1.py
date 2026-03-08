#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from asia2026.data import load_track
from asia2026.utils import ensure_dir, make_run_id, read_json, utc_now_iso, write_json

LOGGER = logging.getLogger(__name__)


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


def _prepare_base_features(
    df: pd.DataFrame,
    base_cols: List[str],
    fill_values: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    X_base = df[base_cols].copy()
    cat_cols = [
        c
        for c in base_cols
        if pd.api.types.is_object_dtype(X_base[c]) or pd.api.types.is_categorical_dtype(X_base[c])
    ]
    for c in cat_cols:
        X_base[c] = X_base[c].fillna("MISSING").astype(str)
    num_cols = [c for c in base_cols if c not in cat_cols]
    if fill_values and num_cols:
        X_base[num_cols] = X_base[num_cols].fillna(value=fill_values)
    return X_base


def _align_by_id(df: pd.DataFrame, ids: np.ndarray, label: str) -> pd.DataFrame:
    if "ID" not in df.columns:
        raise ValueError(f"{label} missing ID column")
    df_ids = df["ID"].astype(str)
    target_ids = pd.Series(ids.astype(str))
    if df_ids.equals(target_ids):
        return df
    aligned = df.set_index("ID").reindex(target_ids)
    if aligned.isna().any().any():
        raise ValueError(f"{label} does not contain all required IDs")
    return aligned.reset_index()


def _parse_alphas(alpha_str: str) -> List[float]:
    alphas = []
    for part in alpha_str.split(","):
        part = part.strip()
        if part:
            alphas.append(float(part))
    if not alphas:
        raise ValueError("No alphas provided")
    return alphas


def _build_block_features(
    X_base: pd.DataFrame,
    X_features: pd.DataFrame,
    baseline_block: np.ndarray,
    block_cols: List[str],
    add_missing: bool,
    feature_cols: List[str],
) -> pd.DataFrame:
    X = X_base.copy()
    for j, col in enumerate(block_cols):
        X[f"oof__{col}"] = baseline_block[:, j]
    if add_missing:
        for col in block_cols:
            if col in X_features.columns:
                X[f"miss__{col}"] = X_features[col].isna().astype(np.int8)
    X = X.reindex(columns=feature_cols)
    return X


def run_one(
    data_root: str,
    run_root: str,
    catboost_run: Path,
    tabpfn_submission: Path,
    alphas: List[float],
    limit_test_rows: Optional[int],
) -> Path:
    track = 1
    data = load_track(track, data_root)

    cfg = read_json(catboost_run / "residual_config.json")

    target_cols = cfg["target_cols"]
    base_cols = cfg["base_cols"]
    blocks = cfg["blocks"]
    feature_columns = cfg["feature_columns"]
    add_missing = bool(cfg.get("add_missing_indicators", True))
    num_fill = cfg.get("num_fill_values", {})

    Xte = data.X_test.copy()
    if limit_test_rows is not None:
        Xte = Xte.iloc[:limit_test_rows].reset_index(drop=True)

    base_test = _prepare_base_features(Xte, base_cols, num_fill)

    tabpfn_df = pd.read_csv(tabpfn_submission)
    tabpfn_df = _align_by_id(tabpfn_df, Xte["ID"].to_numpy(), "tabpfn submission")

    missing_targets = [c for c in target_cols if c not in tabpfn_df.columns]
    if missing_targets:
        raise ValueError(f"TabPFN submission missing target columns: {missing_targets}")

    tabpfn_vals = tabpfn_df[target_cols].to_numpy(dtype=np.float32)

    residual = np.zeros_like(tabpfn_vals, dtype=np.float32)

    for name, cols in blocks.items():
        model_path = catboost_run / "models" / f"{name}.cbm"
        if not model_path.exists():
            raise ValueError(f"Missing model file: {model_path}")

        block_idx = [target_cols.index(c) for c in cols]
        baseline_block = tabpfn_vals[:, block_idx]
        X_block = _build_block_features(
            base_test,
            Xte,
            baseline_block,
            cols,
            add_missing,
            feature_columns[name],
        )

        model = CatBoostRegressor()
        model.load_model(model_path)
        block_pred = model.predict(X_block)
        if block_pred.ndim == 1:
            block_pred = block_pred.reshape(-1, 1)

        for j, col in enumerate(cols):
            out_idx = target_cols.index(col)
            miss_mask = Xte[col].isna().to_numpy()
            residual[miss_mask, out_idx] = block_pred[miss_mask, j]

    run_id = make_run_id(prefix="t1_tabpfn_plus_catboost")
    out_dir = ensure_dir(Path(run_root) / run_id)

    outputs: Dict[str, str] = {}
    for alpha in alphas:
        blended = tabpfn_vals.copy()
        for name, cols in blocks.items():
            block_idx = [target_cols.index(c) for c in cols]
            imp = Xte[cols].isna().to_numpy()
            block_base = blended[:, block_idx]
            block_resid = residual[:, block_idx]
            block_base[imp] = block_base[imp] + (alpha * block_resid[imp])
            blended[:, block_idx] = block_base
        blended = np.nan_to_num(blended, nan=0.0)
        blended = _clip_predictions(blended, target_cols)
        blended = _apply_copy_through(blended, Xte, target_cols)

        out_csv = out_dir / f"predictions_test_lambda{alpha:.2f}.csv"
        sub = data.sample_submission.copy()
        sub = sub.iloc[: len(Xte)].copy()
        sub["ID"] = Xte["ID"].values
        for j, col in enumerate(target_cols):
            sub[col] = blended[:, j]
        sub.to_csv(out_csv, index=False)
        outputs[f"lambda_{alpha:.2f}"] = out_csv.name
        LOGGER.info("Wrote %s", out_csv)

    write_json(
        out_dir / "run_summary.json",
        {
            "run_id": run_id,
            "finished_utc": utc_now_iso(),
            "track": track,
            "method": "t1_tabpfn_plus_catboost",
            "tabpfn_submission": str(tabpfn_submission),
            "catboost_run": str(catboost_run),
            "alphas": alphas,
            "limit_test_rows": limit_test_rows,
            "artifacts": outputs,
        },
    )

    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--catboost-run", required=True)
    p.add_argument("--tabpfn-submission", required=True)
    p.add_argument("--alphas", default="0.1,0.2,0.3")
    p.add_argument("--limit-test-rows", type=int, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        catboost_run=Path(args.catboost_run),
        tabpfn_submission=Path(args.tabpfn_submission),
        alphas=_parse_alphas(args.alphas),
        limit_test_rows=args.limit_test_rows,
    )
    print(f"[asia2026 t1 tabpfn+catboost] done -> {out_dir}")


if __name__ == "__main__":
    main()
