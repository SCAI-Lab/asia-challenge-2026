#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pypots.imputation import SAITS

from asia2026.data import load_track
from asia2026.saits_pypots_t1 import (
    apply_copy_through,
    build_sens_grid,
    build_target_mapping,
    clip_predictions,
    grid_to_targets,
)
from asia2026.tabpfn_model_t1_discrete import tabpfn_predict_multioutput_t1_discrete
from asia2026.utils import ensure_dir, make_run_id, utc_now_iso, write_json

LOGGER = logging.getLogger("t1_saits_pypots")


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_train_val(idx: np.ndarray, val_frac: float) -> tuple[np.ndarray, np.ndarray]:
    n = len(idx)
    if n <= 1:
        return idx, idx[:0]
    val_size = max(1, int(n * val_frac))
    if n - val_size < 1:
        val_size = n - 1
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]
    return tr_idx, val_idx


def _maybe_limit_rows(df: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.iloc[:max_rows].reset_index(drop=True)


def _rmse_imputed_only(pred: np.ndarray, truth: np.ndarray, obs_mask: np.ndarray) -> float | None:
    missing = ~obs_mask
    missing_count = int(missing.sum())
    if missing_count < 1:
        return None
    diff = pred[missing] - truth[missing]
    return float(np.sqrt(np.mean(diff**2)))


def _resolve_imputed_array(imputed: Any) -> np.ndarray:
    if isinstance(imputed, dict):
        for key in ("imputation", "imputed_data", "X_imputed"):
            if key in imputed:
                return np.asarray(imputed[key])
    return np.asarray(imputed)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=os.environ.get("ASIA2026_DATA_DIR", "data/staged"))
    parser.add_argument("--run-root", default=os.environ.get("ASIA2026_SAITS_RUNS_DIR", "runs_saits_pypots"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ffn", type=int, default=128)
    parser.add_argument("--d-k", type=int, default=32)
    parser.add_argument("--d-v", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attn-dropout", type=float, default=0.1)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-test-rows", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 1)
        args.patience = min(args.patience, 1)
        args.batch_size = min(args.batch_size, 16)
        args.max_train_rows = args.max_train_rows or 64
        args.max_test_rows = args.max_test_rows or 2
        args.val_frac = max(args.val_frac, 0.2)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Starting SAITS PyPOTS run (smoke=%s)", args.smoke)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)

    _set_seed(args.seed)

    data = load_track(1, args.data_root)
    target_cols = data.target_cols

    X_train = _maybe_limit_rows(data.X_train, args.max_train_rows)
    y_train = data.y_train.iloc[: len(X_train)].reset_index(drop=True)
    X_test = _maybe_limit_rows(data.X_test, args.max_test_rows)
    sub_df = _maybe_limit_rows(data.sample_submission, args.max_test_rows)

    Xtr_sens = build_sens_grid(X_train)
    Xtr_ori = build_sens_grid(y_train)
    Xte_sens = build_sens_grid(X_test)
    train_missing = int(np.isnan(Xtr_sens).sum())
    train_label_missing = int(np.isnan(Xtr_ori).sum())
    test_missing = int(np.isnan(Xte_sens).sum())
    LOGGER.info(
        "Missing counts: train=%s train_labels=%s test=%s",
        train_missing,
        train_label_missing,
        test_missing,
    )

    Xtr_saits = Xtr_sens
    Xte_saits = Xte_sens
    Xtr_ori_full = Xtr_ori

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(Xtr_saits))
    tr_idx, val_idx = _split_train_val(idx, args.val_frac)
    if len(tr_idx) == 0:
        tr_idx = idx[:1]
        val_idx = idx[1:2]

    train_set = {"X": Xtr_saits[tr_idx], "X_ori": Xtr_ori_full[tr_idx]}
    val_set = {"X": Xtr_saits[val_idx], "X_ori": Xtr_ori_full[val_idx]} if len(val_idx) else None
    test_set = {"X": Xte_saits}

    saits = SAITS(
        n_steps=Xtr_saits.shape[1],
        n_features=Xtr_saits.shape[2],
        n_layers=args.n_layers,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_k=args.d_k,
        d_v=args.d_v,
        d_ffn=args.d_ffn,
        dropout=args.dropout,
        attn_dropout=args.attn_dropout,
        diagonal_attention_mask=True,
        ORT_weight=0.0,
        MIT_weight=1.0,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        num_workers=0,
        device=args.device,
    )

    LOGGER.info("Training SAITS on %s rows (val=%s)", len(tr_idx), len(val_idx))
    saits.fit(train_set, val_set)

    LOGGER.info("Imputing test (%s rows)", len(Xte_saits))
    imputed_test = _resolve_imputed_array(saits.impute(test_set))
    if imputed_test.shape != Xte_saits.shape:
        raise RuntimeError(f"Unexpected imputed shape {imputed_test.shape} vs {Xte_saits.shape}")

    val_rmse = None
    if val_set is not None and len(val_idx):
        LOGGER.info("Imputing val for sanity metrics")
        imputed_val = _resolve_imputed_array(saits.impute({"X": Xtr_saits[val_idx]}))
        obs_mask = ~np.isnan(Xtr_saits[val_idx, :, :4])
        val_rmse = _rmse_imputed_only(imputed_val[:, :, :4], Xtr_ori[val_idx], obs_mask)

    anyana_pred = tabpfn_predict_multioutput_t1_discrete(
        X_train,
        y_train,
        X_test,
        target_cols=["anyana"],
        sensory_target_cols=[],
        seed=args.seed,
    ).reshape(-1)

    pred_grid = imputed_test[:, :, :4]
    mapping = build_target_mapping(target_cols)
    preds = grid_to_targets(pred_grid, mapping, anyana_pred)
    preds = clip_predictions(preds, target_cols)
    preds = apply_copy_through(preds, X_test, target_cols)
    imp_mask = X_test[target_cols].isna().to_numpy()
    vals = preds[imp_mask]
    if vals.size:
        LOGGER.info(
            "Test imputed-only pred stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
            float(vals.mean()),
            float(vals.std()),
            float(vals.min()),
            float(vals.max()),
        )

    mismatches = 0
    for j, c in enumerate(target_cols):
        if c in X_test.columns:
            obs = X_test[c].to_numpy()
            m = ~pd.isna(obs)
            if m.any():
                mismatches += int(np.sum(np.abs(preds[m, j] - obs[m]) > 1e-6))

    run_id = make_run_id("t1_saits_pypots_smoke" if args.smoke else "t1_saits_pypots")
    out_dir = ensure_dir(Path(args.run_root) / run_id)

    pred_df = sub_df.copy()
    pred_df[target_cols] = preds
    pred_path = out_dir / "predictions_test.csv"
    pred_df.to_csv(pred_path, index=False)

    summary = {
        "method": "t1_saits_pypots",
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "data_root": args.data_root,
        "run_root": args.run_root,
        "seed": args.seed,
        "device": args.device,
        "smoke": args.smoke,
        "train_rows": int(len(tr_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(X_test)),
        "saits_input": "sensory_only",
        "val_rmse_imputed_only": val_rmse,
        "copy_through_mismatches": mismatches,
        "missing_counts": {
            "train_missing": train_missing,
            "train_label_missing": train_label_missing,
            "test_missing": test_missing,
        },
        "saits_config": {
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "d_ffn": args.d_ffn,
            "d_k": args.d_k,
            "d_v": args.d_v,
            "dropout": args.dropout,
            "attn_dropout": args.attn_dropout,
        },
        "notes": args.notes,
    }
    write_json(out_dir / "run_summary.json", summary)
    LOGGER.info("Done. Output -> %s", out_dir)


if __name__ == "__main__":
    main()
