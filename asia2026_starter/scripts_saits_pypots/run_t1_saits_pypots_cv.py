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
from asia2026.eval import (
    compute_imputed_only_breakdown,
    compute_imputed_only_metrics,
    make_time_stratified_folds,
    save_oof_npz,
)
from asia2026.saits_pypots_t1 import (
    apply_copy_through,
    build_sens_grid,
    build_target_mapping,
    clip_predictions,
    grid_to_targets,
)
from asia2026.tabpfn_model_t1_discrete import tabpfn_predict_multioutput_t1_discrete
from asia2026.utils import ensure_dir, make_run_id, utc_now_iso, write_json

LOGGER = logging.getLogger("t1_saits_pypots_cv")


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


def _resolve_imputed_array(imputed: Any) -> np.ndarray:
    if isinstance(imputed, dict):
        for key in ("imputation", "imputed_data", "X_imputed"):
            if key in imputed:
                return np.asarray(imputed[key])
    return np.asarray(imputed)


def _init_saits(args: argparse.Namespace, n_steps: int, n_features: int) -> SAITS:
    return SAITS(
        n_steps=n_steps,
        n_features=n_features,
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
    parser.add_argument("--n-splits", type=int, default=5)
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
        args.n_splits = min(args.n_splits, 2)
        args.val_frac = max(args.val_frac, 0.2)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Starting SAITS PyPOTS CV run (smoke=%s)", args.smoke)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    torch.set_num_threads(1)

    _set_seed(args.seed)

    data = load_track(1, args.data_root)
    target_cols = data.target_cols
    mapping = build_target_mapping(target_cols)

    X_train = _maybe_limit_rows(data.X_train, args.max_train_rows)
    y_train = data.y_train.iloc[: len(X_train)].reset_index(drop=True)
    ids = data.X_train["ID"].iloc[: len(X_train)].to_numpy()

    X_test = _maybe_limit_rows(data.X_test, args.max_test_rows)
    sub_df = _maybe_limit_rows(data.sample_submission, args.max_test_rows)

    folds = make_time_stratified_folds(
        X_train["time"].to_numpy() if "time" in X_train.columns else np.arange(len(X_train)),
        n_splits=args.n_splits,
        seed=args.seed,
    )

    oof = np.full((len(X_train), len(target_cols)), np.nan, dtype=np.float32)

    for fold_idx, (tr_idx, val_idx) in enumerate(folds, start=1):
        if len(val_idx) == 0:
            continue
        LOGGER.info("Fold %s/%s (train=%s, val=%s)", fold_idx, len(folds), len(tr_idx), len(val_idx))
        _set_seed(args.seed + fold_idx)

        X_tr = X_train.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_train.iloc[tr_idx].reset_index(drop=True)
        X_val = X_train.iloc[val_idx].reset_index(drop=True)
        y_val = y_train.iloc[val_idx].reset_index(drop=True)

        X_tr_sens = build_sens_grid(X_tr)
        X_val_sens = build_sens_grid(X_val)
        X_tr_ori = build_sens_grid(y_tr)
        X_val_ori = build_sens_grid(y_val)
        tr_missing = int(np.isnan(X_tr_sens).sum())
        val_missing = int(np.isnan(X_val_sens).sum())
        tr_label_missing = int(np.isnan(X_tr_ori).sum())
        val_label_missing = int(np.isnan(X_val_ori).sum())
        LOGGER.info(
            "Fold %s missing counts: train=%s val=%s train_labels=%s val_labels=%s",
            fold_idx,
            tr_missing,
            val_missing,
            tr_label_missing,
            val_label_missing,
        )

        X_tr_saits = X_tr_sens
        X_val_saits = X_val_sens

        train_set = {"X": X_tr_saits, "X_ori": X_tr_ori}
        val_set = {"X": X_val_saits, "X_ori": X_val_ori}

        saits = _init_saits(args, X_tr_saits.shape[1], X_tr_saits.shape[2])
        saits.fit(train_set, val_set)

        imputed_val = _resolve_imputed_array(saits.impute({"X": X_val_saits}))
        if imputed_val.shape != X_val_saits.shape:
            raise RuntimeError(f"Unexpected imputed shape {imputed_val.shape} vs {X_val_saits.shape}")

        anyana_pred = tabpfn_predict_multioutput_t1_discrete(
            X_tr,
            y_tr,
            X_val,
            target_cols=["anyana"],
            sensory_target_cols=[],
            seed=args.seed + fold_idx,
        ).reshape(-1)

        preds_val = grid_to_targets(imputed_val[:, :, :4], mapping, anyana_pred)
        preds_val = clip_predictions(preds_val, target_cols)
        preds_val = apply_copy_through(preds_val, X_val, target_cols)
        imp_mask = X_val[target_cols].isna().to_numpy()
        if imp_mask.shape[1] != len(target_cols):
            raise RuntimeError("Imputed mask shape mismatch vs target cols")
        grid_mask = np.zeros_like(imp_mask, dtype=bool)
        for j, m in enumerate(mapping):
            if m is None:
                continue
            d_idx, s_idx = m
            grid_mask[:, j] = np.isnan(X_val_sens[:, d_idx, s_idx])
        sens_idx = [i for i, c in enumerate(target_cols) if c != "anyana"]
        mismatch = int(np.sum(grid_mask[:, sens_idx] != imp_mask[:, sens_idx]))
        if mismatch:
            LOGGER.warning("Fold %s missing-mask mismatch count: %s", fold_idx, mismatch)
        vals = preds_val[imp_mask]
        if vals.size:
            LOGGER.info(
                "Fold %s imputed-only pred stats: mean=%.4f std=%.4f min=%.4f max=%.4f",
                fold_idx,
                float(vals.mean()),
                float(vals.std()),
                float(vals.min()),
                float(vals.max()),
            )

        oof[val_idx] = preds_val.astype(np.float32)

    if np.isnan(oof).any():
        raise RuntimeError("OOF contains NaNs; check fold coverage.")

    oof_metrics = compute_imputed_only_metrics(
        y_train,
        oof,
        X_train,
        target_cols,
        data.sensory_target_cols,
    )
    oof_breakdown = compute_imputed_only_breakdown(
        y_train,
        oof,
        X_train,
        target_cols,
        data.sensory_target_cols,
    )

    LOGGER.info("Training full SAITS for test prediction")
    _set_seed(args.seed)
    X_tr_sens = build_sens_grid(X_train)
    X_tr_ori = build_sens_grid(y_train)
    X_te_sens = build_sens_grid(X_test)
    train_missing = int(np.isnan(X_tr_sens).sum())
    train_label_missing = int(np.isnan(X_tr_ori).sum())
    test_missing = int(np.isnan(X_te_sens).sum())
    LOGGER.info(
        "Full-train missing counts: train=%s train_labels=%s test=%s",
        train_missing,
        train_label_missing,
        test_missing,
    )

    X_tr_saits = X_tr_sens
    X_te_saits = X_te_sens
    X_tr_ori_full = X_tr_ori

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(X_tr_saits))
    tr_idx, val_idx = _split_train_val(idx, args.val_frac)
    if len(tr_idx) == 0:
        tr_idx = idx[:1]
        val_idx = idx[1:2]

    train_set = {"X": X_tr_saits[tr_idx], "X_ori": X_tr_ori_full[tr_idx]}
    val_set = {"X": X_tr_saits[val_idx], "X_ori": X_tr_ori_full[val_idx]} if len(val_idx) else None

    saits = _init_saits(args, X_tr_saits.shape[1], X_tr_saits.shape[2])
    saits.fit(train_set, val_set)

    imputed_test = _resolve_imputed_array(saits.impute({"X": X_te_saits}))
    if imputed_test.shape != X_te_saits.shape:
        raise RuntimeError(f"Unexpected imputed shape {imputed_test.shape} vs {X_te_saits.shape}")

    anyana_test = tabpfn_predict_multioutput_t1_discrete(
        X_train,
        y_train,
        X_test,
        target_cols=["anyana"],
        sensory_target_cols=[],
        seed=args.seed,
    ).reshape(-1)

    preds_test = grid_to_targets(imputed_test[:, :, :4], mapping, anyana_test)
    preds_test = clip_predictions(preds_test, target_cols)
    preds_test = apply_copy_through(preds_test, X_test, target_cols)

    run_id = make_run_id("t1_saits_pypots_cv_smoke" if args.smoke else "t1_saits_pypots_cv")
    out_dir = ensure_dir(Path(args.run_root) / run_id)

    save_oof_npz(out_dir / "oof_predictions_train.npz", ids, target_cols, oof)

    pred_df = sub_df.copy()
    pred_df[target_cols] = preds_test
    pred_df.to_csv(out_dir / "predictions_test.csv", index=False)

    summary = {
        "method": "t1_saits_pypots_cv",
        "run_id": run_id,
        "timestamp": utc_now_iso(),
        "data_root": args.data_root,
        "run_root": args.run_root,
        "seed": args.seed,
        "device": args.device,
        "smoke": args.smoke,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "n_splits": args.n_splits,
        "val_frac": args.val_frac,
        "saits_input": "sensory_only",
        "oof_metrics": oof_metrics.to_dict(),
        "oof_breakdown": oof_breakdown,
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
