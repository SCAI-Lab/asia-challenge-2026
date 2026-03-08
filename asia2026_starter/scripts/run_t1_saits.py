#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from asia2026.data import load_track
from asia2026.eval import (
    compute_imputed_only_breakdown,
    compute_imputed_only_metrics,
    make_time_stratified_folds,
    save_oof_npz,
)
from asia2026.utils import ensure_dir, make_run_id, utc_now_iso, write_json

ID_COL = "ID"
SUFFIXES = ["ltl", "ltr", "ppl", "ppr"]
LOGGER = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _derive_dermatomes(target_cols: List[str]) -> List[str]:
    derms = [c[: -len("ltl")] for c in target_cols if c.endswith("ltl")]
    if len(derms) != 28:
        raise ValueError(f"Expected 28 dermatomes from ltl columns, got {len(derms)}")
    return derms


def _build_sensory_arrays(
    df: pd.DataFrame,
    derms: List[str],
    suffixes: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    seq = np.zeros((n, len(derms), len(suffixes)), dtype=np.float32)
    mask = np.zeros_like(seq, dtype=np.float32)
    for i, derm in enumerate(derms):
        for j, suf in enumerate(suffixes):
            col = f"{derm}{suf}"
            if col not in df.columns:
                # Some suffixes (e.g., ltr) may omit s45; treat as always-missing.
                continue
            vals = df[col].to_numpy(dtype=np.float32)
            m = ~pd.isna(vals)
            seq[:, i, j] = np.nan_to_num(vals, nan=0.0)
            mask[:, i, j] = m.astype(np.float32)
    return seq, mask


def _prepare_static_features(
    Xtr: pd.DataFrame,
    Xte: pd.DataFrame,
    cols: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    Xtr = Xtr[cols].copy()
    Xte = Xte[cols].copy()

    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(Xtr[c])]
    num_cols = [c for c in cols if c not in cat_cols]

    for c in num_cols:
        med = float(Xtr[c].median()) if not Xtr[c].isna().all() else 0.0
        Xtr[c] = Xtr[c].fillna(med)
        Xte[c] = Xte[c].fillna(med)
        mean = float(Xtr[c].mean())
        std = float(Xtr[c].std())
        if std == 0.0 or math.isnan(std):
            std = 1.0
        Xtr[c] = (Xtr[c] - mean) / std
        Xte[c] = (Xte[c] - mean) / std

    for c in cat_cols:
        Xtr[c] = Xtr[c].fillna("MISSING")
        Xte[c] = Xte[c].fillna("MISSING")

    combined = pd.concat([Xtr, Xte], axis=0, ignore_index=True)
    combined = pd.get_dummies(combined, columns=cat_cols, dtype=np.float32)

    n_tr = len(Xtr)
    Xtr_out = combined.iloc[:n_tr].to_numpy(dtype=np.float32)
    Xte_out = combined.iloc[n_tr:].to_numpy(dtype=np.float32)
    return Xtr_out, Xte_out


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


def _build_target_mapping(target_cols: List[str], derms: List[str]) -> List[Tuple[int, int] | None]:
    derm_idx = {d: i for i, d in enumerate(derms)}
    mapping: List[Tuple[int, int] | None] = []
    for col in target_cols:
        if col == "anyana":
            mapping.append(None)
            continue
        matched = False
        for s_idx, suf in enumerate(SUFFIXES):
            if col.endswith(suf):
                derm = col[: -len(suf)]
                if derm not in derm_idx:
                    raise ValueError(f"Unknown dermatome for column {col}")
                mapping.append((derm_idx[derm], s_idx))
                matched = True
                break
        if not matched:
            raise ValueError(f"Unsupported target column: {col}")
    return mapping


def _grid_to_targets(
    pred_grid: np.ndarray,
    mapping: List[Tuple[int, int] | None],
    anyana_value: float,
) -> np.ndarray:
    out = np.zeros((pred_grid.shape[0], len(mapping)), dtype=np.float32)
    for j, m in enumerate(mapping):
        if m is None:
            out[:, j] = anyana_value
        else:
            d_idx, s_idx = m
            out[:, j] = pred_grid[:, d_idx, s_idx]
    return out


class SaitsDataset(Dataset):
    def __init__(self, seq_obs: np.ndarray, seq_true: np.ndarray, mask: np.ndarray, static: np.ndarray):
        self.seq_obs = seq_obs
        self.seq_true = seq_true
        self.mask = mask
        self.static = static

    def __len__(self) -> int:
        return len(self.seq_obs)

    def __getitem__(self, idx: int):
        return (
            self.seq_obs[idx],
            self.seq_true[idx],
            self.mask[idx],
            self.static[idx],
        )


class SaitsImputer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 64, n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        h = self.encoder(h)
        return self.out_proj(h)


def _loss_imputed(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, recon_weight: float) -> torch.Tensor:
    missing = 1.0 - mask
    missing_sum = missing.sum()
    if missing_sum < 1:
        missing_sum = torch.tensor(1.0, device=pred.device)
    loss_missing = ((pred - target) ** 2 * missing).sum() / missing_sum
    if recon_weight <= 0.0:
        return loss_missing
    obs_sum = mask.sum()
    if obs_sum < 1:
        obs_sum = torch.tensor(1.0, device=pred.device)
    loss_obs = ((pred - target) ** 2 * mask).sum() / obs_sum
    return loss_missing + recon_weight * loss_obs


def _train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    recon_weight: float,
    log_prefix: str,
) -> nn.Module:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0

    for epoch in tqdm(range(1, epochs + 1), desc=f"Epochs{log_prefix}"):
        model.train()
        total = 0.0
        count = 0
        for seq_obs, seq_true, mask, static in train_loader:
            seq_obs = seq_obs.to(device)
            seq_true = seq_true.to(device)
            mask = mask.to(device)
            static = static.to(device)

            static_exp = static.unsqueeze(1).repeat(1, seq_obs.size(1), 1)
            x_in = torch.cat([seq_obs, mask, static_exp], dim=-1)

            pred = model(x_in)
            loss = _loss_imputed(pred, seq_true, mask, recon_weight)

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += float(loss.item())
            count += 1

        train_loss = total / max(1, count)

        if val_loader is None:
            continue

        model.eval()
        val_total = 0.0
        val_count = 0
        with torch.no_grad():
            for seq_obs, seq_true, mask, static in val_loader:
                seq_obs = seq_obs.to(device)
                seq_true = seq_true.to(device)
                mask = mask.to(device)
                static = static.to(device)

                static_exp = static.unsqueeze(1).repeat(1, seq_obs.size(1), 1)
                x_in = torch.cat([seq_obs, mask, static_exp], dim=-1)
                pred = model(x_in)
                loss = _loss_imputed(pred, seq_true, mask, recon_weight)

                val_total += float(loss.item())
                val_count += 1

        val_loss = val_total / max(1, val_count)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_grid(
    model: nn.Module,
    seq_obs: np.ndarray,
    mask: np.ndarray,
    static: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        seq_obs_t = torch.from_numpy(seq_obs).to(device)
        mask_t = torch.from_numpy(mask).to(device)
        stat_t = torch.from_numpy(static).to(device)
        stat_exp = stat_t.unsqueeze(1).repeat(1, seq_obs_t.size(1), 1)
        x_in = torch.cat([seq_obs_t, mask_t, stat_exp], dim=-1)
        pred = model(x_in).cpu().numpy().astype(np.float32)
    return pred


def _split_train_val(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    val_size = max(1, int(val_frac * n)) if n > 1 else 0
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]
    return tr_idx, val_idx


def _mask_reconstruct(
    model: nn.Module,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    derms: List[str],
    mapping: List[Tuple[int, int] | None],
    mask_frac: float,
    device: torch.device,
    target_cols: List[str],
    static_cols: List[str],
) -> dict:
    rng = np.random.default_rng(42)
    X_masked = X_train.copy()
    mask_artificial = np.zeros((len(X_train), len(target_cols)), dtype=bool)

    for j, col in enumerate(target_cols):
        if col == "anyana":
            continue
        if col not in X_masked.columns:
            continue
        obs = ~X_masked[col].isna().to_numpy()
        rand = rng.random(len(X_masked))
        m = (rand < mask_frac) & obs
        if m.any():
            X_masked.loc[m, col] = np.nan
            mask_artificial[:, j] = m

    seq_obs, mask = _build_sensory_arrays(X_masked, derms, SUFFIXES)
    Xtr_stat, _ = _prepare_static_features(X_train, X_train, static_cols)

    pred_grid = _predict_grid(model, seq_obs, mask, Xtr_stat, device)
    anyana_mean = float(y_train["anyana"].mean()) if "anyana" in target_cols else 0.0
    pred_vec = _grid_to_targets(pred_grid, mapping, anyana_mean)
    pred_vec = _clip_predictions(pred_vec, target_cols)

    true_vals = y_train[target_cols].to_numpy(dtype=np.float32)
    m = mask_artificial
    if m.sum() == 0:
        rmse = 0.0
        mae = 0.0
    else:
        rmse = float(np.sqrt(np.mean((true_vals[m] - pred_vec[m]) ** 2)))
        mae = float(np.mean(np.abs(true_vals[m] - pred_vec[m])))

    return {
        "mask_frac": mask_frac,
        "rmse_masked_only": rmse,
        "mae_masked_only": mae,
        "n_masked": int(m.sum()),
    }


def run_one(
    data_root: str,
    run_root: str,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    recon_weight: float,
    mask_frac: float,
    n_splits: int,
) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for Stage 4 SAITS run.")

    _set_seed(seed)

    data = load_track(1, data_root)
    target_cols = data.target_cols
    derms = _derive_dermatomes(target_cols)
    mapping = _build_target_mapping(target_cols, derms)

    Xtr = data.X_train
    Xte = data.X_test

    seq_obs_tr, mask_tr = _build_sensory_arrays(Xtr, derms, SUFFIXES)
    seq_true_tr, _ = _build_sensory_arrays(data.y_train, derms, SUFFIXES)
    seq_obs_te, mask_te = _build_sensory_arrays(Xte, derms, SUFFIXES)

    static_cols = list(dict.fromkeys(data.motor_cols + data.meta_cols + ["time"]))
    Xtr_stat, Xte_stat = _prepare_static_features(Xtr, Xte, static_cols)

    method = "t1_saits"
    run_id = make_run_id(prefix=method)
    out_dir = ensure_dir(Path(run_root) / run_id)

    folds = make_time_stratified_folds(Xtr["time"].to_numpy(), n_splits=n_splits, seed=seed)
    oof = np.zeros((len(Xtr), len(target_cols)), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(tqdm(folds, desc="Folds")):
        LOGGER.info("Fold %d: train=%d val=%d", fold, len(tr_idx), len(va_idx))
        seq_obs_f = seq_obs_tr[tr_idx]
        seq_true_f = seq_true_tr[tr_idx]
        mask_f = mask_tr[tr_idx]
        stat_f = Xtr_stat[tr_idx]

        seq_obs_val = seq_obs_tr[va_idx]
        mask_val = mask_tr[va_idx]
        stat_val = Xtr_stat[va_idx]

        tr_sub_idx, val_sub_idx = _split_train_val(len(seq_obs_f), val_frac=0.1, seed=seed + fold)
        train_ds = SaitsDataset(seq_obs_f[tr_sub_idx], seq_true_f[tr_sub_idx], mask_f[tr_sub_idx], stat_f[tr_sub_idx])
        val_ds = SaitsDataset(seq_obs_f[val_sub_idx], seq_true_f[val_sub_idx], mask_f[val_sub_idx], stat_f[val_sub_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        input_dim = seq_obs_f.shape[-1] + mask_f.shape[-1] + stat_f.shape[-1]
        model = SaitsImputer(input_dim=input_dim).to(device)

        model = _train(
            model,
            train_loader,
            val_loader,
            device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            recon_weight=recon_weight,
            log_prefix=f" (fold={fold}) ",
        )

        pred_val_grid = _predict_grid(model, seq_obs_val, mask_val, stat_val, device)
        anyana_mean = float(data.y_train.iloc[tr_idx]["anyana"].mean()) if "anyana" in target_cols else 0.0
        pred_val = _grid_to_targets(pred_val_grid, mapping, anyana_mean)
        pred_val = _clip_predictions(pred_val, target_cols)
        pred_val = _apply_copy_through(pred_val, Xtr.iloc[va_idx].reset_index(drop=True), target_cols)
        oof[va_idx] = pred_val

    oof_path = out_dir / "oof_predictions_train.npz"
    save_oof_npz(oof_path, ids=Xtr["ID"].to_numpy(), target_cols=target_cols, preds=oof)

    metrics = compute_imputed_only_metrics(data.y_train, oof, Xtr, target_cols, data.sensory_target_cols)
    breakdown = compute_imputed_only_breakdown(data.y_train, oof, Xtr, target_cols, data.sensory_target_cols)
    LOGGER.info("OOF imputed-only RMSE (sensory): %.6f", metrics.rmse_sensory_imputed_only)
    LOGGER.info("OOF imputed-only RMSE (all): %.6f", metrics.rmse_all_imputed_only)

    # Full-train model for test predictions
    full_tr_idx, full_val_idx = _split_train_val(len(seq_obs_tr), val_frac=0.1, seed=seed)
    train_ds = SaitsDataset(seq_obs_tr[full_tr_idx], seq_true_tr[full_tr_idx], mask_tr[full_tr_idx], Xtr_stat[full_tr_idx])
    val_ds = SaitsDataset(seq_obs_tr[full_val_idx], seq_true_tr[full_val_idx], mask_tr[full_val_idx], Xtr_stat[full_val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = seq_obs_tr.shape[-1] + mask_tr.shape[-1] + Xtr_stat.shape[-1]
    model = SaitsImputer(input_dim=input_dim).to(device)
    model = _train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=epochs,
        lr=lr,
        patience=patience,
        recon_weight=recon_weight,
        log_prefix=" (full) ",
    )

    pred_te_grid = _predict_grid(model, seq_obs_te, mask_te, Xte_stat, device)
    anyana_mean = float(data.y_train["anyana"].mean()) if "anyana" in target_cols else 0.0
    out_pred = _grid_to_targets(pred_te_grid, mapping, anyana_mean)
    out_pred = _clip_predictions(out_pred, target_cols)
    out_pred = _apply_copy_through(out_pred, Xte, target_cols)

    mask_metrics = _mask_reconstruct(
        model,
        Xtr,
        data.y_train,
        derms,
        mapping,
        mask_frac=mask_frac,
        device=device,
        target_cols=target_cols,
        static_cols=static_cols,
    )

    sub = data.sample_submission.copy()
    sub[ID_COL] = Xte[ID_COL].values
    for j, c in enumerate(target_cols):
        sub[c] = out_pred[:, j]

    sub.to_csv(out_dir / "predictions_test.csv", index=False)

    write_json(
        out_dir / "run_summary.json",
        {
            "run_id": run_id,
            "finished_utc": utc_now_iso(),
            "track": 1,
            "method": method,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "patience": patience,
            "recon_weight": recon_weight,
            "n_splits": n_splits,
            "mask_frac": mask_frac,
            "metrics": metrics.to_dict(),
            "metrics_breakdown": breakdown,
            "mask_reconstruct_metrics": mask_metrics,
            "static_cols": static_cols,
            "dermatomes": derms,
            "artifacts": {
                "submission_csv": "predictions_test.csv",
                "oof_npz": "oof_predictions_train.npz",
            },
        },
    )

    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--recon-weight", type=float, default=0.1)
    p.add_argument("--mask-frac", type=float, default=0.1)
    p.add_argument("--n-splits", type=int, default=5)
    args = p.parse_args()

    out_dir = run_one(
        data_root=args.data_root,
        run_root=args.run_root,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        recon_weight=args.recon_weight,
        mask_frac=args.mask_frac,
        n_splits=args.n_splits,
    )
    print(f"[asia2026 t1 saits] done -> {out_dir}")


if __name__ == "__main__":
    main()
