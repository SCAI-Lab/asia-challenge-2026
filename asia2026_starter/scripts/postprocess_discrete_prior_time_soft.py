#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

ID_COL = "ID"
TIME_COL = "time"


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def clip_pred(df, target_cols):
    df[target_cols] = df[target_cols].clip(0.0, 2.0)
    if "anyana" in df.columns:
        df["anyana"] = df["anyana"].clip(0.0, 1.0)
    return df


def restore_observed(pred_df, features_test_df, target_cols):
    ft = features_test_df[[ID_COL] + [c for c in target_cols if c in features_test_df.columns]].copy()
    merged = pred_df.merge(ft, on=ID_COL, how="left", suffixes=("", "_feat"))
    for c in target_cols:
        cf = f"{c}_feat"
        if cf in merged.columns:
            obs = ~merged[cf].isna()
            merged.loc[obs, c] = merged.loc[obs, cf]
            merged.drop(columns=[cf], inplace=True)
    return merged


def compute_train_priors_by_time(y_train, time_train, target_cols):
    priors = {}
    for t in sorted(time_train.unique()):
        idx = (time_train == t).to_numpy()
        y_t = y_train.loc[idx, :]
        for c in target_cols:
            vals = y_t[c].to_numpy()
            uniq = set(np.unique(vals).tolist())
            if uniq.issubset({0, 1}):
                p0 = float(np.mean(vals == 0))
                p1 = float(np.mean(vals == 1))
                priors[(t, c)] = (p0, p1, None)
            else:
                p0 = float(np.mean(vals == 0))
                p1 = float(np.mean(vals == 1))
                p2 = float(np.mean(vals == 2))
                priors[(t, c)] = (p0, p1, p2)
    return priors


def prior_matched_thresholds(pred_vals, p0, p1):
    t01 = float(np.quantile(pred_vals, p0))
    t12 = float(np.quantile(pred_vals, p0 + p1))
    if t12 < t01:
        t12 = t01
    return t01, t12


def soft_probs_from_thresholds(y, t01, t12, tau):
    y = np.asarray(y, dtype=np.float32)
    p0 = sigmoid((t01 - y) / tau)
    p2 = sigmoid((y - t12) / tau)
    p1 = 1.0 - p0 - p2
    p1 = np.clip(p1, 0.0, 1.0)
    s = p0 + p1 + p2
    p0, p1, p2 = p0 / s, p1 / s, p2 / s
    return p0, p1, p2


def discrete_from_probs(p0, p1, p2):
    probs = np.stack([p0, p1, p2], axis=1)
    return np.argmax(probs, axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--features_test_csv", required=True)
    ap.add_argument("--labels_train_csv", required=True)
    ap.add_argument("--features_train_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--dummy_csv", default=None)
    ap.add_argument("--tau", type=float, default=0.15)
    ap.add_argument(
        "--binary_thr_fallback",
        type=float,
        default=0.5,
        help="used if a binary column has no imputed cells in a time group",
    )
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)
    ft_te = pd.read_csv(args.features_test_csv)
    ytr = pd.read_csv(args.labels_train_csv)
    ft_tr = pd.read_csv(args.features_train_csv)

    target_cols = [c for c in ytr.columns if c != ID_COL]

    for c in [ID_COL, TIME_COL]:
        if c not in ft_te.columns:
            raise ValueError(f"{args.features_test_csv} missing required column: {c}")
        if c not in ft_tr.columns:
            raise ValueError(f"{args.features_train_csv} missing required column: {c}")

    ft_tr_idx = ft_tr.set_index(ID_COL)
    ytr_ids = ytr[ID_COL].to_numpy()
    time_tr = ft_tr_idx.loc[ytr_ids, TIME_COL].reset_index(drop=True)

    ft_te_idx = ft_te.set_index(ID_COL)
    pred_ids = pred[ID_COL].to_numpy()
    time_te = ft_te_idx.loc[pred_ids, TIME_COL].reset_index(drop=True)

    pred = clip_pred(pred, target_cols)
    priors = compute_train_priors_by_time(ytr[target_cols], time_tr, target_cols)

    ft_te_targets = ft_te_idx.loc[pred_ids, [c for c in target_cols if c in ft_te_idx.columns]]
    masks = {}
    for c in target_cols:
        if c in ft_te_targets.columns:
            masks[c] = ft_te_targets[c].isna().to_numpy()
        else:
            masks[c] = np.ones(len(pred), dtype=bool)

    out = pred.copy()
    for tval in sorted(pd.Series(time_te).unique()):
        idx_t = (pd.Series(time_te) == tval).to_numpy()
        if idx_t.sum() == 0:
            continue

        for c in target_cols:
            p = priors.get((tval, c), None)
            if p is None:
                continue

            p0, p1, p2 = p
            mask_imputed = masks[c] & idx_t
            if mask_imputed.sum() == 0:
                continue

            vals = out.loc[mask_imputed, c].to_numpy(dtype=np.float32)

            if p2 is None:
                thr = float(np.quantile(vals, p0)) if mask_imputed.sum() > 0 else args.binary_thr_fallback
                out.loc[mask_imputed, c] = (vals >= thr).astype(np.float32)
                continue

            t01, t12 = prior_matched_thresholds(vals, p0, p1)
            p0v, p1v, p2v = soft_probs_from_thresholds(vals, t01, t12, tau=args.tau)
            out.loc[mask_imputed, c] = discrete_from_probs(p0v, p1v, p2v)

    out = restore_observed(out, ft_te.reset_index(drop=False), target_cols)

    if args.dummy_csv is not None:
        dummy = pd.read_csv(args.dummy_csv)
        out = out[dummy.columns]
    else:
        out = out[[ID_COL] + target_cols]

    if out.isna().any().any():
        raise ValueError("NaNs detected in output. Abort.")
    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()
