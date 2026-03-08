#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

ID_COL = "ID"


def clip_ranges(df, target_cols):
    df[target_cols] = df[target_cols].clip(0.0, 2.0)
    if "anyana" in df.columns:
        df["anyana"] = df["anyana"].clip(0.0, 1.0)
    return df


def restore_observed(pred_df, features_test_df, target_cols):
    ft = features_test_df[[ID_COL] + [c for c in target_cols if c in features_test_df.columns]].copy()
    merged = pred_df.merge(ft, on=ID_COL, how="left", suffixes=("", "_feat"))
    for c in target_cols:
        c_feat = f"{c}_feat"
        if c_feat in merged.columns:
            obs = ~merged[c_feat].isna()
            merged.loc[obs, c] = merged.loc[obs, c_feat]
            merged.drop(columns=[c_feat], inplace=True)
    return merged


def greedy_discretize_012(x, t01, t12):
    x = np.asarray(x, dtype=np.float32)
    out = np.zeros_like(x, dtype=np.float32)
    out[x >= t01] = 1.0
    out[x >= t12] = 2.0
    return out


def greedy_discretize_binary(x, thr=0.5):
    x = np.asarray(x, dtype=np.float32)
    return (x >= thr).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--features_test_csv", required=True)
    ap.add_argument("--labels_train_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--dummy_csv", default=None, help="optional: enforce exact Kaggle column order")
    ap.add_argument("--t01", type=float, default=0.5, help="threshold between class 0 and 1")
    ap.add_argument("--t12", type=float, default=1.5, help="threshold between class 1 and 2")
    ap.add_argument("--binary_thr", type=float, default=0.5, help="threshold for binary columns (anyana)")
    args = ap.parse_args()

    pred = pd.read_csv(args.pred_csv)
    ft = pd.read_csv(args.features_test_csv)
    ytr = pd.read_csv(args.labels_train_csv)

    target_cols = [c for c in ytr.columns if c != ID_COL]
    missing = [c for c in [ID_COL] + target_cols if c not in pred.columns]
    if missing:
        raise ValueError(f"pred_csv missing required columns (first 10 shown): {missing[:10]} (total {len(missing)})")

    pred = clip_ranges(pred, target_cols)
    pred = restore_observed(pred, ft, target_cols)

    for c in target_cols:
        if c == "anyana":
            pred[c] = greedy_discretize_binary(pred[c].to_numpy(), thr=args.binary_thr)
        else:
            pred[c] = greedy_discretize_012(pred[c].to_numpy(), t01=args.t01, t12=args.t12)

    pred = restore_observed(pred, ft, target_cols)

    if args.dummy_csv is not None:
        dummy = pd.read_csv(args.dummy_csv)
        pred = pred[dummy.columns]
    else:
        pred = pred[[ID_COL] + target_cols]

    if pred.isna().any().any():
        raise ValueError("NaNs detected in output. Abort.")
    pred.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)


if __name__ == "__main__":
    main()
