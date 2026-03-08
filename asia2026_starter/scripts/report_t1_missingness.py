#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from asia2026.utils import ensure_dir, make_run_id, utc_now_iso, write_json

try:
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit("matplotlib is required for plotting missingness histograms") from exc

LOGGER = logging.getLogger(__name__)


def _load_track1_features(data_root: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    tdir = Path(data_root) / "track1"
    feats_tr = pd.read_csv(tdir / "features_train_1.csv")
    feats_te = pd.read_csv(tdir / "features_test_1.csv")
    y_tr = pd.read_csv(tdir / "labels_train_1.csv")
    target_cols = [c for c in y_tr.columns if c != "ID"]

    missing_tr = [c for c in target_cols if c not in feats_tr.columns]
    missing_te = [c for c in target_cols if c not in feats_te.columns]
    if missing_tr or missing_te:
        raise ValueError(
            f"Missing target columns in features "
            f"(train_missing={missing_tr[:5]}, test_missing={missing_te[:5]})"
        )

    return feats_tr, feats_te, target_cols


def _missingness_df(df: pd.DataFrame, target_cols: List[str]) -> pd.DataFrame:
    total = len(df)
    missing_counts = df[target_cols].isna().sum()
    missing_pct = (missing_counts / float(total) * 100.0).astype(np.float32)
    present_counts = total - missing_counts
    out = pd.DataFrame(
        {
            "target": missing_counts.index,
            "missing_pct": missing_pct.values,
            "missing_count": missing_counts.values,
            "present_count": present_counts.values,
            "total_count": total,
        }
    )
    return out.sort_values("missing_pct", ascending=False, ignore_index=True)


def _plot_hist(missing_pct: np.ndarray, title: str, out_path: Path, bins: np.ndarray) -> None:
    plt.figure(figsize=(8, 4.8))
    plt.hist(missing_pct, bins=bins, color="#4C78A8", edgecolor="black", alpha=0.85)
    plt.title(title)
    plt.xlabel("Missingness (%)")
    plt.ylabel("Target count")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_bar(
    missing_df: pd.DataFrame,
    title: str,
    out_path: Path,
    label_commas: bool = True,
) -> None:
    labels = missing_df["target"].astype(str).tolist()
    if label_commas and labels:
        labels = [f"{name}," for name in labels[:-1]] + [labels[-1]]
    values = missing_df["missing_pct"].to_numpy()
    x = np.arange(len(values))
    fig_w = max(12.0, len(labels) * 0.2)
    plt.figure(figsize=(fig_w, 6.0))
    bars = plt.bar(x, values, width=0.95, color="#F58518", edgecolor="black", linewidth=0.3)
    plt.title(title)
    plt.ylabel("Missingness (%)")
    plt.ylim(0, 100)
    plt.xticks(x, labels, rotation=90, fontsize=6)
    for rect, val in zip(bars, values):
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height() + 0.3,
            f"{val:,.2f}",
            ha="center",
            va="bottom",
            fontsize=5,
            rotation=90,
        )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _print_topk(label: str, df: pd.DataFrame, k: int) -> None:
    topk = df.head(k)
    LOGGER.info("Top-%d most-missing targets (%s):", k, label)
    print(f"\nTop-{k} most-missing targets ({label}):")
    print(topk.to_string(index=False))


def _write_table_txt(df: pd.DataFrame, out_path: Path, title: str) -> None:
    text = f"{title}\n" + df.to_string(index=False) + "\n"
    out_path.write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to staged data root")
    ap.add_argument("--run-root", default="runs", help="Output root for the report files")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--bins", type=int, default=20)
    args = ap.parse_args()

    feats_tr, feats_te, target_cols = _load_track1_features(args.data_root)

    run_id = make_run_id(prefix="t1_missingness_report")
    out_dir = ensure_dir(Path(args.run_root) / run_id)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    root_logger.handlers = [console, file_handler]

    train_df = _missingness_df(feats_tr, target_cols)
    test_df = _missingness_df(feats_te, target_cols)

    train_csv = out_dir / "missingness_train.csv"
    test_csv = out_dir / "missingness_test.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    topk = int(args.topk)
    train_top_csv = out_dir / "missingness_train_topk.csv"
    test_top_csv = out_dir / "missingness_test_topk.csv"
    train_df.head(topk).to_csv(train_top_csv, index=False)
    test_df.head(topk).to_csv(test_top_csv, index=False)

    train_txt = out_dir / "missingness_train.txt"
    test_txt = out_dir / "missingness_test.txt"
    _write_table_txt(train_df, train_txt, "Track 1 missingness (train)")
    _write_table_txt(test_df, test_txt, "Track 1 missingness (test)")

    bins = np.linspace(0.0, 100.0, int(args.bins) + 1)
    train_hist = out_dir / "missingness_train_hist.png"
    test_hist = out_dir / "missingness_test_hist.png"
    _plot_hist(
        train_df["missing_pct"].to_numpy(),
        "Track 1 Missingness Histogram (Train)",
        train_hist,
        bins,
    )
    _plot_hist(
        test_df["missing_pct"].to_numpy(),
        "Track 1 Missingness Histogram (Test)",
        test_hist,
        bins,
    )

    train_bar = out_dir / "missingness_train_bar.png"
    test_bar = out_dir / "missingness_test_bar.png"
    _plot_bar(
        train_df,
        "Track 1 Missingness by Target (Train)",
        train_bar,
    )
    _plot_bar(
        test_df,
        "Track 1 Missingness by Target (Test)",
        test_bar,
    )

    _print_topk("train", train_df, topk)
    _print_topk("test", test_df, topk)

    summary = {
        "run_id": run_id,
        "finished_utc": utc_now_iso(),
        "track": 1,
        "method": "t1_missingness_report",
        "data_root": args.data_root,
        "num_targets": len(target_cols),
        "train_always_missing": int((train_df["missing_pct"] >= 100.0).sum()),
        "test_always_missing": int((test_df["missing_pct"] >= 100.0).sum()),
        "artifacts": {
            "missingness_train_csv": str(train_csv.name),
            "missingness_test_csv": str(test_csv.name),
            "missingness_train_topk_csv": str(train_top_csv.name),
            "missingness_test_topk_csv": str(test_top_csv.name),
            "missingness_train_txt": str(train_txt.name),
            "missingness_test_txt": str(test_txt.name),
            "missingness_train_hist_png": str(train_hist.name),
            "missingness_test_hist_png": str(test_hist.name),
            "missingness_train_bar_png": str(train_bar.name),
            "missingness_test_bar_png": str(test_bar.name),
            "run_log": "run.log",
        },
    }
    write_json(out_dir / "run_summary.json", summary)

    LOGGER.info("Wrote missingness report to %s", out_dir)
    print(f"[t1_missingness_report] done -> {out_dir}")
    print("Artifacts:")
    print(f"  run_summary.json: {out_dir / 'run_summary.json'}")
    print(f"  train_hist: {train_hist}")
    print(f"  test_hist: {test_hist}")
    print(f"  train_bar: {train_bar}")
    print(f"  test_bar: {test_bar}")
    print(f"  train_csv: {train_csv}")
    print(f"  test_csv: {test_csv}")
    print(f"  train_topk_csv: {train_top_csv}")
    print(f"  test_topk_csv: {test_top_csv}")
    print(f"  train_txt: {train_txt}")
    print(f"  test_txt: {test_txt}")
    print(f"  run_log: {out_dir / 'run.log'}")


if __name__ == "__main__":
    main()
