#!/usr/bin/env python3

"""Unpack the bundled Track 1 / Track 2 zip files into the expected data layout.

This script is intentionally dependency-light and safe to run repeatedly.
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def _extract_zip(zip_path: Path, out_dir: Path) -> None:
    tmp = out_dir / "_tmp_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    # Remove common macOS junk
    macosx = tmp / "__MACOSX"
    if macosx.exists():
        shutil.rmtree(macosx)

    # The shared zips contain a single top folder (Share_Track1 or Share_Track2)
    children = [p for p in tmp.iterdir() if p.is_dir()]
    if len(children) == 1:
        src = children[0]
    else:
        # fallback: use tmp
        src = tmp

    out_dir.mkdir(parents=True, exist_ok=True)
    # Move CSVs into out_dir without overwriting existing files.
    for csv in src.rglob("*.csv"):
        dest = out_dir / csv.name
        if dest.exists():
            continue
        shutil.copy2(csv, dest)

    shutil.rmtree(tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track1-zip", type=str, required=True)
    ap.add_argument("--track2-zip", type=str, required=True)
    ap.add_argument("--out-root", type=str, required=True)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    _extract_zip(Path(args.track1_zip), out_root / "track1")
    _extract_zip(Path(args.track2_zip), out_root / "track2")

    print(f"[unpack_data] done -> {out_root}")


if __name__ == "__main__":
    main()
