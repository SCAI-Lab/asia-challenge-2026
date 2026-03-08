#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from asia2026.data import load_track
from asia2026.saits_pypots_t1 import build_sens_grid, build_target_mapping, grid_to_targets


def main() -> None:
    data = load_track(1, "data/staged")
    target_cols = data.target_cols
    mapping = build_target_mapping(target_cols)

    y = data.y_train.reset_index(drop=True)
    grid = build_sens_grid(y)
    dummy_anyana = np.zeros((len(y),), dtype=np.float32)

    vec = grid_to_targets(grid, mapping, dummy_anyana)

    sens_cols = [c for c in target_cols if c != "anyana"]
    idxs = [target_cols.index(c) for c in sens_cols]
    y_true = y[sens_cols].to_numpy(np.float32)
    y_back = vec[:, idxs]

    max_abs = float(np.nanmax(np.abs(y_true - y_back)))
    print("max_abs_diff(sensory) =", max_abs)
    if max_abs >= 1e-6:
        raise SystemExit("FAIL: mapping/grid construction mismatch!")


if __name__ == "__main__":
    main()
