#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from asia2026.data import load_track
from asia2026.saits_pypots_t1 import build_sens_grid
from asia2026.sci_dermatomes import DERMS_28, SUFFIXES_4


def main() -> None:
    data = load_track(1, "data/staged")
    y = data.y_train.reset_index(drop=True)
    grid = build_sens_grid(y)

    always_nan = np.all(np.isnan(grid), axis=0)
    print("Always-NaN slots in grid:", int(always_nan.sum()))

    for di, derm in enumerate(DERMS_28):
        for ci, suf in enumerate(SUFFIXES_4):
            if always_nan[di, ci]:
                print("missing slot:", f"{derm}{suf}")


if __name__ == "__main__":
    main()
