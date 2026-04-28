"""Walk-forward cross-validation splits for univariate weekly series."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

from ..config import Config


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_idx: pd.DatetimeIndex
    val_idx: pd.DatetimeIndex


class WalkForwardSplitter:
    """Expanding-window walk-forward splits.

    Produces up to `cv.max_folds` folds by stepping `cv.step` weeks forward,
    starting from `cv.initial_train_weeks`. Each validation window is exactly
    `cv.horizon` weeks. The final fold uses up the remainder of the series.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def split(self, idx: pd.DatetimeIndex) -> Iterator[Fold]:
        cv = self.cfg.cv
        n = len(idx)
        if n < cv.initial_train_weeks + cv.horizon:
            return
        start_train_end = cv.initial_train_weeks
        fold_id = 0
        while True:
            train_end = start_train_end + fold_id * cv.step
            val_start = train_end
            val_end = val_start + cv.horizon
            if val_end > n or fold_id >= cv.max_folds:
                break
            yield Fold(
                fold_id=fold_id,
                train_idx=idx[:train_end],
                val_idx=idx[val_start:val_end],
            )
            fold_id += 1
