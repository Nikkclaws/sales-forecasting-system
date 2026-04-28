"""Per-state preprocessing: reindex to weekly grid, hybrid imputation, outlier handling."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils.logging import get_logger

log = get_logger(__name__)


class Preprocessor:
    """Reshape long sales frame -> per-state weekly series with imputed/cleaned target."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    # --- Public API ------------------------------------------------------- #

    def transform(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Return {state -> per-state weekly DataFrame[index=date, target, outlier_flag]}."""
        d = self.cfg.data
        out: dict[str, pd.DataFrame] = {}
        for state, grp in df.groupby(d.state_col, sort=True):
            weekly = self._aggregate_to_freq(grp)
            weekly = self._reindex_full_range(weekly)
            n_missing = weekly[d.target_col].isna().sum()
            weekly = self._impute(weekly)
            weekly, n_outliers = self._handle_outliers(weekly)
            if len(weekly) < self.cfg.data.min_history_weeks:
                log.warning(
                    "Skipping state %s: only %d weeks (< min_history_weeks=%d)",
                    state,
                    len(weekly),
                    self.cfg.data.min_history_weeks,
                )
                continue
            log.info(
                "State %-22s | weeks=%d | imputed=%d | outliers=%d",
                state,
                len(weekly),
                int(n_missing),
                int(n_outliers),
            )
            out[str(state)] = weekly
        log.info("Preprocessing complete: %d states retained", len(out))
        return out

    # --- Stages ----------------------------------------------------------- #

    def _aggregate_to_freq(self, grp: pd.DataFrame) -> pd.DataFrame:
        d = self.cfg.data
        s = grp.set_index(d.date_col)[d.target_col].resample(self.cfg.data.freq).sum(min_count=1)
        return s.to_frame(name=d.target_col)

    def _reindex_full_range(self, weekly: pd.DataFrame) -> pd.DataFrame:
        if weekly.empty:
            return weekly
        full_idx = pd.date_range(
            start=weekly.index.min(),
            end=weekly.index.max(),
            freq=self.cfg.data.freq,
        )
        return weekly.reindex(full_idx)

    def _impute(self, weekly: pd.DataFrame) -> pd.DataFrame:
        c = self.cfg.preprocessing.imputation
        d = self.cfg.data
        s = weekly[d.target_col]
        # Stage 1: linear interpolation for short gaps
        s = s.interpolate(method=c.interpolation_method, limit=c.interpolation_limit, limit_direction="both")
        # Stage 2: forward-fill for medium gaps
        s = s.ffill(limit=c.forward_fill_limit)
        # Stage 3: back-fill for any remaining leading NaNs
        s = s.bfill()
        weekly[d.target_col] = s
        return weekly

    def _handle_outliers(self, weekly: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        d = self.cfg.data
        o = self.cfg.preprocessing.outliers
        s = weekly[d.target_col].astype(float)

        # IQR
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        iqr_low, iqr_high = q1 - o.iqr_multiplier * iqr, q3 + o.iqr_multiplier * iqr

        # Z-score (robust to scale via mean/std on log-transformed series for skewed data)
        log_s = np.log1p(s.clip(lower=0))
        mu, sd = log_s.mean(), log_s.std(ddof=0)
        z = (log_s - mu) / (sd if sd > 0 else 1.0)

        flag = (s < iqr_low) | (s > iqr_high) | (z.abs() > o.zscore_threshold)
        n = int(flag.sum())
        weekly = weekly.copy()
        weekly["outlier_flag"] = flag.astype(int)

        if o.strategy == "cap":
            weekly[d.target_col] = s.clip(lower=iqr_low, upper=iqr_high)
        # 'mark' leaves values intact but keeps the flag.
        return weekly, n
