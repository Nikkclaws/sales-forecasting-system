"""Reusable feature engineering for weekly sales series."""

from __future__ import annotations

from dataclasses import dataclass, field

import holidays as holidays_lib
import numpy as np
import pandas as pd

from ..config import Config


@dataclass
class FeatureEngineerArtifacts:
    """Per-state artifacts captured at fit-time and reused at transform/forecast-time."""

    state: str
    feature_columns: list[str] = field(default_factory=list)
    target_col: str = "y"
    last_observed_date: pd.Timestamp | None = None
    history_target: pd.Series | None = None  # full history of raw target (used for lag forecasts)
    trend_origin: pd.Timestamp | None = None  # for linear trend
    changepoints: list[pd.Timestamp] = field(default_factory=list)


class FeatureEngineer:
    """Generate lag, rolling, trend, Fourier, calendar and holiday features."""

    def __init__(self, cfg: Config, state: str):
        self.cfg = cfg
        self.state = state
        self.artifacts = FeatureEngineerArtifacts(state=state)
        self._holiday_calendar = holidays_lib.country_holidays(cfg.features.holidays.country)

    # --- Public ----------------------------------------------------------- #

    def fit_transform(self, weekly: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Fit artifacts on the training window and return engineered features."""
        self.artifacts.target_col = target_col
        self.artifacts.history_target = weekly[target_col].copy()
        self.artifacts.last_observed_date = weekly.index.max()
        self.artifacts.trend_origin = weekly.index.min()
        self.artifacts.changepoints = self._compute_changepoints(weekly.index)
        return self._build(weekly[target_col])

    def transform_history(self) -> pd.DataFrame:
        """Re-emit features for the stored history (after fit_transform was called)."""
        if self.artifacts.history_target is None:
            raise RuntimeError("Call fit_transform first.")
        return self._build(self.artifacts.history_target)

    def make_future_frame(self, horizon: int) -> pd.DatetimeIndex:
        """Future weekly index, anchored at the same frequency as training."""
        if self.artifacts.last_observed_date is None:
            raise RuntimeError("Call fit_transform first.")
        return pd.date_range(
            start=self.artifacts.last_observed_date + pd.tseries.frequencies.to_offset(self.cfg.data.freq),
            periods=horizon,
            freq=self.cfg.data.freq,
        )

    def transform_for_forecast(
        self, future_targets: pd.Series, full_history: pd.Series | None = None
    ) -> pd.DataFrame:
        """Build features for a forecast horizon.

        `future_targets` is a Series indexed by future dates with whatever target values
        are currently known (NaN for unknown). For recursive forecasting, the caller fills
        values one step at a time and re-invokes this method.
        """
        history = full_history if full_history is not None else self.artifacts.history_target
        if history is None:
            raise RuntimeError("Missing history; call fit_transform first.")
        combined = pd.concat([history, future_targets]).sort_index()
        combined = combined[~combined.index.duplicated(keep="last")]
        feats = self._build(combined)
        return feats.loc[future_targets.index]

    # --- Internals -------------------------------------------------------- #

    def _build(self, y: pd.Series) -> pd.DataFrame:
        idx = y.index
        feats = pd.DataFrame(index=idx)
        feats[self.artifacts.target_col] = y.values

        # Lag features (lags are in *weeks* since we are on a weekly grid).
        for lag in self.cfg.features.lags:
            feats[f"lag_{lag}"] = y.shift(lag)

        # Rolling stats (computed on lag-1 to avoid target leakage).
        shifted = y.shift(1)
        for w in self.cfg.features.rolling_windows:
            roll = shifted.rolling(window=w, min_periods=max(2, w // 2))
            for stat in self.cfg.features.rolling_stats:
                feats[f"roll_{stat}_{w}"] = getattr(roll, stat)()

        # Calendar features
        feats["week"] = idx.isocalendar().week.astype(int).values
        feats["month"] = idx.month.values
        feats["quarter"] = idx.quarter.values
        feats["year"] = idx.year.values

        # Trend features
        if self.cfg.features.trend.include_linear:
            origin = self.artifacts.trend_origin or idx.min()
            feats["trend_lin"] = (idx - origin).days.astype(float) / 7.0
        if self.cfg.features.trend.include_changepoints and self.artifacts.changepoints:
            for i, cp in enumerate(self.artifacts.changepoints):
                feats[f"cp_{i}"] = np.maximum(0.0, (idx - cp).days.astype(float) / 7.0)

        # Fourier seasonality
        feats = self._add_fourier(feats, idx)

        # Holiday flags
        feats = self._add_holidays(feats, idx)

        # Persist column order (excluding the target itself)
        cols = [c for c in feats.columns if c != self.artifacts.target_col]
        self.artifacts.feature_columns = cols
        return feats

    def _add_fourier(self, feats: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        origin = self.artifacts.trend_origin or idx.min()
        t = (idx - origin).days.astype(float) / 7.0
        fc = self.cfg.features.fourier
        if fc.yearly:
            for k in range(1, fc.yearly.order + 1):
                arg = 2 * np.pi * k * t / fc.yearly.period
                feats[f"fy_sin_{k}"] = np.sin(arg)
                feats[f"fy_cos_{k}"] = np.cos(arg)
        if fc.quarterly:
            for k in range(1, fc.quarterly.order + 1):
                arg = 2 * np.pi * k * t / fc.quarterly.period
                feats[f"fq_sin_{k}"] = np.sin(arg)
                feats[f"fq_cos_{k}"] = np.cos(arg)
        return feats

    def _add_holidays(self, feats: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        cfg_h = self.cfg.features.holidays
        flags = []
        days_to_next = []
        for d in idx:
            week_start = d - pd.Timedelta(days=6)
            in_week = any((week_start.date() <= h <= d.date()) for h in self._holidays_between(week_start, d))
            flags.append(int(in_week))
            if cfg_h.add_distance_to_next_holiday:
                future_holidays = self._holidays_between(d, d + pd.Timedelta(days=180))
                if future_holidays:
                    days_to_next.append((min(future_holidays) - d.date()).days)
                else:
                    days_to_next.append(180)
        feats["holiday_in_week"] = flags
        if cfg_h.add_distance_to_next_holiday:
            feats["days_to_next_holiday"] = days_to_next
        return feats

    def _holidays_between(self, start: pd.Timestamp, end: pd.Timestamp) -> list:
        # The `holidays` library lazy-loads each year on first access. Iterating
        # over the calendar's keys yields nothing until you've explicitly queried
        # at least one date in each year, so we walk the date range directly.
        out: list = []
        for d in pd.date_range(start, end):
            dd = d.date()
            if dd in self._holiday_calendar:
                out.append(dd)
        return out

    def _compute_changepoints(self, idx: pd.DatetimeIndex) -> list[pd.Timestamp]:
        if not self.cfg.features.trend.include_changepoints:
            return []
        n = self.cfg.features.trend.n_changepoints
        if len(idx) < n + 1:
            return []
        # Place changepoints uniformly across first 80% of the series.
        cutoff = int(len(idx) * 0.8)
        positions = np.linspace(1, cutoff - 1, n).round().astype(int)
        return [pd.Timestamp(idx[p]) for p in positions]
