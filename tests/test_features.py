"""Feature engineering tests on synthetic data."""

from __future__ import annotations

import pandas as pd

from sales_forecast.features import FeatureEngineer


def test_feature_engineer_produces_expected_columns(cfg, synthetic_weekly_series):
    fe = FeatureEngineer(cfg, state="_test_")
    feats = fe.fit_transform(synthetic_weekly_series.to_frame("y"), target_col="y")
    expected = {f"lag_{lag}" for lag in cfg.features.lags}
    expected |= {
        f"roll_{stat}_{w}" for w in cfg.features.rolling_windows for stat in cfg.features.rolling_stats
    }
    expected |= {"holiday_in_week"}
    assert expected.issubset(feats.columns)
    # last n rows have lag features filled
    first_lag = cfg.features.lags[0]
    assert feats[f"lag_{first_lag}"].iloc[-1] == synthetic_weekly_series.iloc[-1 - first_lag]


def test_make_future_frame(cfg, synthetic_weekly_series):
    fe = FeatureEngineer(cfg, state="_test_")
    fe.fit_transform(synthetic_weekly_series.to_frame("y"), target_col="y")
    future = fe.make_future_frame(horizon=8)
    assert len(future) == 8
    assert future.min() > synthetic_weekly_series.index.max()
    assert isinstance(future, pd.DatetimeIndex)
