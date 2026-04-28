"""Model interface smoke tests."""

from __future__ import annotations

import numpy as np

from sales_forecast.features import FeatureEngineer
from sales_forecast.models import ARIMAForecaster, SARIMAForecaster, XGBoostForecaster


def test_arima_fits_and_forecasts(synthetic_weekly_series):
    m = ARIMAForecaster(order=(1, 1, 1)).fit(synthetic_weekly_series)
    res = m.forecast(horizon=8)
    assert len(res.mean) == 8
    assert res.lower is not None and res.upper is not None
    # CIs should bracket the mean
    assert (res.lower.values <= res.mean.values + 1e-6).all()
    assert (res.upper.values >= res.mean.values - 1e-6).all()


def test_sarima_fits_and_forecasts(synthetic_weekly_series):
    # Trim seasonal_order m to series length for speed.
    m = SARIMAForecaster(order=(1, 0, 0), seasonal_order=(1, 0, 0, 12)).fit(synthetic_weekly_series)
    res = m.forecast(horizon=8)
    assert len(res.mean) == 8


def test_xgboost_fits_and_forecasts(cfg, synthetic_weekly_series, monkeypatch):
    # Speed up Optuna for the test
    cfg.models.xgboost.optuna_trials = 3
    cfg.models.xgboost.optuna_timeout_seconds = 30
    fe = FeatureEngineer(cfg, state="_test_")
    feats = fe.fit_transform(synthetic_weekly_series.to_frame("y"), target_col="y")
    model = XGBoostForecaster(
        optuna_trials=3,
        optuna_timeout_seconds=30,
        early_stopping_rounds=20,
        random_state=cfg.project.random_seed,
    )
    model.fit(history=synthetic_weekly_series, exog=feats, engineer=fe)
    res = model.forecast(horizon=8)
    assert len(res.mean) == 8
    assert np.isfinite(res.mean.values).all()
