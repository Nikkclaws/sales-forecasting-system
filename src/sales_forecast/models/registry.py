"""Factory for instantiating models from config."""

from __future__ import annotations

from typing import Any

from ..config import Config
from .arima_model import ARIMAForecaster
from .base import BaseForecaster
from .lstm_model import LSTMForecaster
from .prophet_model import ProphetForecaster
from .sarima_model import SARIMAForecaster
from .xgboost_model import XGBoostForecaster

_FACTORY: dict[str, Any] = {
    "arima": ARIMAForecaster,
    "sarima": SARIMAForecaster,
    "prophet": ProphetForecaster,
    "xgboost": XGBoostForecaster,
    "lstm": LSTMForecaster,
}


def available_models() -> list[str]:
    return list(_FACTORY.keys())


def build_model(name: str, cfg: Config, **overrides: Any) -> BaseForecaster:
    name = name.lower()
    if name not in _FACTORY:
        raise KeyError(f"Unknown model {name!r}. Available: {available_models()}")

    if name == "arima":
        c = cfg.models.arima
        return ARIMAForecaster(order=tuple(c.order), **overrides)
    if name == "sarima":
        c = cfg.models.sarima
        return SARIMAForecaster(
            order=tuple(c.order),
            seasonal_order=tuple(c.seasonal_order),
            enforce_stationarity=c.enforce_stationarity,
            enforce_invertibility=c.enforce_invertibility,
            **overrides,
        )
    if name == "prophet":
        c = cfg.models.prophet
        return ProphetForecaster(
            yearly_seasonality=c.yearly_seasonality,
            weekly_seasonality=c.weekly_seasonality,
            daily_seasonality=c.daily_seasonality,
            seasonality_mode=c.seasonality_mode,
            interval_width=c.interval_width,
            holidays_country=c.holidays_country,
            **overrides,
        )
    if name == "xgboost":
        c = cfg.models.xgboost
        return XGBoostForecaster(
            optuna_trials=c.optuna_trials,
            optuna_timeout_seconds=c.optuna_timeout_seconds,
            early_stopping_rounds=c.early_stopping_rounds,
            base_params=c.base_params,
            random_state=cfg.project.random_seed,
            **overrides,
        )
    if name == "lstm":
        c = cfg.models.lstm
        # Allow caller to override horizon via overrides dict.
        horizon = overrides.pop("horizon", c.horizon)
        return LSTMForecaster(
            sequence_length=c.sequence_length,
            horizon=horizon,
            hidden_size=c.hidden_size,
            num_layers=c.num_layers,
            dropout=c.dropout,
            batch_size=c.batch_size,
            epochs=c.epochs,
            learning_rate=c.learning_rate,
            patience=c.patience,
            random_state=cfg.project.random_seed,
            **overrides,
        )
    raise KeyError(name)  # pragma: no cover - guarded above
