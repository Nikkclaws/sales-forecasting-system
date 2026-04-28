"""ARIMA wrapper around statsmodels."""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .base import BaseForecaster, ForecastResult


class ARIMAForecaster(BaseForecaster):
    name = "arima"

    def __init__(self, order: tuple[int, int, int] = (2, 1, 2)):
        self.order = tuple(order)
        self._fitted = None
        self._last_index: pd.Timestamp | None = None
        self._freq: str | None = None

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> ARIMAForecaster:
        y = history.astype(float)
        self._freq = pd.infer_freq(y.index) or y.index.freqstr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ARIMA(y, order=self.order, enforce_stationarity=False, enforce_invertibility=False)
            self._fitted = model.fit()
        self._last_index = y.index.max()
        return self

    def forecast(
        self, horizon: int, exog_future: pd.DataFrame | None = None, ci_alpha: float = 0.1
    ) -> ForecastResult:
        if self._fitted is None:
            raise RuntimeError("Call fit() first.")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = self._fitted.get_forecast(steps=horizon)
        mean = res.predicted_mean
        ci = res.conf_int(alpha=ci_alpha)
        future_idx = pd.date_range(
            start=self._last_index + pd.tseries.frequencies.to_offset(self._freq or "W-SUN"),
            periods=horizon,
            freq=self._freq or "W-SUN",
        )
        mean.index = future_idx
        lower = pd.Series(ci.iloc[:, 0].values, index=future_idx)
        upper = pd.Series(ci.iloc[:, 1].values, index=future_idx)
        return ForecastResult(
            mean=mean.astype(float),
            lower=lower.astype(float),
            upper=upper.astype(float),
            metadata={"order": list(self.order)},
        )
