"""SARIMA wrapper around statsmodels SARIMAX (no exogenous regressors here)."""

from __future__ import annotations

import warnings

import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseForecaster, ForecastResult


class SARIMAForecaster(BaseForecaster):
    name = "sarima"

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 1, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 52),
        enforce_stationarity: bool = False,
        enforce_invertibility: bool = False,
    ):
        self.order = tuple(order)
        self.seasonal_order = tuple(seasonal_order)
        self.enforce_stationarity = enforce_stationarity
        self.enforce_invertibility = enforce_invertibility
        self._fitted = None
        self._last_index: pd.Timestamp | None = None
        self._freq: str | None = None

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> SARIMAForecaster:
        y = history.astype(float)
        self._freq = pd.infer_freq(y.index) or y.index.freqstr
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                y,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=self.enforce_stationarity,
                enforce_invertibility=self.enforce_invertibility,
            )
            self._fitted = model.fit(disp=False, maxiter=200)
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
            metadata={"order": list(self.order), "seasonal_order": list(self.seasonal_order)},
        )
