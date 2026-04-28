"""Prophet wrapper with country holidays and optional regressors."""

from __future__ import annotations

import contextlib
import logging

import pandas as pd

from .base import BaseForecaster, ForecastResult

# Prophet pulls in a noisy stan logger; quiet it before the import takes effect.
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)


class ProphetForecaster(BaseForecaster):
    name = "prophet"
    supports_exog = True

    def __init__(
        self,
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        seasonality_mode: str = "multiplicative",
        interval_width: float = 0.9,
        holidays_country: str = "US",
        regressor_columns: list[str] | None = None,
    ):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.seasonality_mode = seasonality_mode
        self.interval_width = interval_width
        self.holidays_country = holidays_country
        self.regressor_columns = regressor_columns or []
        self._model = None
        self._freq: str = "W-SUN"
        self._exog_columns: list[str] = []

    def fit(self, history: pd.Series, exog: pd.DataFrame | None = None) -> ProphetForecaster:
        from prophet import Prophet  # imported lazily for faster module load

        df = pd.DataFrame({"ds": history.index, "y": history.astype(float).values})
        self._freq = pd.infer_freq(history.index) or history.index.freqstr or "W-SUN"

        m = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
        )
        with contextlib.suppress(Exception):  # holidays-country mismatch shouldn't fail training
            m.add_country_holidays(country_name=self.holidays_country)

        if exog is not None and self.regressor_columns:
            self._exog_columns = [c for c in self.regressor_columns if c in exog.columns]
            for col in self._exog_columns:
                m.add_regressor(col)
            df = df.merge(
                exog[self._exog_columns].reset_index().rename(columns={"index": "ds"}),
                left_on="ds",
                right_on="ds",
                how="left",
            )
            df[self._exog_columns] = df[self._exog_columns].ffill().bfill()
        m.fit(df)
        self._model = m
        return self

    def forecast(
        self, horizon: int, exog_future: pd.DataFrame | None = None, ci_alpha: float = 0.1
    ) -> ForecastResult:
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        future = self._model.make_future_dataframe(periods=horizon, freq=self._freq, include_history=False)
        if self._exog_columns and exog_future is not None:
            ex = exog_future[self._exog_columns].copy()
            ex.index.name = "ds"
            ex = ex.reset_index()
            future = future.merge(ex, on="ds", how="left").ffill().bfill()
        fc = self._model.predict(future)
        mean = pd.Series(fc["yhat"].values, index=pd.DatetimeIndex(fc["ds"]))
        lower = pd.Series(fc["yhat_lower"].values, index=mean.index)
        upper = pd.Series(fc["yhat_upper"].values, index=mean.index)
        return ForecastResult(
            mean=mean.astype(float),
            lower=lower.astype(float),
            upper=upper.astype(float),
            metadata={"interval_width": self.interval_width},
        )
