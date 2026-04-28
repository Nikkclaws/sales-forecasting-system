"""Common forecaster interface."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class ForecastResult:
    mean: pd.Series
    lower: pd.Series | None = None
    upper: pd.Series | None = None
    metadata: dict[str, Any] | None = None


class BaseForecaster:
    """All forecasters operate on a univariate weekly target.

    Subclasses may use exogenous features (XGBoost, LSTM) by reading from `exog`.
    """

    name: str = "base"
    supports_exog: bool = False

    def fit(
        self,
        history: pd.Series,
        exog: pd.DataFrame | None = None,
    ) -> BaseForecaster:  # pragma: no cover - abstract
        raise NotImplementedError

    def forecast(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
        ci_alpha: float = 0.1,
    ) -> ForecastResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def save(self, path: str | Path) -> None:  # pragma: no cover - implemented per model
        from joblib import dump

        dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> BaseForecaster:  # pragma: no cover
        from joblib import load

        return load(path)
