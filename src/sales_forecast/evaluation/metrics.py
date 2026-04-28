"""Forecast evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_numpy(a) -> np.ndarray:
    if isinstance(a, pd.Series | pd.DataFrame):
        return a.to_numpy(dtype=float)
    return np.asarray(a, dtype=float)


def rmse(y_true, y_pred) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true, y_pred) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true, y_pred, eps: float = 1e-8) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    denom = np.where(np.abs(yt) < eps, eps, np.abs(yt))
    return float(np.mean(np.abs((yt - yp) / denom)) * 100.0)


def smape(y_true, y_pred, eps: float = 1e-8) -> float:
    yt, yp = _to_numpy(y_true), _to_numpy(y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2.0
    denom = np.where(denom < eps, eps, denom)
    return float(np.mean(np.abs(yt - yp) / denom) * 100.0)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
