"""Metric and drift utility tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sales_forecast.evaluation import compute_metrics
from sales_forecast.utils.drift import detect_drift, population_stability_index


def test_metrics_basic():
    y = np.array([100.0, 110.0, 90.0])
    yp = np.array([101.0, 108.0, 88.0])
    m = compute_metrics(y, yp)
    assert {"rmse", "mae", "mape", "smape"}.issubset(m.keys())
    assert m["rmse"] >= 0


def test_psi_and_drift():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 1000))
    cur_same = pd.Series(rng.normal(0, 1, 1000))
    cur_shift = pd.Series(rng.normal(2, 1, 1000))
    psi_same = population_stability_index(ref.values, cur_same.values)
    psi_shift = population_stability_index(ref.values, cur_shift.values)
    assert psi_shift > psi_same
    out = detect_drift(ref, cur_shift, psi_threshold=0.2, ks_pvalue_threshold=0.05)
    assert out["drifted"] is True
