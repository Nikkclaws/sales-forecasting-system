"""Lightweight drift detection: PSI on the target distribution + KS-test on residuals."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI between reference and current distributions, with quantile binning."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    ref_pct = np.where(ref_hist == 0, 1e-6, ref_hist / ref_hist.sum())
    cur_pct = np.where(cur_hist == 0, 1e-6, cur_hist / cur_hist.sum())
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def detect_drift(
    reference: pd.Series,
    current: pd.Series,
    psi_threshold: float = 0.2,
    ks_pvalue_threshold: float = 0.05,
) -> dict:
    """Return PSI + KS-test results and a boolean drift flag."""
    psi = population_stability_index(reference.values, current.values)
    ks_stat, ks_p = stats.ks_2samp(reference.values, current.values)
    drifted = bool((psi > psi_threshold) or (ks_p < ks_pvalue_threshold))
    return {
        "psi": float(psi),
        "psi_threshold": psi_threshold,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
        "ks_pvalue_threshold": ks_pvalue_threshold,
        "drifted": drifted,
    }
