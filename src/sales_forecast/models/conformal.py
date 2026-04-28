"""Split-conformal prediction wrapper for distribution-free, finite-sample-valid CIs.

Given any base forecaster + a calibration window of (y_true, y_pred) pairs,
the conformal half-width at miscoverage alpha is the (1 - alpha) quantile of
absolute residuals. Apply that half-width symmetrically around any mean
forecast to obtain valid prediction intervals.

Reference: Vovk et al. 2005; Romano, Patterson, Candes 2019 (CQR).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import ForecastResult


@dataclass
class ConformalCalibrator:
    """Per-horizon conformal half-widths computed from CV fold residuals."""

    half_widths: np.ndarray  # shape (H,) in target units
    alpha: float

    @classmethod
    def from_residuals(
        cls, residuals_per_horizon: list[np.ndarray], alpha: float = 0.1
    ) -> ConformalCalibrator:
        """Compute the (1 - alpha)-quantile of |residuals| at each horizon step.

        `residuals_per_horizon` is a list with one ndarray per horizon step;
        each ndarray contains the residuals from each CV fold at that step.
        """
        widths = np.array(
            [float(np.quantile(np.abs(r), 1 - alpha)) if len(r) else 0.0 for r in residuals_per_horizon]
        )
        return cls(half_widths=widths, alpha=alpha)

    def apply(self, mean: pd.Series) -> ForecastResult:
        n = len(mean)
        widths = self.half_widths
        if len(widths) < n:
            # Repeat the last calibrated step if horizon exceeds calibration.
            widths = np.concatenate([widths, np.full(n - len(widths), widths[-1] if len(widths) else 0.0)])
        widths = widths[:n]
        lower = pd.Series(mean.values - widths, index=mean.index)
        upper = pd.Series(mean.values + widths, index=mean.index)
        return ForecastResult(
            mean=mean,
            lower=lower,
            upper=upper,
            metadata={"calibration": "split-conformal", "alpha": self.alpha},
        )
