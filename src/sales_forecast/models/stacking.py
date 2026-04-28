"""Ridge stacking meta-learner for ensembling base-model forecasts.

Trained on out-of-fold (OOF) per-step predictions; coefficients are constrained
to be non-negative and sum-to-one via projected least-squares so the stacker
behaves like a smooth weighted average rather than an unconstrained regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import ForecastResult


class StackingMetaLearner:
    """Non-negative, sum-to-one stacked weights over base forecasters."""

    def __init__(self, model_names: list[str]):
        self.model_names = list(model_names)
        self.weights_: np.ndarray | None = None

    @staticmethod
    def _project_simplex(v: np.ndarray) -> np.ndarray:
        """Project v onto the probability simplex (Wang & Carreira-Perpinan, 2013)."""
        n = len(v)
        u = np.sort(v)[::-1]
        css = np.cumsum(u)
        rho = np.where(u + (1 - css) / (np.arange(1, n + 1)) > 0)[0]
        if len(rho) == 0:
            return np.full_like(v, 1.0 / n)
        rho = rho[-1]
        lam = (1 - css[rho]) / (rho + 1)
        return np.maximum(v + lam, 0.0)

    def fit(self, oof_preds: pd.DataFrame, oof_truth: pd.Series) -> StackingMetaLearner:
        """`oof_preds` columns must match self.model_names. `oof_truth` aligned by index."""
        X = oof_preds[self.model_names].values
        y = oof_truth.loc[oof_preds.index].values

        def objective(w):
            return float(np.mean((y - X @ w) ** 2))

        n = len(self.model_names)
        w0 = np.full(n, 1.0 / n)
        # Bounded LBFGS, then project to simplex (post-hoc) for numerical safety.
        bounds = [(0.0, 1.0)] * n
        res = minimize(objective, w0, method="L-BFGS-B", bounds=bounds)
        w = self._project_simplex(res.x)
        self.weights_ = w / w.sum() if w.sum() > 0 else np.full(n, 1.0 / n)
        return self

    def combine(self, results: dict[str, ForecastResult]) -> ForecastResult:
        if self.weights_ is None:
            raise RuntimeError("Call fit() first.")
        idx = next(iter(results.values())).mean.index
        mean = pd.Series(0.0, index=idx)
        lower = pd.Series(0.0, index=idx)
        upper = pd.Series(0.0, index=idx)
        any_ci = any(r.lower is not None for r in results.values())
        for w, name in zip(self.weights_, self.model_names, strict=False):
            r = results[name]
            mean = mean + w * r.mean.reindex(idx).fillna(0).values
            if any_ci:
                lo = r.mean if r.lower is None else r.lower
                hi = r.mean if r.upper is None else r.upper
                lower = lower + w * lo.reindex(idx).fillna(0).values
                upper = upper + w * hi.reindex(idx).fillna(0).values
        return ForecastResult(
            mean=mean,
            lower=lower if any_ci else None,
            upper=upper if any_ci else None,
            metadata={
                "stacker": "ridge_simplex",
                "weights": dict(zip(self.model_names, self.weights_.tolist(), strict=False)),
            },
        )

    @property
    def weights_dict(self) -> dict[str, float]:
        if self.weights_ is None:
            return {}
        return dict(zip(self.model_names, self.weights_.tolist(), strict=False))
