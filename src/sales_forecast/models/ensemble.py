"""Dynamic weighted ensemble of top-K forecasters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import ForecastResult


class WeightedEnsemble:
    """Combine multiple ForecastResults using user-defined weights."""

    def __init__(self, weights: dict[str, float]):
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Ensemble weights must sum to a positive number.")
        self.weights = {k: v / total for k, v in weights.items()}

    def combine(self, results: dict[str, ForecastResult]) -> ForecastResult:
        models = [m for m in self.weights if m in results]
        if not models:
            raise ValueError("None of the requested models were available.")
        idx = results[models[0]].mean.index
        mean = pd.Series(0.0, index=idx)
        lower = pd.Series(0.0, index=idx)
        upper = pd.Series(0.0, index=idx)
        any_lower = any(results[m].lower is not None for m in models)
        any_upper = any(results[m].upper is not None for m in models)
        for m in models:
            w = self.weights[m]
            mean = mean + w * results[m].mean.reindex(idx).values
            if any_lower:
                fb_low = results[m].mean if results[m].lower is None else results[m].lower
                lower = lower + w * fb_low.reindex(idx).values
            if any_upper:
                fb_up = results[m].mean if results[m].upper is None else results[m].upper
                upper = upper + w * fb_up.reindex(idx).values
        return ForecastResult(
            mean=mean,
            lower=lower if any_lower else None,
            upper=upper if any_upper else None,
            metadata={"weights": self.weights, "members": models},
        )

    @staticmethod
    def from_scores(
        scores: dict[str, float], top_k: int = 2, scheme: str = "inverse_rmse"
    ) -> WeightedEnsemble:
        """Build an ensemble by selecting the top-K lowest-error models and weighting them."""
        if not scores:
            raise ValueError("Empty scores; cannot build ensemble.")
        ordered = sorted(scores.items(), key=lambda kv: kv[1])
        top = ordered[:top_k]
        names = [n for n, _ in top]
        vals = np.array([v for _, v in top], dtype=float)
        if scheme == "equal":
            weights = {n: 1.0 / len(names) for n in names}
        elif scheme == "softmax":
            # Lower errors get higher weights via -score softmax.
            x = -vals
            x = x - x.max()
            ex = np.exp(x)
            ex /= ex.sum()
            weights = dict(zip(names, ex.tolist(), strict=False))
        else:  # inverse_rmse (default)
            inv = 1.0 / np.maximum(vals, 1e-9)
            inv /= inv.sum()
            weights = dict(zip(names, inv.tolist(), strict=False))
        return WeightedEnsemble(weights)
