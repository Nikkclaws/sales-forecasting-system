"""SHAP explainability for the XGBoost forecaster.

Uses XGBoost's native `pred_contribs=True` predict path to compute exact tree-SHAP
values. This sidesteps the well-known SHAP/XGBoost JSON `base_score` incompatibility
that surfaces with bleeding-edge XGBoost versions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from ..utils.io import ensure_dir
from ..utils.logging import get_logger

log = get_logger(__name__)


def _shap_values_xgb(model, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Compute exact tree-SHAP values via XGBoost's native predict API.

    Returns (shap_values, base_value). `shap_values` shape == X.shape; the
    bias (base value) column produced by XGBoost is dropped here and returned
    separately so we can use it in waterfall-style plots if needed.
    """
    booster = model.get_booster()
    dmat = xgb.DMatrix(X.values, feature_names=list(X.columns))
    contribs = booster.predict(dmat, pred_contribs=True)
    # contribs is shape (n, n_features + 1); the trailing column is the bias term.
    shap_values = contribs[:, :-1]
    base_value = float(contribs[:, -1].mean())
    return shap_values, base_value


def explain_xgboost(
    model,
    X: pd.DataFrame,
    output_dir: str | Path,
    state: str,
    max_display: int = 20,
) -> dict:
    """Compute SHAP via XGBoost native API and persist summary + bar plots.

    Falls back to the `shap` library's TreeExplainer if available; otherwise
    uses the xgboost-native path. Plots use matplotlib only.
    """
    out = ensure_dir(output_dir)
    summary_path = out / f"shap_summary_{state}.png"
    bar_path = out / f"feature_importance_{state}.png"

    shap_values, base_value = _shap_values_xgb(model, X)

    # 1) Beeswarm-ish summary: per-feature, plot SHAP vs feature value (colored by feature value).
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    feat_names = list(X.columns)

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(order))))
    for plot_row, idx in enumerate(order):
        sv = shap_values[:, idx]
        fv = X.iloc[:, idx].values.astype(float)
        # Normalize feature values for color
        color_norm = (fv - fv.min()) / np.ptp(fv) if np.ptp(fv) > 0 else np.zeros_like(fv)
        y_jitter = plot_row + np.random.uniform(-0.15, 0.15, size=sv.shape[0])
        ax.scatter(sv, y_jitter, c=color_norm, cmap="coolwarm", alpha=0.7, s=12, edgecolor="none")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feat_names[i] for i in order])
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("SHAP value (impact on log-target)")
    ax.set_title(f"SHAP summary - {state} (top {len(order)} features)")
    sm = plt.cm.ScalarMappable(cmap="coolwarm")
    sm.set_array([0, 1])
    cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("Normalized feature value", rotation=270, labelpad=12)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=120)
    plt.close(fig)

    # 2) Bar plot of mean(|SHAP|) for the top features.
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(order))))
    top_names = [feat_names[i] for i in order[::-1]]
    top_vals = mean_abs[order][::-1]
    ax.barh(top_names, top_vals, color="#2c7fb8")
    ax.set_xlabel("Mean |SHAP|")
    ax.set_title(f"Feature importance (mean |SHAP|) - {state}")
    plt.tight_layout()
    plt.savefig(bar_path, dpi=120)
    plt.close(fig)

    log.info("Wrote SHAP plots: %s, %s", summary_path, bar_path)
    return {
        "summary_plot": str(summary_path),
        "bar_plot": str(bar_path),
        "base_value": base_value,
        "top_features": dict(
            zip(
                [feat_names[i] for i in order[: min(10, len(order))]],
                mean_abs[order[: min(10, len(order))]].tolist(),
                strict=False,
            )
        ),
    }
