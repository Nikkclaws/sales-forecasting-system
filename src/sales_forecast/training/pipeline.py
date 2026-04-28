"""End-to-end training pipeline: walk-forward CV, model selection, ensembling, persistence."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import Config, load_config
from ..data import DataLoader, Preprocessor, WalkForwardSplitter
from ..evaluation import compute_metrics, explain_xgboost
from ..features import FeatureEngineer
from ..models import build_model
from ..models.conformal import ConformalCalibrator
from ..models.ensemble import WeightedEnsemble
from ..models.stacking import StackingMetaLearner
from ..utils.drift import detect_drift
from ..utils.io import save_joblib, save_json
from ..utils.logging import get_logger, setup_logging
from ..utils.seed import set_seed
from ..utils.versioning import ModelRegistry

log = get_logger(__name__)


@dataclass
class StateReport:
    state: str
    history_weeks: int
    cv_metrics: dict[str, dict[str, list[float]]] = field(default_factory=dict)
    aggregate_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    selected_models: list[str] = field(default_factory=list)
    ensemble_weights: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)
    explainability: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    duration_seconds: float = 0.0
    error: str | None = None


@dataclass
class TrainingReport:
    version: str
    created_at: float
    config_snapshot: dict
    states: dict[str, StateReport] = field(default_factory=dict)


class TrainingPipeline:
    """Train and evaluate every model on every state, then select the top-K ensemble."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        setup_logging(
            log_dir=self.cfg.path(self.cfg.project.log_dir),
            level=self.cfg.project.log_level,
            filename="train.log",
        )
        set_seed(self.cfg.project.random_seed)
        self.registry = ModelRegistry(self.cfg.path(self.cfg.project.registry_dir))

    # --- Public ----------------------------------------------------------- #

    def run(self, states: list[str] | None = None) -> TrainingReport:
        loader = DataLoader(self.cfg)
        df = loader.load()
        prep = Preprocessor(self.cfg)
        per_state = prep.transform(df)

        if states:
            per_state = {s: v for s, v in per_state.items() if s in states}
            if not per_state:
                raise ValueError(f"No requested states present after preprocessing: {states}")

        version = self.registry.new_version()
        log.info("Starting training run %s for %d states", version.version, len(per_state))

        report = TrainingReport(
            version=version.version,
            created_at=version.created_at,
            config_snapshot=self.cfg.model_dump(mode="json"),
        )
        for state, weekly in per_state.items():
            try:
                report.states[state] = self._run_state(state, weekly, version.path)
            except Exception as e:  # noqa: BLE001
                log.exception("State %s failed: %s", state, e)
                report.states[state] = StateReport(
                    state=state, history_weeks=len(weekly), error=f"{type(e).__name__}: {e}"
                )

        # Persist run-level metadata + manifest.
        self.registry.write_metadata(
            {
                "version": version.version,
                "created_at": version.created_at,
                "config": self.cfg.model_dump(mode="json"),
                "states": {
                    s: {
                        "selected_models": r.selected_models,
                        "ensemble_weights": r.ensemble_weights,
                        "aggregate_metrics": r.aggregate_metrics,
                        "history_weeks": r.history_weeks,
                        "error": r.error,
                    }
                    for s, r in report.states.items()
                },
            },
            version=version.version,
        )
        self.registry.set_current(version.version)
        # Save report
        save_json(_report_to_dict(report), version.path / "report.json")
        log.info("Run %s complete. Promoted to current.", version.version)
        return report

    # --- Per-state -------------------------------------------------------- #

    def _run_state(self, state: str, weekly: pd.DataFrame, version_path: Path) -> StateReport:
        t0 = time.time()
        target_col = self.cfg.data.target_col
        y = weekly[target_col].astype(float).copy()
        y.name = "y"
        rep = StateReport(state=state, history_weeks=len(y))

        # Optional log1p target transform for ML/DL models that benefit from variance stabilization.
        target_transform = self.cfg.features.target_transform
        y_for_ml = np.log1p(y.clip(lower=0)) if target_transform == "log1p" else y.copy()
        y_for_ml.name = "y"

        splitter = WalkForwardSplitter(self.cfg)
        folds = list(splitter.split(y.index))
        log.info("State %s: %d CV folds", state, len(folds))

        per_model_per_fold_rmse: dict[str, list[float]] = {}
        per_model_metrics: dict[str, dict[str, list[float]]] = {}
        # OOF predictions per model: list of (val_idx, pred_array, truth_array) per fold.
        oof_preds: dict[str, list[pd.Series]] = {}
        oof_truth: list[pd.Series] = []  # appended once per fold (raw truth aligned to val_idx)

        models_enabled = self.cfg.models.enabled
        for fold in folds:
            train_idx, val_idx = fold.train_idx, fold.val_idx
            y_train_raw = y.loc[train_idx]
            y_val_raw = y.loc[val_idx]
            oof_truth.append(y_val_raw)

            # ---- Train each model on this fold and score on val ---- #
            for model_name in models_enabled:
                try:
                    pred = self._fit_and_predict_fold(
                        model_name=model_name,
                        y_train_raw=y_train_raw,
                        y_train_ml=y_for_ml.loc[train_idx],
                        val_idx=val_idx,
                        target_transform=target_transform,
                    )
                    metrics = compute_metrics(y_val_raw.values, pred.values)
                    per_model_per_fold_rmse.setdefault(model_name, []).append(metrics["rmse"])
                    bucket = per_model_metrics.setdefault(model_name, {})
                    for k, v in metrics.items():
                        bucket.setdefault(k, []).append(v)
                    oof_preds.setdefault(model_name, []).append(
                        pd.Series(pred.values, index=val_idx, name=model_name)
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning(
                        "State=%s fold=%d model=%s failed during CV: %s\n%s",
                        state,
                        fold.fold_id,
                        model_name,
                        e,
                        traceback.format_exc(limit=3),
                    )

        rep.cv_metrics = per_model_metrics
        rep.aggregate_metrics = {
            m: {k: float(np.mean(v)) for k, v in metrics.items()} for m, metrics in per_model_metrics.items()
        }

        # ---- Rank models by mean RMSE; require at least min_validation_folds ---- #
        valid_models = {
            m: rmses
            for m, rmses in per_model_per_fold_rmse.items()
            if len(rmses) >= self.cfg.cv.min_validation_folds
        }
        if not valid_models:
            # Fall back to any model with at least one fold; fewer guarantees but still usable.
            valid_models = per_model_per_fold_rmse
        if not valid_models:
            raise RuntimeError(f"No model produced any successful CV fold for state {state}.")

        mean_rmse = {m: float(np.mean(v)) for m, v in valid_models.items()}
        ordered = sorted(mean_rmse.items(), key=lambda kv: kv[1])
        top_k = ordered[: self.cfg.ensemble.top_k]
        rep.selected_models = [m for m, _ in top_k]

        weighting_scheme = self.cfg.ensemble.weighting
        if weighting_scheme == "stacking":
            stacker = self._fit_stacker(rep.selected_models, oof_preds, oof_truth)
            if stacker is not None:
                rep.ensemble_weights = stacker.weights_dict
            else:
                # Fall back to inverse-RMSE if stacking fails (e.g., not enough OOF rows).
                weighting_scheme = "inverse_rmse"
        if weighting_scheme != "stacking":
            ensemble = WeightedEnsemble.from_scores(
                {m: s for m, s in mean_rmse.items() if m in rep.selected_models},
                top_k=self.cfg.ensemble.top_k,
                scheme=weighting_scheme,
            )
            rep.ensemble_weights = ensemble.weights

        # ---- Refit selected models on full history and persist artifacts ---- #
        state_dir = self.registry.state_dir(state, version=version_path.name)
        artifacts: dict[str, str] = {}

        # Engineer fit on the same scale that ML/DL models train on (log1p when enabled).
        engineer = FeatureEngineer(self.cfg, state)
        feats_full = engineer.fit_transform(y_for_ml.to_frame("y"), target_col="y")
        save_joblib(engineer, state_dir / "feature_engineer.joblib")
        artifacts["feature_engineer"] = str(state_dir / "feature_engineer.joblib")

        for model_name in rep.selected_models:
            try:
                fitted = self._fit_full(model_name, y_for_ml, target_transform, engineer, feats_full)
                model_path = state_dir / f"{model_name}.joblib"
                save_joblib(fitted, model_path)
                artifacts[model_name] = str(model_path)
            except Exception as e:  # noqa: BLE001
                log.warning("Final fit failed state=%s model=%s: %s", state, model_name, e)
                continue
            # SHAP runs in its own try-block so its failure can't drop the saved artifact.
            if model_name == "xgboost":
                try:
                    feats = engineer.transform_history()
                    X = feats[engineer.artifacts.feature_columns].dropna()
                    rep.explainability = explain_xgboost(
                        fitted.model,
                        X,
                        output_dir=state_dir / "shap",
                        state=state,
                    )
                except Exception as e:  # noqa: BLE001
                    log.warning("SHAP explainability failed state=%s: %s", state, e)

        # ---- Persist OOF predictions for backtest endpoint ---- #
        if oof_preds:
            try:
                truth_concat = pd.concat(oof_truth).sort_index()
                truth_concat.name = "y_true"
                rows = pd.DataFrame({"y_true": truth_concat})
                for m, series_list in oof_preds.items():
                    s = pd.concat(series_list).sort_index()
                    rows[m] = s.reindex(rows.index).values
                rows.index.name = "date"
                rows.to_csv(state_dir / "cv_predictions.csv")
                artifacts["cv_predictions"] = str(state_dir / "cv_predictions.csv")
            except Exception as e:  # noqa: BLE001
                log.warning("Failed to write cv_predictions.csv: %s", e)

        # ---- Conformal calibration: per-model, per-horizon ---- #
        if self.cfg.conformal.enabled and oof_preds:
            calibs: dict[str, dict] = {}
            alpha = self.cfg.conformal.alpha
            for m in rep.selected_models:
                if m not in oof_preds or len(oof_preds[m]) == 0:
                    continue
                # Stack residuals per horizon step across folds.
                horizon_steps = self.cfg.cv.horizon
                residuals_per_step: list[list[float]] = [[] for _ in range(horizon_steps)]
                for f_idx, pred_series in enumerate(oof_preds[m]):
                    truth = oof_truth[f_idx].reindex(pred_series.index)
                    res = (truth.values - pred_series.values).tolist()
                    for h, r in enumerate(res[:horizon_steps]):
                        if np.isfinite(r):
                            residuals_per_step[h].append(r)
                calib = ConformalCalibrator.from_residuals(
                    [np.asarray(r) for r in residuals_per_step], alpha=alpha
                )
                calibs[m] = {
                    "alpha": calib.alpha,
                    "half_widths": calib.half_widths.tolist(),
                }
            if calibs:
                save_json(calibs, state_dir / "conformal.json")
                artifacts["conformal"] = str(state_dir / "conformal.json")

        # ---- Drift detection: compare last `feature_window_weeks` to earlier history ---- #
        win = self.cfg.drift.feature_window_weeks
        if len(y) > 2 * win:
            ref = y.iloc[: len(y) - win]
            cur = y.iloc[-win:]
            rep.drift = detect_drift(
                ref,
                cur,
                psi_threshold=self.cfg.drift.psi_threshold,
                ks_pvalue_threshold=self.cfg.drift.ks_pvalue_threshold,
            )

        rep.artifacts = artifacts
        rep.duration_seconds = time.time() - t0
        log.info(
            "State %s done in %.1fs | top=%s | weights=%s",
            state,
            rep.duration_seconds,
            rep.selected_models,
            {k: round(v, 3) for k, v in rep.ensemble_weights.items()},
        )
        return rep

    # --- Helpers ---------------------------------------------------------- #

    def _fit_and_predict_fold(
        self,
        model_name: str,
        y_train_raw: pd.Series,
        y_train_ml: pd.Series,
        val_idx: pd.DatetimeIndex,
        target_transform: str | None,
    ) -> pd.Series:
        """Fit `model_name` on the training fold and predict for the val window."""
        horizon = len(val_idx)
        if model_name in {"arima", "sarima", "prophet"}:
            model = build_model(model_name, self.cfg)
            model.fit(y_train_raw)
            pred = model.forecast(horizon=horizon, ci_alpha=self.cfg.forecast.ci_alpha).mean
            pred.index = val_idx
            return pred

        if model_name == "xgboost":
            engineer = FeatureEngineer(self.cfg, state="_cv_")
            feats = engineer.fit_transform(y_train_ml.to_frame("y"), target_col="y")
            model = build_model("xgboost", self.cfg)
            model.fit(history=y_train_ml, exog=feats, engineer=engineer)
            mean = model.forecast(horizon=horizon, ci_alpha=self.cfg.forecast.ci_alpha).mean
            mean.index = val_idx
            return _inverse_transform(mean, target_transform)

        if model_name == "lstm":
            model = build_model("lstm", self.cfg, horizon=max(horizon, self.cfg.models.lstm.horizon))
            model.fit(history=y_train_ml)
            mean = model.forecast(horizon=horizon, ci_alpha=self.cfg.forecast.ci_alpha).mean
            mean.index = val_idx
            return _inverse_transform(mean, target_transform)

        raise KeyError(model_name)

    def _fit_stacker(
        self,
        selected: list[str],
        oof_preds: dict[str, list[pd.Series]],
        oof_truth: list[pd.Series],
    ) -> StackingMetaLearner | None:
        """Fit ridge-simplex stacking weights from OOF CV predictions."""
        try:
            usable = [m for m in selected if m in oof_preds and oof_preds[m]]
            if len(usable) < 2:
                return None
            # Concat OOF folds; align rows across models on shared val indices.
            per_model_oof = {m: pd.concat(oof_preds[m]).sort_index() for m in usable}
            truth = pd.concat(oof_truth).sort_index()
            shared = truth.index
            for m in usable:
                shared = shared.intersection(per_model_oof[m].index)
            if len(shared) < 4:
                return None
            X = pd.DataFrame({m: per_model_oof[m].loc[shared].values for m in usable}, index=shared)
            y = truth.loc[shared]
            return StackingMetaLearner(model_names=usable).fit(X, y)
        except Exception as e:  # noqa: BLE001
            log.warning("Stacker fit failed: %s", e)
            return None

    def _fit_full(
        self,
        model_name: str,
        y_for_ml: pd.Series,
        target_transform: str | None,
        engineer: FeatureEngineer,
        feats_full: pd.DataFrame,
    ):
        """Refit a chosen model on the entire history."""
        if model_name in {"arima", "sarima", "prophet"}:
            # Statistical models trained on raw scale.
            y_raw = np.expm1(y_for_ml) if target_transform == "log1p" else y_for_ml
            return build_model(model_name, self.cfg).fit(y_raw)
        if model_name == "xgboost":
            return build_model("xgboost", self.cfg).fit(history=y_for_ml, exog=feats_full, engineer=engineer)
        if model_name == "lstm":
            return build_model("lstm", self.cfg, horizon=self.cfg.models.lstm.horizon).fit(history=y_for_ml)
        raise KeyError(model_name)


def _inverse_transform(s: pd.Series, transform: str | None) -> pd.Series:
    if transform == "log1p":
        return np.expm1(s).clip(lower=0)
    return s


def _report_to_dict(report: TrainingReport) -> dict:
    return {
        "version": report.version,
        "created_at": report.created_at,
        "states": {
            s: {
                "state": r.state,
                "history_weeks": r.history_weeks,
                "selected_models": r.selected_models,
                "ensemble_weights": r.ensemble_weights,
                "aggregate_metrics": r.aggregate_metrics,
                "cv_metrics": r.cv_metrics,
                "artifacts": r.artifacts,
                "explainability": r.explainability,
                "drift": r.drift,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
            }
            for s, r in report.states.items()
        },
        "config": report.config_snapshot,
    }
