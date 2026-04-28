"""Service layer wrapping the trained registry for the API."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import Config
from ..models import ForecastResult
from ..models.conformal import ConformalCalibrator
from ..models.ensemble import WeightedEnsemble
from ..training import TrainingPipeline
from ..utils.io import load_json
from ..utils.logging import get_logger
from ..utils.versioning import ModelRegistry

log = get_logger(__name__)


def _safe_float(x) -> float | None:
    v = float(x)
    return v if np.isfinite(v) else None


def _result_to_points(result: ForecastResult) -> list[dict]:
    """Convert a ForecastResult into a JSON-friendly list of points."""
    return [
        {
            "date": ts.strftime("%Y-%m-%d"),
            "yhat": _safe_float(result.mean.iloc[i]) or 0.0,
            "yhat_lower": _safe_float(result.lower.iloc[i]) if result.lower is not None else None,
            "yhat_upper": _safe_float(result.upper.iloc[i]) if result.upper is not None else None,
        }
        for i, ts in enumerate(result.mean.index)
    ]


def _sanitize(result: ForecastResult) -> ForecastResult:
    """Replace NaN/Inf in forecast/CIs with safe interpolated values."""
    mean = pd.Series(
        np.nan_to_num(result.mean.values, nan=0.0, posinf=0.0, neginf=0.0), index=result.mean.index
    )

    def _fix_bound(b: pd.Series | None, fallback: pd.Series) -> pd.Series | None:
        if b is None:
            return None
        vals = np.where(np.isfinite(b.values), b.values, fallback.values)
        return pd.Series(vals, index=b.index)

    lower = _fix_bound(result.lower, mean)
    upper = _fix_bound(result.upper, mean)
    if lower is not None and upper is not None:
        # Ensure lower <= mean <= upper element-wise.
        lo = np.minimum(lower.values, mean.values)
        hi = np.maximum(upper.values, mean.values)
        lower = pd.Series(lo, index=lower.index)
        upper = pd.Series(hi, index=upper.index)
    return ForecastResult(mean=mean, lower=lower, upper=upper, metadata=result.metadata)


@dataclass
class TrainJob:
    job_id: str
    status: str = "pending"  # pending | running | succeeded | failed
    started_at: float | None = None
    finished_at: float | None = None
    version: str | None = None
    error: str | None = None
    states_trained: int = 0


@dataclass
class StateBundle:
    state: str
    selected_models: list[str]
    ensemble_weights: dict[str, float]
    feature_engineer: Any
    models: dict[str, Any]
    aggregate_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    drift: dict[str, Any] | None = None
    history_weeks: int = 0
    conformal: dict[str, ConformalCalibrator] = field(default_factory=dict)


class ForecastService:
    """Loads artifacts on demand and serves predictions."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.registry = ModelRegistry(cfg.path(cfg.project.registry_dir))
        self._lock = threading.RLock()
        self._jobs: dict[str, TrainJob] = {}
        self._bundles: dict[str, StateBundle] = {}
        self._loaded_version: str | None = None
        self._metadata: dict = {}

    # --- Lifecycle -------------------------------------------------------- #

    def reload(self) -> None:
        with self._lock:
            self._bundles.clear()
            current = self.registry.current_version()
            self._loaded_version = current
            if current is None:
                log.warning("No trained version available yet.")
                self._metadata = {}
                return
            self._metadata = self.registry.read_metadata(current)
            log.info("Loaded registry version %s", current)

    def is_ready(self) -> bool:
        with self._lock:
            return self._loaded_version is not None

    @property
    def loaded_version(self) -> str | None:
        with self._lock:
            return self._loaded_version

    @property
    def metadata(self) -> dict:
        with self._lock:
            return dict(self._metadata)

    def states_available(self) -> list[str]:
        meta_states = self.metadata.get("states", {})
        return sorted([s for s, info in meta_states.items() if info.get("selected_models")])

    # --- Training jobs ---------------------------------------------------- #

    def submit_training(self, states: list[str] | None = None) -> TrainJob:
        job = TrainJob(job_id=str(uuid.uuid4()), status="pending")
        with self._lock:
            self._jobs[job.job_id] = job
        thread = threading.Thread(target=self._run_training, args=(job, states), daemon=True)
        thread.start()
        return job

    def get_job(self, job_id: str) -> TrainJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def _run_training(self, job: TrainJob, states: list[str] | None) -> None:
        job.status = "running"
        job.started_at = time.time()
        try:
            pipe = TrainingPipeline(self.cfg)
            report = pipe.run(states=states)
            job.version = report.version
            job.states_trained = len(report.states)
            job.status = "succeeded"
        except Exception as e:  # noqa: BLE001
            log.exception("Training job %s failed: %s", job.job_id, e)
            job.status = "failed"
            job.error = f"{type(e).__name__}: {e}"
        finally:
            job.finished_at = time.time()
            self.reload()

    # --- Predictions ------------------------------------------------------ #

    def _load_state_bundle(self, state: str) -> StateBundle:
        with self._lock:
            if state in self._bundles:
                return self._bundles[state]
            if self._loaded_version is None:
                raise RuntimeError("No trained version is loaded; call /train first.")
            meta_state = self._metadata.get("states", {}).get(state)
            if not meta_state or not meta_state.get("selected_models"):
                raise KeyError(f"State {state!r} has no trained models.")
            state_dir = self.registry.state_dir(state, version=self._loaded_version)
            engineer_path = state_dir / "feature_engineer.joblib"
            if not engineer_path.exists():
                raise FileNotFoundError(f"Missing feature_engineer for state {state}.")
            engineer = joblib.load(engineer_path)
            models: dict[str, Any] = {}
            for m in meta_state["selected_models"]:
                p = state_dir / f"{m}.joblib"
                if p.exists():
                    models[m] = joblib.load(p)
                else:
                    log.warning("Missing artifact %s; skipping in ensemble.", p)
            if not models:
                raise FileNotFoundError(f"No model artifacts for state {state}.")
            # Conformal calibrators (per-model, per-horizon half-widths).
            conformal_path = state_dir / "conformal.json"
            conformal: dict[str, ConformalCalibrator] = {}
            if conformal_path.exists():
                raw = load_json(conformal_path)
                for name, payload in raw.items():
                    if name in models:
                        conformal[name] = ConformalCalibrator(
                            half_widths=np.asarray(payload["half_widths"], dtype=float),
                            alpha=float(payload["alpha"]),
                        )
            bundle = StateBundle(
                state=state,
                selected_models=list(models.keys()),
                ensemble_weights={k: v for k, v in meta_state["ensemble_weights"].items() if k in models},
                feature_engineer=engineer,
                models=models,
                aggregate_metrics=meta_state.get("aggregate_metrics", {}),
                drift=self._read_drift_for_state(state),
                history_weeks=int(meta_state.get("history_weeks", 0)),
                conformal=conformal,
            )
            self._bundles[state] = bundle
            return bundle

    def _read_drift_for_state(self, state: str) -> dict | None:
        # Drift was already saved into the report.json; we read it lazily.
        report_path = self.registry.version_path(self._loaded_version) / "report.json"
        if not report_path.exists():
            return None
        try:
            data = load_json(report_path)
            return data.get("states", {}).get(state, {}).get("drift")
        except Exception:  # noqa: BLE001
            return None

    def predict(
        self,
        state: str,
        horizon: int | None = None,
        ci_alpha: float | None = None,
        use_conformal: bool | None = None,
    ) -> dict:
        horizon = horizon or self.cfg.forecast.horizon_weeks
        ci_alpha = ci_alpha if ci_alpha is not None else self.cfg.forecast.ci_alpha
        use_conformal = bool(self.cfg.conformal.enabled if use_conformal is None else use_conformal)
        bundle = self._load_state_bundle(state)

        target_transform = self.cfg.features.target_transform
        results = self._forecast_all_members(bundle, horizon, ci_alpha, target_transform, use_conformal)
        if not results:
            raise RuntimeError(f"All ensemble members failed to forecast for state {state}.")
        ensemble = WeightedEnsemble(bundle.ensemble_weights)
        combined = _sanitize(ensemble.combine(results))

        forecast_points = _result_to_points(combined)
        return {
            "state": state,
            "registry_version": self._loaded_version,
            "horizon_weeks": horizon,
            "selected_models": bundle.selected_models,
            "ensemble_weights": bundle.ensemble_weights,
            "ci_method": "conformal" if (use_conformal and bundle.conformal) else "model_native",
            "forecast": forecast_points,
            "drift": bundle.drift,
        }

    def predict_breakdown(
        self,
        state: str,
        horizon: int | None = None,
        ci_alpha: float | None = None,
        use_conformal: bool | None = None,
    ) -> dict:
        """Return ensemble + each member's individual forecast for transparency."""
        horizon = horizon or self.cfg.forecast.horizon_weeks
        ci_alpha = ci_alpha if ci_alpha is not None else self.cfg.forecast.ci_alpha
        use_conformal = bool(self.cfg.conformal.enabled if use_conformal is None else use_conformal)
        bundle = self._load_state_bundle(state)
        target_transform = self.cfg.features.target_transform
        results = self._forecast_all_members(bundle, horizon, ci_alpha, target_transform, use_conformal)
        if not results:
            raise RuntimeError(f"All ensemble members failed to forecast for state {state}.")
        ensemble = WeightedEnsemble(bundle.ensemble_weights)
        combined = _sanitize(ensemble.combine(results))
        return {
            "state": state,
            "registry_version": self._loaded_version,
            "horizon_weeks": horizon,
            "ensemble_weights": bundle.ensemble_weights,
            "ci_method": "conformal" if (use_conformal and bundle.conformal) else "model_native",
            "ensemble": _result_to_points(combined),
            "members": {name: _result_to_points(_sanitize(r)) for name, r in results.items()},
        }

    def backtest(self, state: str) -> dict:
        """Return saved walk-forward CV predictions vs realized truth + per-model fold metrics."""
        if self._loaded_version is None:
            raise RuntimeError("No trained version is loaded.")
        state_dir = self.registry.state_dir(state, version=self._loaded_version)
        path = state_dir / "cv_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"No cv_predictions.csv for state {state}; rerun training.")
        df = pd.read_csv(path, index_col="date", parse_dates=["date"])
        meta_state = self._metadata.get("states", {}).get(state, {})
        # Per-fold metrics already aggregated; expose alongside raw points.
        return {
            "state": state,
            "registry_version": self._loaded_version,
            "selected_models": meta_state.get("selected_models", []),
            "ensemble_weights": meta_state.get("ensemble_weights", {}),
            "aggregate_metrics": meta_state.get("aggregate_metrics", {}),
            "rows": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "y_true": _safe_float(row["y_true"]),
                    **{col: _safe_float(row[col]) for col in df.columns if col != "y_true"},
                }
                for idx, row in df.iterrows()
            ],
        }

    def holiday_impact(self, state: str) -> dict:
        """Quantify holiday-week lift using historical (raw) data per state."""
        from ..data import DataLoader, Preprocessor

        df = DataLoader(self.cfg).load()
        per_state = Preprocessor(self.cfg).transform(df)
        if state not in per_state:
            raise KeyError(f"Unknown state: {state}")
        weekly = per_state[state]
        target_col = self.cfg.data.target_col
        y = weekly[target_col].astype(float)

        import holidays as hol_lib

        cal = hol_lib.country_holidays(self.cfg.features.holidays.country)
        holiday_flags = []
        next_holiday_name = []
        for d in y.index:
            week_start = d - pd.Timedelta(days=6)
            in_week_dates = [day.date() for day in pd.date_range(week_start, d) if day.date() in cal]
            if in_week_dates:
                holiday_flags.append(1)
                # First holiday name in week
                next_holiday_name.append(cal.get(in_week_dates[0]))
            else:
                holiday_flags.append(0)
                next_holiday_name.append(None)
        flags = np.array(holiday_flags, dtype=bool)
        non_h_mean = float(y[~flags].mean()) if (~flags).any() else 0.0
        h_mean = float(y[flags].mean()) if flags.any() else 0.0
        lift_pct = ((h_mean - non_h_mean) / non_h_mean * 100.0) if non_h_mean > 0 else 0.0
        # Per-named-holiday averages
        per_holiday: dict[str, dict[str, float]] = {}
        for i, name in enumerate(next_holiday_name):
            if name is None:
                continue
            bucket = per_holiday.setdefault(name, {"n": 0, "sum": 0.0})
            bucket["n"] += 1
            bucket["sum"] += float(y.iloc[i])
        per_holiday_clean = {
            k: {
                "n_weeks": int(v["n"]),
                "avg_weekly_sales": float(v["sum"] / v["n"]),
                "lift_vs_non_holiday_pct": (
                    (v["sum"] / v["n"] - non_h_mean) / non_h_mean * 100.0 if non_h_mean > 0 else 0.0
                ),
            }
            for k, v in per_holiday.items()
        }
        return {
            "state": state,
            "non_holiday_avg": non_h_mean,
            "holiday_avg": h_mean,
            "holiday_lift_pct": lift_pct,
            "per_holiday": per_holiday_clean,
        }

    def _forecast_all_members(
        self,
        bundle: StateBundle,
        horizon: int,
        ci_alpha: float,
        target_transform: str | None,
        use_conformal: bool,
    ) -> dict[str, ForecastResult]:
        results: dict[str, ForecastResult] = {}
        for name, model in bundle.models.items():
            try:
                r = self._forecast_single(name, model, horizon, ci_alpha, target_transform)
                # Override CIs with conformal half-widths when calibrators are present.
                if use_conformal and name in bundle.conformal:
                    r = bundle.conformal[name].apply(r.mean)
                results[name] = r
            except Exception as e:  # noqa: BLE001
                log.warning("Forecast failed for state=%s model=%s: %s", bundle.state, name, e)
        return results

    def _forecast_single(
        self,
        name: str,
        model: Any,
        horizon: int,
        ci_alpha: float,
        target_transform: str | None,
    ) -> ForecastResult:
        result = model.forecast(horizon=horizon, ci_alpha=ci_alpha)
        if name in {"arima", "sarima", "prophet"}:
            return _sanitize(result)
        # ML/DL models work in log space; clip log values to avoid overflow then expm1.
        if target_transform == "log1p":

            def _back(s: pd.Series | None) -> pd.Series | None:
                if s is None:
                    return None
                clipped = np.clip(s.values, -50.0, 50.0)
                return pd.Series(np.expm1(clipped).clip(min=0), index=s.index)

            return _sanitize(
                ForecastResult(
                    mean=_back(result.mean),
                    lower=_back(result.lower),
                    upper=_back(result.upper),
                    metadata=result.metadata,
                )
            )
        return _sanitize(result)
