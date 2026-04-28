"""XGBoost forecaster with Optuna hyperparameter tuning and recursive multi-step forecasting."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import BaseForecaster, ForecastResult

optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.WARNING)


class XGBoostForecaster(BaseForecaster):
    name = "xgboost"
    supports_exog = True

    def __init__(
        self,
        feature_columns: list[str] | None = None,
        target_col: str = "y",
        optuna_trials: int = 30,
        optuna_timeout_seconds: int = 600,
        early_stopping_rounds: int = 50,
        base_params: dict | None = None,
        random_state: int = 42,
    ):
        self.feature_columns = feature_columns
        self.target_col = target_col
        self.optuna_trials = optuna_trials
        self.optuna_timeout_seconds = optuna_timeout_seconds
        self.early_stopping_rounds = early_stopping_rounds
        self.base_params = base_params or {}
        self.random_state = random_state

        self.model = None
        self.best_params: dict = {}
        self.residual_std: float = 0.0
        self._engineer = None  # FeatureEngineer instance, set externally for forecasting
        self._target_history: pd.Series | None = None

    # --- Tuning + fit ----------------------------------------------------- #

    def _objective(self, trial, X: pd.DataFrame, y: pd.Series) -> float:
        import xgboost as xgb
        from sklearn.metrics import mean_squared_error

        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
            "n_estimators": 600,
            "random_state": self.random_state,
        }
        # Time-ordered hold-out (last 20%) for trial scoring.
        n = len(X)
        cut = int(n * 0.8)
        if cut < 30 or n - cut < 5:
            cut = max(20, n - 8)
        X_tr, X_va = X.iloc[:cut], X.iloc[cut:]
        y_tr, y_va = y.iloc[:cut], y.iloc[cut:]

        model = xgb.XGBRegressor(**params, early_stopping_rounds=self.early_stopping_rounds)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        preds = model.predict(X_va)
        return float(np.sqrt(mean_squared_error(y_va, preds)))

    def fit(
        self,
        history: pd.Series,
        exog: pd.DataFrame | None = None,
        engineer=None,
    ) -> XGBoostForecaster:
        import optuna
        import xgboost as xgb

        if exog is None:
            raise ValueError("XGBoostForecaster requires engineered features in `exog`.")
        if engineer is None:
            raise ValueError("XGBoostForecaster requires the FeatureEngineer for recursive forecasting.")

        self._engineer = engineer
        self._target_history = history.copy()

        feats = exog.copy()
        # Drop rows with NaN lags (typical at the start of the series).
        feats = feats.dropna()
        y = feats[self.target_col]
        cols = self.feature_columns or [c for c in feats.columns if c != self.target_col]
        self.feature_columns = cols
        X = feats[cols]

        if len(X) < 20:
            # Fallback: too little data to tune; use base params.
            self.best_params = {"max_depth": 4, "learning_rate": 0.05, "n_estimators": 400}
        else:
            sampler = optuna.samplers.TPESampler(seed=self.random_state)
            study = optuna.create_study(direction="minimize", sampler=sampler)
            study.optimize(
                lambda t: self._objective(t, X, y),
                n_trials=self.optuna_trials,
                timeout=self.optuna_timeout_seconds,
                show_progress_bar=False,
                catch=(Exception,),
            )
            self.best_params = study.best_params

        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "n_estimators": 800,
            "random_state": self.random_state,
            **self.best_params,
        }
        # Final fit on all data (no early stopping since we use the full set).
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X, y, verbose=False)

        # Residual std on the in-sample fit, used for residual-bootstrap CIs.
        in_sample_pred = self.model.predict(X)
        self.residual_std = float(np.std(y.values - in_sample_pred, ddof=1))
        return self

    # --- Forecast --------------------------------------------------------- #

    def forecast(
        self,
        horizon: int,
        exog_future: pd.DataFrame | None = None,
        ci_alpha: float = 0.1,
    ) -> ForecastResult:
        from scipy.stats import norm

        if self.model is None:
            raise RuntimeError("Call fit() first.")
        if self._engineer is None or self._target_history is None:
            raise RuntimeError("Engineer/history missing; the forecaster wasn't fit through the pipeline.")

        future_idx = self._engineer.make_future_frame(horizon)
        future_targets = pd.Series(index=future_idx, dtype=float)
        history = self._target_history.copy()
        preds: list[float] = []
        for ts in future_idx:
            # Combine history + already-predicted future for feature lookups.
            combined = pd.concat([history, future_targets[future_targets.notna()]]).sort_index()
            feats = self._engineer.transform_for_forecast(
                pd.Series(np.nan, index=[ts]), full_history=combined
            )
            X_step = feats[self.feature_columns]
            yhat = float(self.model.predict(X_step)[0])
            future_targets.loc[ts] = yhat
            preds.append(yhat)

        mean = pd.Series(preds, index=future_idx, dtype=float)
        # Symmetric residual-bootstrap CI; stretch with horizon to reflect compounding uncertainty.
        z = float(norm.ppf(1 - ci_alpha / 2))
        sigmas = self.residual_std * np.sqrt(np.arange(1, horizon + 1))
        lower = pd.Series(mean.values - z * sigmas, index=future_idx)
        upper = pd.Series(mean.values + z * sigmas, index=future_idx)
        return ForecastResult(
            mean=mean,
            lower=lower,
            upper=upper,
            metadata={"best_params": self.best_params, "residual_std": self.residual_std},
        )
