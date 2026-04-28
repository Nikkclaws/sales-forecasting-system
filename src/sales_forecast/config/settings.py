"""Typed configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectCfg(BaseModel):
    name: str = "sales-forecast"
    random_seed: int = 42
    log_level: str = "INFO"
    log_dir: str = "logs"
    artifacts_dir: str = "artifacts"
    registry_dir: str = "artifacts/registry"


class DataCfg(BaseModel):
    source_path: str
    date_col: str
    state_col: str
    target_col: str
    category_col: str | None = None
    freq: str = "W-SUN"
    min_history_weeks: int = 52


class ImputationCfg(BaseModel):
    interpolation_method: str = "linear"
    interpolation_limit: int = 4
    forward_fill_limit: int = 8


class OutlierCfg(BaseModel):
    iqr_multiplier: float = 3.0
    zscore_threshold: float = 3.0
    strategy: str = "cap"


class PreprocessingCfg(BaseModel):
    imputation: ImputationCfg = ImputationCfg()
    outliers: OutlierCfg = OutlierCfg()


class FourierComponentCfg(BaseModel):
    period: float
    order: int


class FourierCfg(BaseModel):
    yearly: FourierComponentCfg | None = None
    quarterly: FourierComponentCfg | None = None


class HolidayCfg(BaseModel):
    country: str = "US"
    add_distance_to_next_holiday: bool = True


class TrendCfg(BaseModel):
    include_linear: bool = True
    include_changepoints: bool = True
    n_changepoints: int = 8


class FeatureCfg(BaseModel):
    lags: list[int] = Field(default_factory=lambda: [1, 7, 14, 30])
    rolling_windows: list[int] = Field(default_factory=lambda: [4, 8, 13, 26])
    rolling_stats: list[str] = Field(default_factory=lambda: ["mean", "std"])
    fourier: FourierCfg = FourierCfg()
    holidays: HolidayCfg = HolidayCfg()
    trend: TrendCfg = TrendCfg()
    target_transform: str | None = "log1p"


class CVCfg(BaseModel):
    strategy: str = "walk_forward"
    initial_train_weeks: int = 104
    horizon: int = 8
    step: int = 8
    max_folds: int = 6
    min_validation_folds: int = 3


class ARIMACfg(BaseModel):
    order: list[int] = Field(default_factory=lambda: [2, 1, 2])
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False


class SARIMACfg(BaseModel):
    order: list[int] = Field(default_factory=lambda: [1, 1, 1])
    seasonal_order: list[int] = Field(default_factory=lambda: [1, 1, 1, 52])
    enforce_stationarity: bool = False
    enforce_invertibility: bool = False


class ProphetCfg(BaseModel):
    yearly_seasonality: bool = True
    weekly_seasonality: bool = False
    daily_seasonality: bool = False
    seasonality_mode: str = "multiplicative"
    interval_width: float = 0.9
    holidays_country: str = "US"


class XGBCfg(BaseModel):
    optuna_trials: int = 30
    optuna_timeout_seconds: int = 600
    early_stopping_rounds: int = 50
    base_params: dict[str, Any] = Field(default_factory=dict)


class LSTMCfg(BaseModel):
    sequence_length: int = 26
    horizon: int = 8
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 32
    epochs: int = 80
    learning_rate: float = 1e-3
    patience: int = 10


class ModelsCfg(BaseModel):
    enabled: list[str] = Field(default_factory=lambda: ["arima", "sarima", "prophet", "xgboost", "lstm"])
    arima: ARIMACfg = ARIMACfg()
    sarima: SARIMACfg = SARIMACfg()
    prophet: ProphetCfg = ProphetCfg()
    xgboost: XGBCfg = XGBCfg()
    lstm: LSTMCfg = LSTMCfg()


class StackingCfg(BaseModel):
    method: str = "ridge_simplex"


class EnsembleCfg(BaseModel):
    top_k: int = 2
    weighting: str = "inverse_rmse"  # inverse_rmse | softmax | equal | stacking
    stacking: StackingCfg = StackingCfg()


class ConformalCfg(BaseModel):
    enabled: bool = True
    alpha: float = 0.1
    fallback_global_alpha: float = 0.1


class GlobalLSTMCfg(BaseModel):
    enabled: bool = False
    embed_dim: int = 8
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 8


class ForecastCfg(BaseModel):
    horizon_weeks: int = 8
    ci_alpha: float = 0.1


class DriftCfg(BaseModel):
    feature_window_weeks: int = 52
    psi_threshold: float = 0.2
    ks_pvalue_threshold: float = 0.05


class APICfg(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    enable_prometheus: bool = True


class DashboardCfg(BaseModel):
    port: int = 8501
    page_title: str = "Sales Forecasting Dashboard"


class Config(BaseModel):
    project: ProjectCfg = ProjectCfg()
    data: DataCfg
    preprocessing: PreprocessingCfg = PreprocessingCfg()
    features: FeatureCfg = FeatureCfg()
    cv: CVCfg = CVCfg()
    models: ModelsCfg = ModelsCfg()
    ensemble: EnsembleCfg = EnsembleCfg()
    conformal: ConformalCfg = ConformalCfg()
    global_lstm: GlobalLSTMCfg = GlobalLSTMCfg()
    forecast: ForecastCfg = ForecastCfg()
    drift: DriftCfg = DriftCfg()
    api: APICfg = APICfg()
    dashboard: DashboardCfg = DashboardCfg()

    # Resolved paths (filled by load_config).
    project_root: Path = Field(default_factory=Path.cwd, exclude=True)

    def path(self, *parts: str) -> Path:
        return self.project_root.joinpath(*parts)


def find_project_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__).resolve()).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "config.yaml").exists() and (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_config(path: str | Path | None = None) -> Config:
    """Load config from path or auto-detected project root."""
    root = find_project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    with open(cfg_path) as fh:
        raw = yaml.safe_load(fh)
    cfg = Config(**raw)
    cfg.project_root = root
    return cfg
