"""Pydantic schemas for API requests/responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "unknown"
    registry_version: str | None = None
    states_available: int = 0
    timestamp: str


class TrainRequest(BaseModel):
    states: list[str] | None = Field(
        default=None,
        description="Optional subset of states. If null, trains every state with sufficient history.",
    )


class TrainResponse(BaseModel):
    job_id: str
    status: str
    message: str


class TrainStatus(BaseModel):
    job_id: str
    status: str
    started_at: float | None = None
    finished_at: float | None = None
    version: str | None = None
    error: str | None = None
    states_trained: int = 0


class ForecastPoint(BaseModel):
    date: str
    yhat: float
    yhat_lower: float | None = None
    yhat_upper: float | None = None


class PredictResponse(BaseModel):
    state: str
    registry_version: str
    horizon_weeks: int
    selected_models: list[str]
    ensemble_weights: dict[str, float]
    forecast: list[ForecastPoint]
    drift: dict[str, Any] | None = None
    ci_method: str = "model_native"


class BreakdownResponse(BaseModel):
    state: str
    registry_version: str
    horizon_weeks: int
    ensemble_weights: dict[str, float]
    ci_method: str = "model_native"
    ensemble: list[ForecastPoint]
    members: dict[str, list[ForecastPoint]]


class BacktestRow(BaseModel):
    date: str
    y_true: float | None = None
    # Per-model OOF predictions live in extra fields. Allowed via extra="allow".
    model_config = {"extra": "allow"}


class BacktestResponse(BaseModel):
    state: str
    registry_version: str
    selected_models: list[str]
    ensemble_weights: dict[str, float]
    aggregate_metrics: dict[str, dict[str, float]]
    rows: list[BacktestRow]


class HolidayBucket(BaseModel):
    n_weeks: int
    avg_weekly_sales: float
    lift_vs_non_holiday_pct: float


class HolidayImpactResponse(BaseModel):
    state: str
    non_holiday_avg: float
    holiday_avg: float
    holiday_lift_pct: float
    per_holiday: dict[str, HolidayBucket]


class StateMetrics(BaseModel):
    state: str
    selected_models: list[str]
    ensemble_weights: dict[str, float]
    aggregate_metrics: dict[str, dict[str, float]]
    history_weeks: int


class MetricsResponse(BaseModel):
    registry_version: str
    states: list[StateMetrics]
