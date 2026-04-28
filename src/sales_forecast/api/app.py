"""FastAPI application factory."""

from __future__ import annotations

import time
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from .. import __version__
from ..config import Config, load_config
from ..utils.logging import get_logger, setup_logging
from .schemas import (
    BacktestResponse,
    BreakdownResponse,
    HealthResponse,
    HolidayImpactResponse,
    MetricsResponse,
    PredictResponse,
    RankingsResponse,
    StateMetrics,
    StateRankings,
    TrainRequest,
    TrainResponse,
    TrainStatus,
)
from .service import ForecastService

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    _PROM = True
    REQUEST_COUNT = Counter("sf_requests_total", "Total HTTP requests", ["method", "path", "status"])
    REQUEST_LATENCY = Histogram("sf_request_latency_seconds", "Request latency seconds", ["method", "path"])
    PREDICT_COUNT = Counter("sf_predict_total", "Forecast requests", ["state", "ci_method"])
    PREDICT_ERRORS = Counter("sf_predict_errors_total", "Forecast errors", ["state"])
except Exception:  # noqa: BLE001
    _PROM = False

log = get_logger(__name__)


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    setup_logging(log_dir=cfg.path(cfg.project.log_dir), level=cfg.project.log_level, filename="api.log")
    app = FastAPI(
        title="Sales Forecasting API",
        version=__version__,
        description=(
            "Per-state weekly sales forecasting. Train models, retrieve forecasts with confidence "
            "intervals, inspect metrics, and check service health."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    service = ForecastService(cfg)
    service.reload()
    app.state.service = service
    app.state.cfg = cfg

    if _PROM and cfg.api.enable_prometheus:

        @app.middleware("http")
        async def _prom_mw(request: Request, call_next):  # type: ignore[no-untyped-def]
            t0 = time.perf_counter()
            response = await call_next(request)
            elapsed = time.perf_counter() - t0
            try:
                REQUEST_LATENCY.labels(request.method, request.url.path).observe(elapsed)
                REQUEST_COUNT.labels(request.method, request.url.path, str(response.status_code)).inc()
            except Exception:  # noqa: BLE001
                pass
            return response

        @app.get("/metrics/prom", include_in_schema=False)
        def prom_metrics():
            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    def health():
        s: ForecastService = app.state.service
        return HealthResponse(
            status="ok",
            version=__version__,
            registry_version=s.loaded_version,
            states_available=len(s.states_available()),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    @app.post("/train", response_model=TrainResponse, status_code=202, tags=["models"])
    def train(req: TrainRequest, background_tasks: BackgroundTasks):  # noqa: ARG001
        s: ForecastService = app.state.service
        job = s.submit_training(states=req.states)
        return TrainResponse(
            job_id=job.job_id,
            status=job.status,
            message=(
                "Training accepted. Poll GET /train/{job_id} for status. "
                "Predictions auto-refresh once training succeeds."
            ),
        )

    @app.get("/train/{job_id}", response_model=TrainStatus, tags=["models"])
    def train_status(job_id: str):
        s: ForecastService = app.state.service
        job = s.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id {job_id!r}")
        return TrainStatus(**job.__dict__)

    @app.get("/predict", response_model=PredictResponse, tags=["forecast"])
    def predict(
        state: str = Query(..., description="State name, e.g. 'California'."),
        horizon: int = Query(
            default=cfg.forecast.horizon_weeks,
            ge=1,
            le=52,
            description="Forecast horizon in weeks. Defaults to 8.",
        ),
        ci_alpha: float = Query(
            default=cfg.forecast.ci_alpha,
            gt=0.0,
            lt=1.0,
            description="Two-sided alpha for confidence intervals (e.g. 0.1 -> 90% CI).",
        ),
        conformal: bool = Query(
            default=cfg.conformal.enabled,
            description="If true and calibrators exist, use split-conformal CIs.",
        ),
    ):
        s: ForecastService = app.state.service
        if not s.is_ready():
            raise HTTPException(
                status_code=503,
                detail="No trained model registry is loaded. Call POST /train first.",
            )
        try:
            result = s.predict(state=state, horizon=horizon, ci_alpha=ci_alpha, use_conformal=conformal)
        except KeyError as e:
            if _PROM:
                PREDICT_ERRORS.labels(state).inc()
            raise HTTPException(status_code=404, detail=str(e)) from e
        except FileNotFoundError as e:
            if _PROM:
                PREDICT_ERRORS.labels(state).inc()
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            if _PROM:
                PREDICT_ERRORS.labels(state).inc()
            log.exception("Prediction failed: %s", e)
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
        if _PROM:
            PREDICT_COUNT.labels(state, result.get("ci_method", "model_native")).inc()
        return result

    @app.get("/predict/breakdown", response_model=BreakdownResponse, tags=["forecast"])
    def predict_breakdown(
        state: str = Query(..., description="State name."),
        horizon: int = Query(default=cfg.forecast.horizon_weeks, ge=1, le=52),
        ci_alpha: float = Query(default=cfg.forecast.ci_alpha, gt=0.0, lt=1.0),
        conformal: bool = Query(default=cfg.conformal.enabled),
    ):
        s: ForecastService = app.state.service
        if not s.is_ready():
            raise HTTPException(
                status_code=503,
                detail="No trained model registry is loaded. Call POST /train first.",
            )
        try:
            return s.predict_breakdown(
                state=state, horizon=horizon, ci_alpha=ci_alpha, use_conformal=conformal
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except FileNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            log.exception("Breakdown failed: %s", e)
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    @app.get("/backtest", response_model=BacktestResponse, tags=["forecast"])
    def backtest(state: str = Query(..., description="State name.")):
        s: ForecastService = app.state.service
        if not s.is_ready():
            raise HTTPException(
                status_code=503,
                detail="No trained model registry is loaded. Call POST /train first.",
            )
        try:
            return s.backtest(state=state)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    @app.get("/holiday_impact", response_model=HolidayImpactResponse, tags=["analytics"])
    def holiday_impact(state: str = Query(..., description="State name.")):
        s: ForecastService = app.state.service
        try:
            return s.holiday_impact(state=state)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            log.exception("Holiday-impact failed: %s", e)
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e

    @app.get("/states", tags=["system"])
    def list_states():
        s: ForecastService = app.state.service
        return {"registry_version": s.loaded_version, "states": s.states_available()}

    @app.post("/report", tags=["analytics"])
    def report(
        state: str = Query(..., description="State name."),
        horizon: int = Query(default=cfg.forecast.horizon_weeks, ge=1, le=52),
    ):
        s: ForecastService = app.state.service
        if not s.is_ready():
            raise HTTPException(status_code=503, detail="Train models first.")
        try:
            from .report import build_pdf_report

            pdf_bytes = build_pdf_report(s, state=state, horizon=horizon)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001
            log.exception("Report failed: %s", e)
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="report_{state}.pdf"'},
        )

    @app.get("/rankings", response_model=RankingsResponse, tags=["models"])
    def rankings():
        s: ForecastService = app.state.service
        if not s.is_ready():
            raise HTTPException(
                status_code=503,
                detail="No trained model registry is loaded. Call POST /train first.",
            )
        meta = s.metadata
        winner_counts: dict[str, int] = {}
        states_data: list[StateRankings] = []
        for state, info in meta.get("states", {}).items():
            ranks = info.get("rankings") or []
            if not ranks:
                continue
            winner = ranks[0]["model"]
            winner_counts[winner] = winner_counts.get(winner, 0) + 1
            states_data.append(
                StateRankings(
                    state=state,
                    selected_models=info.get("selected_models", []),
                    ensemble_weights=info.get("ensemble_weights", {}),
                    rankings=ranks,
                )
            )
        return RankingsResponse(
            registry_version=s.loaded_version,
            overall_winner_counts=dict(sorted(winner_counts.items(), key=lambda kv: -kv[1])),
            states=states_data,
        )

    @app.get("/metrics", response_model=MetricsResponse, tags=["models"])
    def metrics():
        s: ForecastService = app.state.service
        if not s.is_ready():
            raise HTTPException(
                status_code=503,
                detail="No trained model registry is loaded. Call POST /train first.",
            )
        meta = s.metadata
        states_data = []
        for state, info in meta.get("states", {}).items():
            if not info.get("selected_models"):
                continue
            states_data.append(
                StateMetrics(
                    state=state,
                    selected_models=info.get("selected_models", []),
                    ensemble_weights=info.get("ensemble_weights", {}),
                    aggregate_metrics=info.get("aggregate_metrics", {}),
                    history_weeks=int(info.get("history_weeks", 0)),
                )
            )
        return MetricsResponse(registry_version=s.loaded_version, states=states_data)

    return app


# Module-level app for `uvicorn sales_forecast.api.app:app` deployments.
app = create_app()
