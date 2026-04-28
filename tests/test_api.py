"""API tests covering /health and /predict (per submission requirements)."""

from __future__ import annotations

import shutil
import time

from fastapi.testclient import TestClient

from sales_forecast.api.app import create_app
from sales_forecast.training import TrainingPipeline


def _trim_states_for_speed(cfg, n: int = 2) -> list[str]:
    """Return the first `n` state names without doing heavy preprocessing twice."""
    from sales_forecast.data import DataLoader, Preprocessor

    df = DataLoader(cfg).load()
    out = Preprocessor(cfg).transform(df)
    return sorted(out.keys())[:n]


def test_health_endpoint_with_no_models(cfg):
    """Without a registry, /health still returns ok; /predict returns 503."""
    # Wipe registry to simulate cold start.
    reg_dir = cfg.path(cfg.project.registry_dir)
    if reg_dir.exists():
        shutil.rmtree(reg_dir)

    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["registry_version"] is None

        r = client.get("/predict", params={"state": "California"})
        assert r.status_code == 503


def test_predict_endpoint_after_smoke_training(cfg):
    """End-to-end: train on 2 states with reduced settings, hit /predict, verify response shape."""
    # Speed up Optuna and LSTM for the test.
    cfg.models.xgboost.optuna_trials = 3
    cfg.models.xgboost.optuna_timeout_seconds = 60
    cfg.models.lstm.epochs = 8
    cfg.models.lstm.patience = 3
    cfg.cv.max_folds = 2
    cfg.cv.initial_train_weeks = 90
    cfg.cv.step = 8
    cfg.models.enabled = ["arima", "prophet", "xgboost"]  # skip slow SARIMA(s=52) and LSTM here

    states = _trim_states_for_speed(cfg, n=1)
    pipe = TrainingPipeline(cfg)
    pipe.run(states=states)

    app = create_app(cfg)
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["registry_version"] is not None
        assert body["states_available"] >= 1

        r = client.get("/predict", params={"state": states[0], "horizon": 8})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["state"] == states[0]
        assert body["horizon_weeks"] == 8
        assert len(body["forecast"]) == 8
        for pt in body["forecast"]:
            assert "date" in pt and "yhat" in pt
            assert pt["yhat_lower"] <= pt["yhat"] <= pt["yhat_upper"]

        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.json()["registry_version"] is not None

        # Background training kick-off.
        r = client.post("/train", json={"states": states})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        # Don't wait for completion; just verify the status endpoint works.
        r = client.get(f"/train/{job_id}")
        assert r.status_code == 200
        # Allow the background thread a moment so it doesn't outlive the test.
        time.sleep(0.1)
