# Sales Forecasting System

Production-grade time series forecasting pipeline that produces an **8-week per-state weekly sales forecast** with confidence intervals. The system trains and benchmarks five model families (ARIMA, SARIMA, Prophet, XGBoost+Optuna, PyTorch LSTM), selects the top performers per state via walk-forward cross-validation, ensembles them with dynamic weights, explains the XGBoost driver via SHAP, monitors drift, and serves everything through a FastAPI service plus an interactive Streamlit dashboard.

## Standout features

- **Walk-forward CV with leakage-safe pipeline** — expanding-window training, fixed 8-week test window, ≥3 folds per state.
- **Five model families benchmarked per state** — ARIMA, SARIMA, Prophet, XGBoost (Optuna-tuned, native tree-SHAP), PyTorch LSTM (multi-step direct).
- **Top-K dynamic ensemble** (inverse-RMSE, softmax, equal, **stacking**) per state.
- **Ridge stacking meta-learner** with simplex-projected weights (non-negative, sum-to-one) trained on out-of-fold predictions.
- **Split-conformal prediction intervals** — finite-sample-valid CIs computed per horizon step from CV residuals; toggled per-request.
- **SHAP explainability** via XGBoost's native `pred_contribs=True` (compatible with XGBoost 3.x JSON dumps; no SHAP-library breakage).
- **Drift detection** — PSI + KS-test on rolling vs reference windows.
- **Versioned model registry** — every run lives under `artifacts/registry/v<TS>/states/<State>/` with `feature_engineer.joblib`, per-model `.joblib`, `conformal.json`, `cv_predictions.csv`, `shap/`.
- **Rich REST API** — `/train`, `/predict`, `/predict/breakdown`, `/backtest`, `/holiday_impact`, `/metrics`, `/states`, `/health`, **`/report` (PDF)**, **`/metrics/prom` (Prometheus)**.
- **Interactive Streamlit dashboard** — forecasts, member breakdown, walk-forward backtest, holiday-impact analytics, model metrics.
- **One-command Docker stack** — `docker compose up` boots API + dashboard.
- **Holiday-impact endpoint** — quantifies historical lift around US holidays.
- **Backtest endpoint** — replays saved walk-forward CV predictions vs realized truth.
- **PDF report endpoint** — multi-page report (forecast, CV table, drift) for any state.

---

## Folder Structure

```
sales-forecasting-system/
├── config.yaml                     # Single source of truth for every parameter
├── pyproject.toml                  # Dependencies + tooling configuration
├── README.md
├── data/
│   └── raw/sales.xlsx              # Source dataset (43 states, weekly, Beverages)
├── artifacts/                      # Versioned model registry + SHAP plots (gitignored)
├── logs/                           # Rotating log files (gitignored)
├── src/sales_forecast/
│   ├── config/                     # Pydantic-typed config loader
│   ├── data/                       # Loader, weekly reindex + hybrid imputation, walk-forward splits
│   ├── features/                   # FeatureEngineer (lags, rolling, trend, Fourier, holidays)
│   ├── models/                     # ARIMA / SARIMA / Prophet / XGBoost+Optuna / LSTM + ensemble
│   ├── training/                   # End-to-end pipeline + CLI
│   ├── evaluation/                 # Metrics (RMSE/MAE/MAPE/SMAPE) + SHAP explainability
│   ├── api/                        # FastAPI app, schemas, service, CLI launcher
│   └── utils/                      # Logging, IO, seed, model registry, drift detection
├── tests/                          # Pytest suite (data, features, models, metrics, drift, API)
└── .github/workflows/ci.yml        # Lint + test CI pipeline
```

## Architecture

```
                 ┌────────────────┐
   sales.xlsx ─▶ │  DataLoader    │  schema validation, mixed-format date parsing
                 └──────┬─────────┘
                        ▼
                 ┌────────────────┐
                 │ Preprocessor   │  reindex W-SUN → hybrid imputation
                 │                │  → IQR + Z-score outlier handling
                 └──────┬─────────┘
                        ▼
                 ┌─────────────────────┐
                 │ Walk-Forward CV     │  expanding train, fixed 8-week val
                 └──────┬──────────────┘
                        ▼
   ┌──────────────────────────────────────────────────┐
   │ For each state x model in {ARIMA, SARIMA,        │
   │ Prophet, XGBoost+Optuna, LSTM (PyTorch)}:        │
   │   ─ FeatureEngineer (lags/rolling/Fourier/...)    │
   │   ─ Fit fold k → score RMSE/MAE/MAPE              │
   └──────┬───────────────────────────────────────────┘
          ▼
   ┌──────────────────┐    ┌───────────────────┐
   │  Rank per state  │ ──▶│  Top-2 weighted   │
   │  by mean RMSE    │    │  ensemble (1/RMSE)│
   └──────┬───────────┘    └─────────┬─────────┘
          │                          ▼
          ▼                  ┌──────────────────┐
   ┌──────────────────┐      │ Refit on full    │
   │ SHAP for XGBoost │      │ history; persist │
   └──────────────────┘      │ to versioned     │
                             │ registry         │
                             └─────────┬────────┘
                                       ▼
   ┌──────────────────────────────────────────────────┐
   │ FastAPI: /train (async) /predict /metrics /health │
   │   ◀──── lazy-loads bundles per state from registry│
   └──────────────────────────────────────────────────┘
```

Key design choices:

- **Single `config.yaml`** drives every component (Pydantic-typed; auto-discovered from project root).
- **Per-state isolation** keeps memory bounded and lets us run states in parallel later if needed.
- **Walk-forward CV** with expanding-window training prevents target leakage.
- **Dynamic top-2 ensemble** weighted by inverse RMSE (configurable to softmax / equal).
- **Versioned registry** (`artifacts/registry/vYYYYMMDD_HHMMSS/`) with a `manifest.json` pointing at the current promoted version. Old versions are kept; old runs can be GC’d via `ModelRegistry.gc()`.
- **Residual-bootstrap CIs** for ML/DL models; native CIs for ARIMA/SARIMA/Prophet.
- **Drift detection** at training time using PSI + KS-test against the most recent N-week window.
- **Background training jobs** so `POST /train` returns immediately with a `job_id` you can poll.

## Setup

```bash
git clone https://github.com/<you>/sales-forecasting-system.git
cd sales-forecasting-system
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> **Note**: First-time Prophet/cmdstanpy install can take a few minutes while it builds Stan models. PyTorch wheels are >700MB; expect 1–3 minutes on a fresh machine.

Place the dataset at `data/raw/sales.xlsx` (already included). Adjust `config.yaml > data` if your schema differs.

## Run

### Train every state

```bash
sales-forecast-train
```

or

```bash
python -m sales_forecast.training.cli --states California Texas
```

This writes:

- `artifacts/registry/v<TS>/states/<State>/{arima,sarima,prophet,xgboost,lstm}.joblib`
- `artifacts/registry/v<TS>/states/<State>/feature_engineer.joblib`
- `artifacts/registry/v<TS>/states/<State>/shap/shap_summary_<State>.png`
- `artifacts/registry/v<TS>/states/<State>/shap/feature_importance_<State>.png`
- `artifacts/registry/v<TS>/metadata.json` (selected models + ensemble weights + aggregate metrics)
- `artifacts/registry/v<TS>/report.json` (full per-state CV detail)
- `artifacts/registry/manifest.json` (`{"current": "v<TS>"}`)

### Serve the API

```bash
sales-forecast-api --port 8000
# or
uvicorn sales_forecast.api.app:app --port 8000 --reload
```

OpenAPI docs are at `http://localhost:8000/docs`.

## API

| Method | Path                          | Description                                                                 |
| ------ | ----------------------------- | --------------------------------------------------------------------------- |
| GET    | `/health`                     | System status + currently loaded registry version                           |
| GET    | `/states`                     | List of states with trained models                                          |
| POST   | `/train`                      | Triggers a training run (async background job; returns `job_id`)            |
| GET    | `/train/{job_id}`             | Poll training job status                                                    |
| GET    | `/predict?state=XYZ`          | 8-week forecast with confidence intervals (`?conformal=true` toggles split-conformal CIs) |
| GET    | `/predict/breakdown?state=XYZ`| Ensemble forecast plus per-member forecasts and weights                     |
| GET    | `/backtest?state=XYZ`         | Walk-forward CV: realized truth vs each model's predictions per fold        |
| GET    | `/holiday_impact?state=XYZ`   | Historical holiday-week sales lift, per US holiday                          |
| GET    | `/metrics`                    | Per-state CV metrics + selected models + ensemble weights                   |
| GET    | `/metrics/prom`               | Prometheus exposition format (request counts, latencies, prediction counts) |
| POST   | `/report?state=XYZ&horizon=8` | Multi-page PDF report (forecast chart + table, CV metrics, drift)           |

### Sample requests/responses

**Health**

```bash
$ curl -s localhost:8000/health
{
  "status": "ok",
  "version": "0.1.0",
  "registry_version": "v20260428_113022",
  "states_available": 43,
  "timestamp": "2026-04-28T11:32:55.214930+00:00"
}
```

**Train**

```bash
$ curl -s -X POST localhost:8000/train -H 'Content-Type: application/json' \
       -d '{"states": ["California", "Texas"]}'
{
  "job_id": "8d8b3df1-2a36-4cd2-bff8-7a0f2e1a93f6",
  "status": "pending",
  "message": "Training accepted. Poll GET /train/{job_id} for status. Predictions auto-refresh once training succeeds."
}

$ curl -s localhost:8000/train/8d8b3df1-2a36-4cd2-bff8-7a0f2e1a93f6
{
  "job_id": "8d8b3df1-2a36-4cd2-bff8-7a0f2e1a93f6",
  "status": "running",
  "started_at": 1745847132.91,
  "finished_at": null,
  "version": null,
  "error": null,
  "states_trained": 0
}
```

**Predict (with split-conformal CIs)**

```bash
$ curl -s "localhost:8000/predict?state=California&horizon=8&conformal=true"
{
  "state": "California",
  "registry_version": "v20260428_113022",
  "horizon_weeks": 8,
  "selected_models": ["xgboost", "prophet"],
  "ensemble_weights": {"xgboost": 0.612, "prophet": 0.388},
  "ci_method": "conformal",
  "forecast": [
    {"date": "2023-12-10", "yhat": 471920134.2, "yhat_lower": 442104871.0, "yhat_upper": 501735397.4},
    ...
  ],
  "drift": {"psi": 0.18, "ks_pvalue": 0.07, "drifted": false}
}
```

**Predict breakdown**

```bash
$ curl -s "localhost:8000/predict/breakdown?state=California" | jq '.members | keys'
["arima", "prophet", "xgboost"]
```

**Backtest**

```bash
$ curl -s "localhost:8000/backtest?state=California" | jq '.rows[0]'
{
  "date": "2021-01-10",
  "y_true": 460381021.4,
  "arima":   452113442.1,
  "sarima":  461773111.0,
  "prophet": 458201001.5,
  "xgboost": 463812114.7
}
```

**Holiday impact**

```bash
$ curl -s "localhost:8000/holiday_impact?state=California" | jq '{lift: .holiday_lift_pct, top: .per_holiday | to_entries | sort_by(-.value.lift_vs_non_holiday_pct) | .[0:3]}'
```

**PDF report**

```bash
$ curl -s -X POST "localhost:8000/report?state=California&horizon=8" -o california_report.pdf
```

**Prometheus**

```bash
$ curl -s localhost:8000/metrics/prom | head
# HELP sf_requests_total Total HTTP requests
# TYPE sf_requests_total counter
sf_requests_total{method="GET",path="/health",status="200"} 17.0
...
```

**Metrics**

```bash
$ curl -s localhost:8000/metrics | jq '.states[0]'
{
  "state": "Alabama",
  "selected_models": ["xgboost", "prophet"],
  "ensemble_weights": {"xgboost": 0.6, "prophet": 0.4},
  "aggregate_metrics": {
    "arima":   {"rmse": 24910342.1, "mae": 19912331.0, "mape": 11.7, "smape": 11.5},
    "sarima":  {"rmse": 22408810.2, "mae": 18055412.0, "mape": 10.9, "smape": 10.8},
    "prophet": {"rmse": 20910340.9, "mae": 16710320.5, "mape":  9.8, "smape":  9.7},
    "xgboost": {"rmse": 19712100.8, "mae": 15823100.0, "mape":  9.1, "smape":  9.1},
    "lstm":    {"rmse": 24501230.3, "mae": 20121400.7, "mape": 11.6, "smape": 11.5}
  },
  "history_weeks": 256
}
```

## Configuration

All knobs live in `config.yaml`. Highlights:

- `data.freq` — resample target frequency (`W-SUN` is default; matches dominant cadence in source data).
- `preprocessing.imputation.*` — interpolation+ffill+bfill window controls.
- `preprocessing.outliers.{iqr_multiplier, zscore_threshold, strategy}` — `cap` clips, `mark` flags only.
- `features.{lags, rolling_windows, fourier, holidays, trend, target_transform}` — feature shaping.
- `cv.{initial_train_weeks, horizon, step, max_folds}` — walk-forward CV definition.
- `models.<name>.*` — per-model hyperparameters; `models.enabled` selects which families participate.
- `ensemble.{top_k, weighting}` — `inverse_rmse | softmax | equal | stacking` (ridge meta-learner with simplex-projected weights, fit on out-of-fold predictions).
- `conformal.{enabled, alpha}` — split-conformal prediction intervals computed from CV residuals at `1 - alpha` quantile.
- `forecast.{horizon_weeks, ci_alpha}` — default forecast settings used by `/predict`.
- `drift.{feature_window_weeks, psi_threshold, ks_pvalue_threshold}` — drift detection sensitivity.
- `api.enable_prometheus` — turn the `/metrics/prom` endpoint and request-level histograms on/off.
- `dashboard.{port, page_title}` — Streamlit dashboard parameters.

## Streamlit dashboard

```bash
# in one shell:
sales-forecast-api --port 8000
# in another:
API_URL=http://localhost:8000 streamlit run dashboard/streamlit_app.py
```

Tabs:

- **Forecast** — interactive forecast chart with conformal toggle.
- **Member breakdown** — line plot per ensemble member + weights bar.
- **Walk-forward backtest** — realized vs predicted across all CV folds.
- **Holiday impact** — historical lift per US holiday.
- **Model metrics** — boxplot + per-state RMSE.

## Docker

```bash
docker compose up --build
# API:  http://localhost:8000
# Dash: http://localhost:8501
```

## Testing & Linting

```bash
pip install -e ".[dev]"
pytest -q                 # full suite (the API end-to-end test runs a smoke training run)
ruff check . && ruff format --check .
```

CI runs lint + the fast subset (data, features, metrics) on every PR — see `.github/workflows/ci.yml`.

## Production Hardening Notes

- **Versioning**: `ModelRegistry` (in `utils/versioning.py`) writes timestamped versions and a manifest. Promote/rollback by editing `manifest.json`'s `current` key.
- **Artifacts**: Every model is persisted via joblib alongside its `FeatureEngineer`, so prediction is deterministic and reproducible.
- **Drift**: `utils/drift.py` ships PSI + Kolmogorov–Smirnov detectors. The training pipeline runs them per state at fit time; the same module can be invoked on incoming production batches.
- **Logging**: Rotating file handler in `utils/logging.py` (10 MB per file, 5 backups) plus stdout. Default config keeps cmdstanpy/Prophet/Optuna logs at WARNING.
- **Reproducibility**: `utils/seed.set_seed(...)` is called at pipeline start for `random`, `numpy`, and `torch`.
- **Concurrency**: API uses a process-local mutex around the in-memory bundle cache and reloads the registry after each training job completes.
