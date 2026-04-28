# Demo & Submission Guide

This doc has two things you need:

1. **A step-by-step "how to run" guide** so you can spin everything up locally and verify each feature.
2. **A 10-minute video script with timestamps** you can read straight off while recording.

---

## Part 1 — Step-by-Step: How to Run the Project

### Prerequisites

- Python 3.10, 3.11, or 3.12 (the project pins `>=3.10,<3.13`).
- Git.
- ~3 GB free disk for dependencies (PyTorch + Prophet are heavy).
- Optional: Docker Desktop, if you want the one-command stack.

### Step 1 — Clone and install

```bash
git clone https://github.com/Nikkclaws/sales-forecasting-system.git
cd sales-forecasting-system

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

> First install can take 3–8 minutes. PyTorch wheels are ~700 MB, Prophet builds Stan models the first time.

### Step 2 — Sanity-check the install

```bash
ruff check . && ruff format --check .
pytest -q
```

You should see **47 files OK** for ruff and **12 passed** for pytest.

### Step 3 — Train models

The dataset is already at `data/raw/sales.xlsx` (43 US states, weekly Beverages sales, Jan 2019 – Dec 2023).

```bash
# Train two states quickly (good for a demo).
python -m sales_forecast.training.cli --states California Texas
```

Expected output:
- 5 models per state (ARIMA, SARIMA, Prophet, XGBoost+Optuna, LSTM) trained with walk-forward CV.
- Top-2 models per state selected and ensembled.
- Artifacts written to `artifacts/registry/v<TIMESTAMP>/states/<State>/`.

Want all 43 states? Drop the `--states` flag (takes ~40–90 minutes).

### Step 4 — Inspect saved artifacts

```bash
ls artifacts/registry/$(cat artifacts/registry/manifest.json | python -c "import sys,json;print(json.load(sys.stdin)['current'])")/states/California/
```

You should see:
```
arima.joblib   feature_engineer.joblib   conformal.json
prophet.joblib xgboost.joblib            cv_predictions.csv
sarima.joblib  lstm.joblib               shap/
```

### Step 5 — Start the FastAPI service

```bash
uvicorn sales_forecast.api.app:app --port 8000 --reload
```

Open http://localhost:8000/docs for interactive Swagger UI.

### Step 6 — Hit every endpoint

```bash
# 1. Health
curl -s http://localhost:8000/health | jq

# 2. List trained states
curl -s http://localhost:8000/states | jq

# 3. Forecast (with split-conformal CIs)
curl -s "http://localhost:8000/predict?state=California&horizon=8&conformal=true" | jq

# 4. Per-member breakdown (each ensemble member's forecast)
curl -s "http://localhost:8000/predict/breakdown?state=California" | jq '.members | keys'

# 5. Walk-forward backtest (truth vs every model on each CV fold)
curl -s "http://localhost:8000/backtest?state=California" | jq '.rows[0]'

# 6. Holiday impact (sales lift around US holidays)
curl -s "http://localhost:8000/holiday_impact?state=California" | jq '{lift_pct: .holiday_lift_pct}'

# 7. Aggregate CV metrics for every state/model
curl -s http://localhost:8000/metrics | jq '.states[0]'

# 8. Prometheus exposition
curl -s http://localhost:8000/metrics/prom | head

# 9. PDF report (downloads california_report.pdf)
curl -s -X POST "http://localhost:8000/report?state=California" -o california_report.pdf
```

### Step 7 — Start the Streamlit dashboard

In a **second terminal**, with the API still running:

```bash
source .venv/bin/activate
API_URL=http://localhost:8000 streamlit run dashboard/streamlit_app.py
```

Open http://localhost:8501. Five tabs:
- **Forecast**: interactive line + CI band, toggle conformal CIs, switch state/horizon.
- **Member breakdown**: ensemble vs each model + weights bar chart.
- **Walk-forward backtest**: realized vs predicted across all CV folds.
- **Holiday impact**: bar chart of sales lift per US holiday.
- **Model metrics**: boxplot of RMSE per model + per-state RMSE bars.

### Step 8 — One-command Docker stack (optional)

```bash
docker compose up --build
```

API on http://localhost:8000, dashboard on http://localhost:8501. Both share artifacts via a volume mount.

---

## Part 2 — 10-Minute Video Script

Total: **10:00**. Read each section out loud while showing the corresponding screen.

### 0:00 – 0:30 · Hook & overview

> "Hi, I'm <name>. This is a production-grade time series forecasting system that predicts the next 8 weeks of sales for 43 US states. It runs five different models per state, picks the top two via walk-forward cross-validation, ensembles them, and serves everything through a REST API and an interactive dashboard. Let me show you what's inside."

**Show:** GitHub repo home page → README headline.

### 0:30 – 1:30 · Architecture overview

> "Here's the architecture. The code is fully modular under `src/sales_forecast/`. The `data/` package handles loading and weekly reindexing with hybrid imputation. The `features/` package generates lag features, rolling statistics, Fourier seasonality terms, US holiday distance, and trend changepoints. The `models/` package implements ARIMA, SARIMA, Prophet, XGBoost with Optuna tuning, and a PyTorch LSTM. The `training/` pipeline runs walk-forward cross-validation, ranks models per state, and persists artifacts to a versioned registry. The `evaluation/` package handles metrics and SHAP explainability. And `api/` is the FastAPI service."

**Show:** File tree (`tree src/sales_forecast -L 2` or VS Code sidebar). Highlight folders as you say each.

### 1:30 – 2:30 · Data pipeline

> "The dataset is 43 states of weekly Beverages sales from January 2019 through December 2023. Some weeks were missing. The data loader reindexes every state to a weekly Sunday-anchored grid and applies hybrid imputation: linear interpolation up to 4 weeks, forward-fill up to 8 weeks, and back-fill for any leading gaps. Outliers are detected with both IQR and Z-score and capped — not deleted. Look at this log line: 'imputed=91, outliers=1' — that's the report for Alabama."

**Show:** `python -m sales_forecast.training.cli --states California` for ~10 seconds, point at the preprocessor logs.

### 2:30 – 3:30 · Feature engineering

> "Now features. The `FeatureEngineer` class builds lag features at 1, 2, 3, 4, 7, 14, and 30 weeks. Rolling statistics — mean, standard deviation, min, max — at windows of 4, 8, 13, and 26. Fourier seasonality terms with yearly period 52.18 and order 6, plus quarterly. Trend features: a linear trend plus 8 uniform changepoints. And US holiday distance — days to next holiday plus a holiday flag. Everything is fit on training data only, then transformed for inference, so there's no leakage."

**Show:** `src/sales_forecast/features/engineer.py` open in VS Code, scroll through `_build_lags`, `_build_rolling`, `_build_fourier`, `_holidays_between`.

### 3:30 – 4:30 · Models + walk-forward CV

> "Five models compete on every state. ARIMA and SARIMA from statsmodels. Prophet with US holidays as regressors. XGBoost — and this one's tuned with Optuna across 30 trials, with native tree-SHAP explainability via `pred_contribs=True`, which is the only way to get exact SHAP values that work with XGBoost 3 onwards. And a PyTorch LSTM that does multi-step direct forecasting. They all run inside a walk-forward cross-validation loop: 104 weeks of training, 8-week validation window, sliding by 8. Minimum 3 folds, max 6."

**Show:** `src/sales_forecast/training/pipeline.py` — point at the `_run_state` method and the fold loop.

### 4:30 – 5:30 · Ensembling — top-2 + advanced features

> "After CV, models are ranked per state by mean RMSE. The top 2 are kept and combined into a weighted ensemble. The default scheme is inverse-RMSE, but I also implemented a ridge stacking meta-learner that fits non-negative weights summing to one, projected onto the probability simplex. And for the prediction intervals, I added split-conformal prediction: for each horizon step, we take the (1 minus alpha) quantile of absolute residuals from CV, and apply that half-width symmetrically around the forecast. That gives you finite-sample-valid 90% intervals — distribution-free, works for any base model."

**Show:** `src/sales_forecast/models/conformal.py` — focus on `from_residuals` and `apply`. Then `models/stacking.py` showing `_project_simplex`.

### 5:30 – 6:30 · Run the API

> "Now let me run the service."

```bash
uvicorn sales_forecast.api.app:app --port 8000
```

> "OpenAPI docs are auto-generated at slash docs."

**Show:** Open http://localhost:8000/docs in browser. Scroll through the endpoint list.

> "Let me hit /predict for California with conformal intervals enabled."

```bash
curl -s "http://localhost:8000/predict?state=California&conformal=true" | jq
```

> "There's the 8-week forecast with lower and upper bounds, the selected models, the ensemble weights, the CI method, and a drift report. Notice ci_method says conformal — that means we're using the calibrated half-widths, not the model's native intervals."

### 6:30 – 7:30 · Advanced API endpoints

> "Beyond predict, I built endpoints other people probably won't have. /predict/breakdown returns each ensemble member's forecast separately so you can defend why the ensemble looks the way it does."

```bash
curl -s "http://localhost:8000/predict/breakdown?state=California" | jq '.members | keys'
```

> "/backtest returns the saved walk-forward CV predictions vs realized truth — proves the model actually worked historically."

```bash
curl -s "http://localhost:8000/backtest?state=California" | jq '.rows[0]'
```

> "/holiday_impact quantifies how much sales lifted in holiday weeks compared to non-holiday weeks."

```bash
curl -s "http://localhost:8000/holiday_impact?state=California" | jq '{lift_pct: .holiday_lift_pct}'
```

> "/metrics/prom exposes Prometheus metrics — request counts, latencies, prediction counters."

> "And POST /report generates a multi-page PDF with the forecast chart, a metrics table, and the drift summary."

```bash
curl -s -X POST "http://localhost:8000/report?state=California" -o california_report.pdf
```

**Show:** Open the generated PDF in your viewer; flip through the pages.

### 7:30 – 9:00 · Streamlit dashboard

> "For non-technical reviewers I built an interactive dashboard."

```bash
streamlit run dashboard/streamlit_app.py
```

> "There are five tabs. The Forecast tab — pick any state, slide the horizon, toggle conformal intervals, see the prediction band update live. The Member breakdown tab shows each model's forecast individually plus the ensemble weights bar chart. The Backtest tab plays back walk-forward CV history. The Holiday impact tab visualizes lift per holiday — see Christmas and Thanksgiving stand out. And the Model metrics tab has a boxplot showing RMSE distribution across all states for each model family."

**Show:** Click through every tab. Switch from California to Texas to demonstrate state switching.

### 9:00 – 9:30 · Drift detection + versioning

> "Two more things. Every training run gets a timestamped version under `artifacts/registry/`. Manifests track the current production version, so you can roll back. And there's drift detection — every state computes PSI and a KS-test comparing the recent 52-week window to earlier history. If PSI exceeds 0.2 or the KS p-value drops below 0.05, the state is flagged."

**Show:** `ls artifacts/registry/` then `cat artifacts/registry/manifest.json`.

### 9:30 – 10:00 · Wrap-up

> "To recap — 5 model families, walk-forward CV, top-2 ensembling with ridge stacking, split-conformal intervals, native XGBoost SHAP, drift detection, ten REST endpoints, a Streamlit dashboard, and a one-command Docker stack. Twelve unit tests passing, lint-clean, and a CI workflow on GitHub. Code and docs are in the repo. Thanks for watching."

**Show:** GitHub repo + the green CI checkmark on PR #1.

---

## Tips for Recording

- Use **OBS Studio** (free) or **QuickTime** for screen capture. Set output to 1080p/30fps minimum.
- Record audio with a decent mic — webcam mics are fine if you're close. Avoid background noise.
- Have **two terminals open side by side** before you start: one for `uvicorn`, one for `streamlit`.
- Pre-warm the API once (curl `/predict` once) so the first response in the recording is fast.
- Pre-train at least California and Texas before hitting record so you don't waste video time on training.
- Hide notifications, tabs you don't need, and bookmarks bar.
- Aim for **9:30 of content + 30s buffer** so you don't go over the 10-minute cap.

---

## What to Submit

1. **Code** — the GitHub repo URL: https://github.com/Nikkclaws/sales-forecasting-system
2. **Documentation** — `README.md` + this `DEMO.md` cover everything: setup, architecture, every endpoint with sample requests, configuration, testing.
3. **Video** — your 10-minute walkthrough recorded using the script above.
4. **Optional zip** — the `sales-forecasting-system.zip` attached earlier in the conversation, if a file submission is required instead of a repo link.
