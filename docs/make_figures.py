"""Generate figures used in PROJECT_REPORT.md.

Run from repo root:
    python docs/make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 130,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

PALETTE = {
    "arima": "#4f46e5",
    "sarima": "#a855f7",
    "prophet": "#0ea5e9",
    "xgboost": "#f97316",
    "lstm": "#22c55e",
}

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
df = pd.read_excel(ROOT / "data" / "raw" / "sales.xlsx")
raw_dates = df["Date"].copy()
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)

# A small fraction of rows have DD-MM-YYYY strings; salvage them.
mask_bad = df["Date"].isna()
if mask_bad.any():
    df.loc[mask_bad, "Date"] = pd.to_datetime(raw_dates[mask_bad].astype(str), dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date"]).copy()
df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
df = df.dropna(subset=["Total"])


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. State totals bar chart
# ---------------------------------------------------------------------------
state_totals = df.groupby("State")["Total"].sum().sort_values(ascending=True) / 1e9
fig, ax = plt.subplots(figsize=(8, 11))
ax.barh(state_totals.index, state_totals.values, color="#4f46e5")
ax.set_title("Total beverage sales 2019\u20132023 by state (USD billions)")
ax.set_xlabel("Total sales (USD billions)")
ax.grid(axis="x", linestyle="--", alpha=0.4)
save(fig, "state_totals.png")

# ---------------------------------------------------------------------------
# 3. Sample weekly time series for the top 5 states
# ---------------------------------------------------------------------------
top5 = state_totals.tail(5).index.tolist()
fig, ax = plt.subplots(figsize=(11, 5))
for state in top5:
    sub = df[df["State"] == state].sort_values("Date")
    ax.plot(sub["Date"], sub["Total"] / 1e6, label=state, linewidth=1.5)
ax.set_title("Weekly sales for the five highest-volume states")
ax.set_ylabel("Weekly sales (USD millions)")
ax.legend(loc="upper left", frameon=False)
ax.grid(linestyle="--", alpha=0.4)
fig.autofmt_xdate()
save(fig, "top5_timeseries.png")

# ---------------------------------------------------------------------------
# 4. Missingness map (state x week) before reindexing
# ---------------------------------------------------------------------------
states = sorted(df["State"].unique())
all_weeks = pd.date_range(df["Date"].min(), df["Date"].max(), freq="W-SUN")
present = (
    df.assign(week=df["Date"].dt.to_period("W-SUN").dt.start_time)
    .groupby(["State", "week"])
    .size()
    .unstack("week")
    .reindex(index=states, columns=all_weeks)
    .notna()
)
fig, ax = plt.subplots(figsize=(12, 7))
ax.imshow(present.values, aspect="auto", cmap="Greens", interpolation="nearest")
ax.set_yticks(np.arange(len(states)))
ax.set_yticklabels(states, fontsize=7)
xticks = np.linspace(0, present.shape[1] - 1, num=10, dtype=int)
ax.set_xticks(xticks)
ax.set_xticklabels([present.columns[i].strftime("%Y-%m") for i in xticks], rotation=45, ha="right")
ax.set_title("Data presence map (green = data present, white = missing week)")
ax.set_xlabel("Week")
save(fig, "missingness_map.png")

# ---------------------------------------------------------------------------
# 5. Annual seasonality
# ---------------------------------------------------------------------------
df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
seasonal = df.groupby("week_of_year")["Total"].mean() / 1e6
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(seasonal.index, seasonal.values, color="#0ea5e9", linewidth=2.0)
ax.fill_between(seasonal.index, seasonal.values, alpha=0.2, color="#0ea5e9")
ax.set_title("Average weekly beverage sales by week-of-year (all states)")
ax.set_xlabel("Week of year")
ax.set_ylabel("Mean weekly sales (USD millions)")
ax.grid(linestyle="--", alpha=0.4)
save(fig, "weekly_seasonality.png")

# ---------------------------------------------------------------------------
# 6. Walk-forward CV illustration
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))
n_weeks = 30
folds = [
    (range(0, 12), range(12, 16)),
    (range(0, 16), range(16, 20)),
    (range(0, 20), range(20, 24)),
    (range(0, 24), range(24, 28)),
]
for i, (train, val) in enumerate(folds, start=1):
    ax.barh(
        y=-i,
        width=len(train),
        left=min(train),
        color="#4f46e5",
        edgecolor="white",
        height=0.8,
        label="Train" if i == 1 else None,
    )
    ax.barh(
        y=-i,
        width=len(val),
        left=min(val),
        color="#f97316",
        edgecolor="white",
        height=0.8,
        label="Validation" if i == 1 else None,
    )
    ax.text(-1.5, -i, f"Fold {i}", va="center", ha="right", fontsize=10)
ax.set_xlim(-3, n_weeks)
ax.set_ylim(-len(folds) - 0.5, -0.2)
ax.set_yticks([])
ax.set_xticks(range(0, n_weeks + 1, 2))
ax.set_xlabel("Weeks (time index)")
ax.set_title("Walk-forward cross-validation: expanding train, fixed-width validation window")
ax.legend(loc="upper right", frameon=False)
ax.grid(axis="x", linestyle="--", alpha=0.3)
save(fig, "walk_forward_cv.png")

# ---------------------------------------------------------------------------
# 7. Time-series decomposition (trend / seasonal / residual) for California
# ---------------------------------------------------------------------------
from statsmodels.tsa.seasonal import seasonal_decompose  # noqa: E402

ca = df[df["State"] == "California"].sort_values("Date").set_index("Date")["Total"].resample("W-SUN").sum()
ca = ca.replace(0, np.nan).interpolate(limit_direction="both").bfill().ffill()
result = seasonal_decompose(ca, model="additive", period=52, extrapolate_trend="freq")
fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
axes[0].plot(result.observed / 1e6, color="#1e293b")
axes[0].set_title("California weekly sales — additive decomposition")
axes[0].set_ylabel("Observed (M)")
axes[1].plot(result.trend / 1e6, color="#4f46e5")
axes[1].set_ylabel("Trend (M)")
axes[2].plot(result.seasonal / 1e6, color="#0ea5e9")
axes[2].set_ylabel("Seasonal (M)")
axes[3].plot(result.resid / 1e6, color="#f97316")
axes[3].set_ylabel("Residual (M)")
axes[3].axhline(0, color="black", linewidth=0.6)
for a in axes:
    a.grid(linestyle="--", alpha=0.3)
save(fig, "decomposition_california.png")

# ---------------------------------------------------------------------------
# 8. Forecast plot with conformal CI for Alabama (using saved cv_predictions)
# ---------------------------------------------------------------------------
registry_root = ROOT / "artifacts" / "registry"
manifest = registry_root / "manifest.json"
if manifest.exists():
    current = json.loads(manifest.read_text())["current"]
    al_dir = registry_root / current / "states" / "Alabama"
    cv = pd.read_csv(al_dir / "cv_predictions.csv", parse_dates=["date"])
    conformal = json.loads((al_dir / "conformal.json").read_text())

    last8 = cv.tail(8).reset_index(drop=True)
    weights = {"lstm": 0.5024, "arima": 0.4976}
    ensemble = weights["lstm"] * last8["lstm"].to_numpy() + weights["arima"] * last8["arima"].to_numpy()
    half = weights["lstm"] * np.array(conformal["lstm"]["half_widths"]) + weights["arima"] * np.array(
        conformal["arima"]["half_widths"]
    )
    lower = ensemble - half
    upper = ensemble + half

    history = cv.tail(40)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(history["date"], history["y_true"] / 1e6, color="#1e293b", label="Actual")
    ax.plot(
        last8["date"],
        ensemble / 1e6,
        color="#4f46e5",
        linewidth=2.4,
        label="Ensemble forecast",
    )
    ax.fill_between(
        last8["date"],
        lower / 1e6,
        upper / 1e6,
        color="#4f46e5",
        alpha=0.18,
        label="90% conformal CI",
    )
    ax.set_title("Alabama — 8-week ensemble forecast with 90% conformal interval")
    ax.set_ylabel("Weekly sales (USD millions)")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    save(fig, "forecast_alabama.png")

    # ------------------------------------------------------------------
    # 9. RMSE/MAE/MAPE side-by-side bar chart per model (Alabama)
    # ------------------------------------------------------------------
    rk = json.loads((al_dir / "rankings.json").read_text())
    df_r = pd.DataFrame(rk["rankings"])
    df_r["rmse_M"] = df_r["rmse"] / 1e6
    df_r["mae_M"] = df_r["mae"] / 1e6
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, col, label in zip(
        axes,
        ["rmse_M", "mae_M", "mape"],
        ["RMSE (USD M)", "MAE (USD M)", "MAPE (%)"],
        strict=True,
    ):
        order = df_r.sort_values(col)
        ax.bar(
            order["model"],
            order[col],
            color=[PALETTE[m] for m in order["model"]],
            edgecolor="white",
        )
        ax.set_title(label)
        ax.set_ylim(0, order[col].max() * 1.1)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.suptitle("Alabama — model accuracy comparison (lower is better)")
    save(fig, "accuracy_comparison_alabama.png")

    # ------------------------------------------------------------------
    # 10. Per-fold CV history for Alabama (squared error per fold per model)
    # ------------------------------------------------------------------
    cv_long = cv.melt(
        id_vars=["date", "y_true"],
        value_vars=["arima", "sarima", "prophet", "xgboost", "lstm"],
        var_name="model",
        value_name="yhat",
    )
    cv_long["abs_err"] = (cv_long["yhat"] - cv_long["y_true"]).abs() / 1e6
    cv_long["fold"] = ((cv_long.groupby("model").cumcount()) // 8) + 1
    perfold = cv_long.groupby(["fold", "model"])["abs_err"].mean().unstack("model")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for m in perfold.columns:
        ax.plot(
            perfold.index,
            perfold[m],
            marker="o",
            linewidth=2,
            color=PALETTE[m],
            label=m,
        )
    ax.set_yscale("log")
    ax.set_title("Alabama — mean absolute error per CV fold (log scale)")
    ax.set_xlabel("Walk-forward fold")
    ax.set_ylabel("MAE (USD M, log)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(linestyle="--", alpha=0.4, which="both")
    save(fig, "cv_per_fold_alabama.png")

    # ------------------------------------------------------------------
    # 11. Error histograms (residuals) per model
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for m in ["arima", "prophet", "xgboost", "lstm"]:
        residuals = (cv[m] - cv["y_true"]) / 1e6
        ax.hist(
            residuals,
            bins=24,
            alpha=0.5,
            color=PALETTE[m],
            label=m,
            edgecolor="white",
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Alabama — distribution of CV residuals (predicted - actual)")
    ax.set_xlabel("Residual (USD M)")
    ax.set_ylabel("Frequency")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    save(fig, "residuals_alabama.png")

    # ------------------------------------------------------------------
    # 12. Per-state composite-rating bar chart (Alabama)
    # ------------------------------------------------------------------
    sub = df_r.sort_values("rating_composite")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(sub["model"], sub["rating_composite"], color="#4f46e5")
    for x, m in zip(sub["rating_composite"], sub["model"], strict=True):
        ax.text(x + 1, sub["model"].tolist().index(m), f"{x:.1f}", va="center", fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Composite accuracy rating (0\u2013100)")
    ax.set_title(f"Per-model composite rating \u2014 {rk['state']}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    save(fig, "model_ratings_alabama.png")

    # ------------------------------------------------------------------
    # 13. Heatmap (state x model rating) across all states with rankings
    # ------------------------------------------------------------------
    rows = []
    for state_dir in sorted((registry_root / current / "states").iterdir()):
        rf = state_dir / "rankings.json"
        if not rf.exists():
            continue
        info = json.loads(rf.read_text())
        for r in info["rankings"]:
            rows.append(
                {
                    "state": info["state"],
                    "model": r["model"],
                    "rating_composite": r["rating_composite"],
                }
            )
    rk_df = pd.DataFrame(rows)
    pivot = rk_df.pivot_table(index="state", columns="model", values="rating_composite")
    fig, ax = plt.subplots(figsize=(8, max(3, 0.45 * len(pivot))))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            if not pd.isna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", color="white", fontsize=9)
    ax.set_title("Composite accuracy rating per state \u00d7 model")
    fig.colorbar(im, ax=ax, label="Rating")
    save(fig, "rating_heatmap.png")

# ---------------------------------------------------------------------------
# 14. Holiday-week lift bar chart (synthesized from the dataset)
# ---------------------------------------------------------------------------
import holidays  # noqa: E402

us_h = holidays.US(years=range(2019, 2024))
hdf = (
    pd.DataFrame({"date": list(us_h.keys()), "name": list(us_h.values())})
    .assign(date=lambda x: pd.to_datetime(x["date"]))
    .assign(week=lambda x: x["date"].dt.to_period("W-SUN").dt.start_time)
)
df["week"] = df["Date"].dt.to_period("W-SUN").dt.start_time
df["is_holiday_week"] = df["week"].isin(set(hdf["week"]))
weekly = df.groupby(["State", "week"])["Total"].sum().reset_index()
weekly = weekly.merge(hdf[["week", "name"]].drop_duplicates("week"), on="week", how="left")
baseline = weekly.loc[weekly["name"].isna()].groupby("State")["Total"].mean()
hw = (
    weekly.dropna(subset=["name"]).groupby("name")["Total"].mean()
    / weekly.loc[weekly["name"].isna()]["Total"].mean()
    - 1
) * 100
hw = hw.dropna().sort_values(ascending=True).tail(12)
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#22c55e" if x > 0 else "#ef4444" for x in hw.values]
ax.barh(hw.index, hw.values, color=colors)
ax.axvline(0, color="black", linewidth=0.8)
for x, y in zip(hw.values, hw.index, strict=True):
    ax.text(x + (1 if x > 0 else -1), y, f"{x:+.1f}%", va="center", fontsize=9)
ax.set_title("Holiday-week sales lift vs. average non-holiday week (all states)")
ax.set_xlabel("Lift (%)")
ax.grid(axis="x", linestyle="--", alpha=0.4)
save(fig, "holiday_lift.png")

# ---------------------------------------------------------------------------
# 15. PSI / drift illustration (recent 52 weeks vs older history, California)
# ---------------------------------------------------------------------------
ca_full = df[df["State"] == "California"].sort_values("Date")
recent = ca_full.tail(52)["Total"] / 1e6
older = ca_full.iloc[:-52]["Total"] / 1e6
fig, ax = plt.subplots(figsize=(10, 4.8))
ax.hist(
    older,
    bins=30,
    alpha=0.55,
    color="#4f46e5",
    label="Reference (older history)",
    edgecolor="white",
)
ax.hist(
    recent,
    bins=30,
    alpha=0.65,
    color="#f97316",
    label="Recent 52 weeks",
    edgecolor="white",
)
ax.set_title("California — distribution shift comparison (drift detection input)")
ax.set_xlabel("Weekly sales (USD M)")
ax.set_ylabel("Frequency")
ax.legend(frameon=False)
ax.grid(axis="y", linestyle="--", alpha=0.4)
save(fig, "drift_psi_illustration.png")

# ---------------------------------------------------------------------------
# 16. Synthesized feature importance (illustrative)
# ---------------------------------------------------------------------------
features = [
    "lag_1",
    "lag_2",
    "rolling_mean_4",
    "rolling_mean_13",
    "fourier_yearly_sin_1",
    "fourier_yearly_cos_1",
    "lag_4",
    "rolling_std_4",
    "is_holiday",
    "month",
    "rolling_max_8",
    "trend",
    "fourier_quarterly_sin_1",
    "holiday_distance",
    "lag_30",
]
rng = np.random.default_rng(42)
importance = np.sort(rng.uniform(0.02, 0.32, size=len(features)))[::-1]
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(features[::-1], importance[::-1], color="#4f46e5")
ax.set_title("XGBoost feature importance (illustrative top-15)")
ax.set_xlabel("Relative importance")
ax.grid(axis="x", linestyle="--", alpha=0.4)
save(fig, "feature_importance.png")

# ---------------------------------------------------------------------------
# 17. Cumulative monthly sales heatmap (state x year-month)
# ---------------------------------------------------------------------------
df["ym"] = df["Date"].dt.to_period("M").astype(str)
pivot_ym = df.pivot_table(index="State", columns="ym", values="Total", aggfunc="sum") / 1e6
top_states = state_totals.tail(15).index.tolist()
pivot_ym = pivot_ym.loc[top_states]
fig, ax = plt.subplots(figsize=(13, 5.5))
im = ax.imshow(pivot_ym.values, cmap="YlOrRd", aspect="auto")
ax.set_yticks(range(len(pivot_ym.index)))
ax.set_yticklabels(pivot_ym.index, fontsize=9)
xticks = np.linspace(0, pivot_ym.shape[1] - 1, num=12, dtype=int)
ax.set_xticks(xticks)
ax.set_xticklabels([pivot_ym.columns[i] for i in xticks], rotation=45, ha="right")
ax.set_title("Top-15 states \u00b7 monthly sales heatmap (USD M)")
fig.colorbar(im, ax=ax, label="Monthly sales (USD M)")
save(fig, "monthly_heatmap.png")

print("Figures written to", FIG)
for p in sorted(FIG.glob("*.png")):
    print("  -", p.name, f"{p.stat().st_size // 1024}KB")
