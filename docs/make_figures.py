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

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
df = pd.read_excel(ROOT / "data" / "raw" / "sales.xlsx")
df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)

# A small fraction of rows have DD-MM-YYYY strings; salvage them.
mask_bad = df["Date"].isna()
if mask_bad.any():
    df.loc[mask_bad, "Date"] = pd.to_datetime(
        df.loc[mask_bad, "Date"].astype(str), dayfirst=True, errors="coerce"
    )
df = df.dropna(subset=["Date"]).copy()
df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
df = df.dropna(subset=["Total"])

# ---------------------------------------------------------------------------
# 2. State totals bar chart
# ---------------------------------------------------------------------------
state_totals = (
    df.groupby("State")["Total"].sum().sort_values(ascending=True) / 1e9  # USD B
)
fig, ax = plt.subplots(figsize=(8, 11))
ax.barh(state_totals.index, state_totals.values, color="#4f46e5")
ax.set_title("Total beverage sales 2019\u20132023 by state (USD billions)")
ax.set_xlabel("Total sales (USD billions)")
ax.set_ylabel("")
ax.grid(axis="x", linestyle="--", alpha=0.4)
fig.tight_layout()
fig.savefig(FIG / "state_totals.png")
plt.close(fig)

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
ax.set_xlabel("")
ax.legend(loc="upper left", frameon=False)
ax.grid(linestyle="--", alpha=0.4)
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(FIG / "top5_timeseries.png")
plt.close(fig)

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
qmonths = pd.date_range(df["Date"].min(), df["Date"].max(), freq="QS").normalize()
xticks = [(present.columns.searchsorted(d)) for d in qmonths if d in present.columns or True]
xticks = np.linspace(0, present.shape[1] - 1, num=10, dtype=int)
ax.set_xticks(xticks)
ax.set_xticklabels([present.columns[i].strftime("%Y-%m") for i in xticks], rotation=45, ha="right")
ax.set_title("Data presence map (green = data present, white = missing week)")
ax.set_xlabel("Week")
fig.tight_layout()
fig.savefig(FIG / "missingness_map.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Annual seasonality (mean weekly sales across all states by week-of-year)
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
fig.tight_layout()
fig.savefig(FIG / "weekly_seasonality.png")
plt.close(fig)

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
y_offset = 0
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
fig.tight_layout()
fig.savefig(FIG / "walk_forward_cv.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Model leaderboard (uses live registry rankings.json if present)
# ---------------------------------------------------------------------------
registry_root = ROOT / "artifacts" / "registry"
manifest = registry_root / "manifest.json"
if manifest.exists():
    current = json.loads(manifest.read_text())["current"]
    rankings = []
    for state_dir in sorted((registry_root / current / "states").iterdir()):
        f = state_dir / "rankings.json"
        if not f.exists():
            continue
        info = json.loads(f.read_text())
        for r in info["rankings"]:
            rankings.append(
                {
                    "state": info["state"],
                    "model": r["model"],
                    "rmse": r["rmse"],
                    "rating_composite": r["rating_composite"],
                }
            )
    rk = pd.DataFrame(rankings)

    if not rk.empty:
        # Per-state composite-rating bar chart for the first state in the registry
        state = rk["state"].iloc[0]
        sub = rk[rk["state"] == state].sort_values("rating_composite", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.barh(sub["model"], sub["rating_composite"], color="#4f46e5")
        for x, m in zip(sub["rating_composite"], sub["model"]):
            ax.text(x + 1, sub["model"].tolist().index(m), f"{x:.1f}", va="center", fontsize=9)
        ax.set_xlim(0, 110)
        ax.set_xlabel("Composite accuracy rating (0\u2013100)")
        ax.set_title(f"Per-model composite rating \u2014 {state}")
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(FIG / "model_ratings_alabama.png")
        plt.close(fig)

        # Heatmap: state x model composite rating (only states with rankings)
        pivot = rk.pivot_table(index="state", columns="model", values="rating_composite")
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
        fig.tight_layout()
        fig.savefig(FIG / "rating_heatmap.png")
        plt.close(fig)

print("Figures written to", FIG)
for p in sorted(FIG.glob("*.png")):
    print("  -", p.name, f"{p.stat().st_size // 1024}KB")
