"""Interactive Streamlit dashboard for the sales-forecasting API.

Run locally:
    API_URL=http://localhost:8000 streamlit run dashboard/streamlit_app.py
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
HTTP_TIMEOUT = float(os.environ.get("HTTP_TIMEOUT", "60"))

st.set_page_config(
    page_title="Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom styling ----------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.0rem;
        font-weight: 700;
        margin-bottom: 0.0rem;
        color: #0e1117;
    }
    .subtle {
        color: #6c7280;
        margin-top: -0.4rem;
        margin-bottom: 1.2rem;
        font-size: 0.95rem;
    }
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e6e8eb;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="stMarkdownContainer"] h3 {
        margin-top: 0.6rem;
        margin-bottom: 0.4rem;
    }
    .insight-box {
        background-color: #eef2ff;
        border-left: 4px solid #4f46e5;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.92rem;
        color: #1e293b;
        margin: 0.6rem 0 1.0rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 10px 14px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4f46e5 !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- API helpers -------------------------------------------------------


def _safe_get(path: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{path}", params=params or {}, timeout=HTTP_TIMEOUT)
        if not r.ok:
            return None
        return r.json()
    except Exception:
        return None


@st.cache_data(ttl=30)
def fetch_health() -> dict | None:
    return _safe_get("/health")


@st.cache_data(ttl=30)
def fetch_states() -> list[str]:
    data = _safe_get("/states")
    return data.get("states", []) if data else []


@st.cache_data(ttl=30)
def fetch_metrics() -> dict | None:
    return _safe_get("/metrics")


@st.cache_data(ttl=30)
def fetch_rankings() -> dict | None:
    return _safe_get("/rankings")


@st.cache_data(ttl=30)
def fetch_predict(state: str, horizon: int, conformal: bool) -> dict | None:
    return _safe_get(
        "/predict",
        {"state": state, "horizon": horizon, "conformal": str(conformal).lower()},
    )


@st.cache_data(ttl=30)
def fetch_breakdown(state: str, horizon: int, conformal: bool) -> dict | None:
    return _safe_get(
        "/predict/breakdown",
        {"state": state, "horizon": horizon, "conformal": str(conformal).lower()},
    )


@st.cache_data(ttl=30)
def fetch_backtest(state: str) -> dict | None:
    return _safe_get("/backtest", {"state": state})


@st.cache_data(ttl=30)
def fetch_holiday(state: str) -> dict | None:
    return _safe_get("/holiday_impact", {"state": state})


def insight(text: str) -> None:
    st.markdown(f"<div class='insight-box'>💡 {text}</div>", unsafe_allow_html=True)


# ---------- Sidebar -----------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    health = fetch_health()
    if not health:
        st.error(f"API unreachable at {API_URL}.")
        st.stop()
    st.success(
        f"API up · registry **{health.get('registry_version') or '—'}** · "
        f"{health.get('states_available', 0)} states ready"
    )

    states = fetch_states()
    if not states:
        st.warning("No trained states. POST /train first.")
        st.stop()

    st.markdown("### State")
    state = st.selectbox("Select a state", states, index=0, label_visibility="collapsed")

    st.markdown("### Forecast")
    horizon = st.slider("Horizon (weeks)", min_value=1, max_value=24, value=8)
    use_conformal = st.toggle("Conformal prediction intervals", value=True)

    st.markdown("---")
    st.caption(f"API: `{API_URL}`")
    st.caption(f"Refreshed: {datetime.utcnow().isoformat(timespec='seconds')}Z")
    if st.button("🔄 Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ---------- Header + KPI cards ------------------------------------------------

st.markdown(
    "<div class='main-title'>📈 Sales Forecasting Dashboard</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtle'>Production-grade weekly state-level forecasting · "
    "5 model families · walk-forward CV · conformal intervals · drift detection</div>",
    unsafe_allow_html=True,
)

pred = fetch_predict(state, horizon, use_conformal)
metrics = fetch_metrics()
rankings = fetch_rankings()

kpi_cols = st.columns(4)
kpi_cols[0].metric("Trained states", health.get("states_available", 0))
kpi_cols[1].metric("Registry", health.get("registry_version") or "—")
if pred:
    df_pred = pd.DataFrame(pred["forecast"])
    kpi_cols[2].metric(
        f"Total forecast ({state})",
        f"${df_pred['yhat'].sum() / 1e6:,.1f}M",
        help="Sum of forecasted weekly sales over the chosen horizon.",
    )
    kpi_cols[3].metric(
        "CI method",
        pred.get("ci_method", "model_native"),
        help="Method used for prediction intervals (split-conformal vs model-native).",
    )
else:
    kpi_cols[2].metric(f"Total forecast ({state})", "—")
    kpi_cols[3].metric("CI method", "—")


# ---------- Tabs --------------------------------------------------------------

tab_forecast, tab_breakdown, tab_compare, tab_backtest, tab_holiday, tab_metrics = st.tabs(
    [
        "📊 Forecast",
        "🧩 Member breakdown",
        "🏆 Model comparison",
        "🔬 Walk-forward backtest",
        "🎉 Holiday impact",
        "📐 Model metrics",
    ]
)


with tab_forecast:
    st.subheader(f"{state} · next {horizon} weeks")
    if not pred:
        st.warning("Could not retrieve forecast from the API.")
    else:
        df = pd.DataFrame(pred["forecast"])
        df["date"] = pd.to_datetime(df["date"])

        fig = go.Figure()
        if df["yhat_upper"].notna().any() and df["yhat_lower"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=df["date"].tolist() + df["date"].tolist()[::-1],
                    y=df["yhat_upper"].tolist() + df["yhat_lower"].tolist()[::-1],
                    fill="toself",
                    fillcolor="rgba(79, 70, 229, 0.18)",
                    line={"width": 0},
                    hoverinfo="skip",
                    name="Prediction interval",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["yhat"],
                mode="lines+markers",
                name="Forecast",
                line={"width": 3, "color": "#4f46e5"},
            )
        )
        fig.update_layout(
            height=440,
            margin={"l": 10, "r": 10, "t": 30, "b": 30},
            xaxis_title="Week",
            yaxis_title="Predicted weekly sales",
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)

        wmin = df["yhat"].min()
        wmax = df["yhat"].max()
        wpeak = df.loc[df["yhat"].idxmax(), "date"].strftime("%b %d, %Y")
        insight(
            f"Forecast for **{state}** ranges from **${wmin / 1e6:,.1f}M** to "
            f"**${wmax / 1e6:,.1f}M** per week, peaking on **{wpeak}**. "
            f"Selected models: {', '.join(pred['selected_models'])}."
        )

        st.markdown("**Forecast table**")
        view = df.copy()
        view["date"] = view["date"].dt.strftime("%Y-%m-%d")
        st.dataframe(
            view.style.format(
                {
                    "yhat": "${:,.0f}",
                    "yhat_lower": "${:,.0f}",
                    "yhat_upper": "${:,.0f}",
                }
            ),
            use_container_width=True,
            height=260,
        )


with tab_breakdown:
    st.subheader("Per-member forecast comparison")
    bd = fetch_breakdown(state, horizon, use_conformal)
    if not bd:
        st.warning("Could not retrieve member breakdown.")
    else:
        weights = bd["ensemble_weights"]
        ens = pd.DataFrame(bd["ensemble"]).assign(model="ensemble")
        members = []
        for name, points in bd["members"].items():
            m = pd.DataFrame(points)
            m["model"] = name
            members.append(m)
        big = pd.concat([ens, *members], ignore_index=True)
        big["date"] = pd.to_datetime(big["date"])

        fig = px.line(big, x="date", y="yhat", color="model", markers=True, height=420)
        fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 30}, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Ensemble weights**")
        w = pd.DataFrame({"model": list(weights.keys()), "weight": list(weights.values())})
        fig2 = px.bar(w, x="model", y="weight", height=240, color="weight", color_continuous_scale="Blues")
        fig2.update_layout(
            margin={"l": 10, "r": 10, "t": 30, "b": 30},
            template="plotly_white",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        top_w = max(weights, key=weights.get) if weights else None
        if top_w:
            insight(
                f"The ensemble for **{state}** is dominated by **{top_w}** "
                f"({weights[top_w] * 100:.1f}% weight). Toggling conformal CIs in the sidebar "
                f"replaces each member's native interval with a calibrated half-width derived from "
                f"out-of-fold residuals."
            )


with tab_compare:
    st.subheader("🏆 Model comparison & accuracy ratings")
    if not rankings or not rankings.get("states"):
        st.info("No rankings available yet. Train models first via POST /train.")
    else:
        winners = rankings.get("overall_winner_counts", {})
        if winners:
            st.markdown("#### Overall: first-place wins per model (across all trained states)")
            wdf = pd.DataFrame({"model": list(winners.keys()), "wins": list(winners.values())}).sort_values(
                "wins", ascending=False
            )
            fig = px.bar(
                wdf,
                x="model",
                y="wins",
                color="wins",
                color_continuous_scale="Blues",
                height=300,
                text="wins",
            )
            fig.update_layout(
                margin={"l": 10, "r": 10, "t": 30, "b": 30},
                template="plotly_white",
                showlegend=False,
            )
            fig.update_traces(textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

            insight(
                f"**{wdf.iloc[0]['model']}** is the most frequently selected best model — "
                f"winning in **{wdf.iloc[0]['wins']} of {sum(winners.values())}** trained states. "
                "This drives the per-state ensemble composition and explains where "
                "tree-based or deep-learning approaches outperform classical statistical models."
            )

        st_entry = next((e for e in rankings["states"] if e["state"] == state), None)
        if st_entry:
            st.markdown(f"#### Leaderboard — {state}")
            lb = pd.DataFrame(st_entry["rankings"])
            st.dataframe(
                lb[
                    [
                        "rank",
                        "model",
                        "rating_composite",
                        "rmse",
                        "mae",
                        "mape",
                        "smape",
                    ]
                ].style.format(
                    {
                        "rating_composite": "{:.1f}",
                        "rmse": "{:,.0f}",
                        "mae": "{:,.0f}",
                        "mape": "{:.2f}",
                        "smape": "{:.2f}",
                    }
                ),
                use_container_width=True,
                height=240,
            )
            fig = px.bar(
                lb.sort_values("rating_composite", ascending=True),
                x="rating_composite",
                y="model",
                orientation="h",
                color="rating_composite",
                color_continuous_scale="Viridis",
                height=320,
                title=f"Composite accuracy rating (0–100) — {state}",
            )
            fig.update_layout(
                margin={"l": 10, "r": 10, "t": 40, "b": 30},
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        rows = []
        for entry in rankings["states"]:
            for r in entry["rankings"]:
                rows.append(
                    {
                        "state": entry["state"],
                        "model": r["model"],
                        "rating_composite": r["rating_composite"],
                    }
                )
        big = pd.DataFrame(rows)
        if not big.empty:
            st.markdown("#### Heatmap: composite rating across all states × models")
            pivot = big.pivot(index="state", columns="model", values="rating_composite")
            fig = px.imshow(
                pivot,
                color_continuous_scale="Viridis",
                aspect="auto",
                height=max(420, 18 * len(pivot)),
                labels={"color": "Rating"},
            )
            fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 30}, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)


with tab_backtest:
    st.subheader("Walk-forward CV: realized vs predicted")
    bt = fetch_backtest(state)
    if not bt:
        st.info("No backtest data available for this state.")
    else:
        rows = pd.DataFrame(bt["rows"])
        rows["date"] = pd.to_datetime(rows["date"])
        long = rows.melt(id_vars=["date"], var_name="series", value_name="value")
        fig = px.line(long, x="date", y="value", color="series", height=420)
        fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 30}, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

        agg = bt.get("aggregate_metrics") or {}
        if agg:
            st.markdown("#### Aggregate CV metrics")
            tbl = pd.DataFrame(agg).T
            st.dataframe(tbl.style.format("{:,.3f}"), use_container_width=True, height=240)
            best = tbl["rmse"].idxmin() if "rmse" in tbl else None
            if best:
                insight(
                    f"Across walk-forward folds, **{best}** has the lowest mean RMSE for "
                    f"**{state}**. The chart above plays back every fold's prediction vs the "
                    f"realized truth — solid validation evidence the model worked historically."
                )


with tab_holiday:
    st.subheader("Historical holiday-week sales lift")
    h = fetch_holiday(state)
    if not h:
        st.warning("Could not retrieve holiday-impact data.")
    else:
        cols = st.columns(3)
        cols[0].metric("Non-holiday avg", f"${h['non_holiday_avg'] / 1e6:,.1f}M")
        cols[1].metric("Holiday avg", f"${h['holiday_avg'] / 1e6:,.1f}M")
        cols[2].metric("Lift", f"{h['holiday_lift_pct']:+.1f}%")
        per = h.get("per_holiday") or {}
        if per:
            df = (
                pd.DataFrame(per)
                .T.reset_index()
                .rename(columns={"index": "holiday"})
                .sort_values("lift_vs_non_holiday_pct", ascending=False)
            )
            fig = px.bar(
                df,
                x="holiday",
                y="lift_vs_non_holiday_pct",
                color="lift_vs_non_holiday_pct",
                color_continuous_scale="RdYlGn",
                height=380,
            )
            fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 30}, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
            top = df.iloc[0]
            insight(
                f"**{top['holiday']}** drives the largest lift in **{state}** "
                f"(+{top['lift_vs_non_holiday_pct']:.1f}% vs the non-holiday baseline). "
                "Prophet automatically encodes this through its country-holidays regressor; "
                "XGBoost picks it up via the engineered `holiday_in_week` flag and "
                "`days_to_next_holiday` feature."
            )
            st.dataframe(df, use_container_width=True, height=240)
        else:
            st.info("No holidays detected in this state's history.")


with tab_metrics:
    st.subheader("Model registry — per-state metrics")
    if not metrics or not metrics.get("states"):
        st.info("No metrics available yet.")
    else:
        recs = []
        for entry in metrics["states"]:
            for model, vals in entry["aggregate_metrics"].items():
                recs.append(
                    {
                        "state": entry["state"],
                        "model": model,
                        **vals,
                        "weight": entry["ensemble_weights"].get(model),
                    }
                )
        df = pd.DataFrame(recs)
        st.dataframe(
            df.sort_values(["state", "rmse"]),
            use_container_width=True,
            height=320,
        )
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x="model", y="rmse", color="model", height=380)
            fig.update_layout(
                margin={"l": 10, "r": 10, "t": 30, "b": 30},
                template="plotly_white",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sel = df[df["state"] == state]
            if not sel.empty:
                fig = px.bar(
                    sel,
                    x="model",
                    y="rmse",
                    color="rmse",
                    color_continuous_scale="Reds",
                    title=f"{state} RMSE per model",
                    height=380,
                )
                fig.update_layout(
                    margin={"l": 10, "r": 10, "t": 40, "b": 30},
                    template="plotly_white",
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)


# ---------- Footer ------------------------------------------------------------

st.markdown("---")
st.caption(
    "Built with FastAPI + Streamlit + Plotly · "
    "5 model families compared per state · "
    "Conformal prediction · Drift detection · Versioned model registry"
)
