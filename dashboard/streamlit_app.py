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
)


@st.cache_data(ttl=30)
def fetch_health() -> dict:
    return requests.get(f"{API_URL}/health", timeout=HTTP_TIMEOUT).json()


@st.cache_data(ttl=30)
def fetch_states() -> list[str]:
    r = requests.get(f"{API_URL}/states", timeout=HTTP_TIMEOUT)
    if r.ok:
        return r.json().get("states", [])
    return []


@st.cache_data(ttl=30)
def fetch_metrics() -> dict:
    r = requests.get(f"{API_URL}/metrics", timeout=HTTP_TIMEOUT)
    return r.json() if r.ok else {}


def fetch_predict(state: str, horizon: int, conformal: bool) -> dict:
    r = requests.get(
        f"{API_URL}/predict",
        params={"state": state, "horizon": horizon, "conformal": str(conformal).lower()},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def fetch_breakdown(state: str, horizon: int, conformal: bool) -> dict:
    r = requests.get(
        f"{API_URL}/predict/breakdown",
        params={"state": state, "horizon": horizon, "conformal": str(conformal).lower()},
        timeout=HTTP_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()


def fetch_backtest(state: str) -> dict:
    r = requests.get(f"{API_URL}/backtest", params={"state": state}, timeout=HTTP_TIMEOUT)
    if r.status_code == 404:
        return {}
    r.raise_for_status()
    return r.json()


def fetch_holiday(state: str) -> dict:
    r = requests.get(f"{API_URL}/holiday_impact", params={"state": state}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    return r.json()


# ----- Sidebar -----------------------------------------------------------

with st.sidebar:
    st.markdown("## Controls")
    try:
        health = fetch_health()
        st.success(
            f"API up • registry `{health.get('registry_version') or '—'}` • "
            f"{health.get('states_available', 0)} states ready"
        )
    except Exception as e:  # noqa: BLE001
        st.error(f"API unreachable at {API_URL}: {e}")
        st.stop()
    states = fetch_states()
    if not states:
        st.warning("No trained states found. POST /train first.")
        st.stop()
    state = st.selectbox("State", states, index=0)
    horizon = st.slider("Forecast horizon (weeks)", min_value=1, max_value=24, value=8)
    use_conformal = st.toggle("Conformal prediction intervals", value=True)
    st.caption(f"API: `{API_URL}`")
    st.caption(f"Last refresh: {datetime.utcnow().isoformat(timespec='seconds')}Z")

# ----- Main tabs ---------------------------------------------------------

tab_forecast, tab_breakdown, tab_backtest, tab_holiday, tab_metrics = st.tabs(
    ["Forecast", "Member breakdown", "Walk-forward backtest", "Holiday impact", "Model metrics"]
)

with tab_forecast:
    st.subheader(f"{state} — next {horizon} weeks")
    pred = fetch_predict(state, horizon, use_conformal)
    df = pd.DataFrame(pred["forecast"])
    df["date"] = pd.to_datetime(df["date"])

    fig = go.Figure()
    if df["yhat_upper"].notna().any() and df["yhat_lower"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=df["date"].tolist() + df["date"].tolist()[::-1],
                y=df["yhat_upper"].tolist() + df["yhat_lower"].tolist()[::-1],
                fill="toself",
                fillcolor="rgba(31,119,180,0.18)",
                line={"width": 0},
                hoverinfo="skip",
                name="Prediction interval",
            )
        )
    fig.add_trace(go.Scatter(x=df["date"], y=df["yhat"], mode="lines+markers", name="Forecast"))
    fig.update_layout(
        height=420,
        margin={"l": 10, "r": 10, "t": 30, "b": 30},
        xaxis_title="Week",
        yaxis_title="Predicted weekly sales",
    )
    st.plotly_chart(fig, use_container_width=True)

    cols = st.columns(4)
    cols[0].metric("CI method", pred.get("ci_method", "model_native"))
    cols[1].metric("Selected models", ", ".join(pred["selected_models"]))
    cols[2].metric("Mean yhat", f"{df['yhat'].mean():,.0f}")
    cols[3].metric("Total forecast", f"{df['yhat'].sum():,.0f}")
    st.dataframe(df, use_container_width=True)

with tab_breakdown:
    st.subheader("Per-member forecast comparison")
    bd = fetch_breakdown(state, horizon, use_conformal)
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
    fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 30})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Ensemble weights**")
    w = pd.DataFrame({"model": list(weights.keys()), "weight": list(weights.values())})
    fig2 = px.bar(w, x="model", y="weight", height=240)
    st.plotly_chart(fig2, use_container_width=True)

with tab_backtest:
    st.subheader("Walk-forward CV: realized vs predicted")
    bt = fetch_backtest(state)
    if not bt:
        st.info("No backtest data available. (Old training run before the cv_predictions.csv feature?)")
    else:
        rows = pd.DataFrame(bt["rows"])
        rows["date"] = pd.to_datetime(rows["date"])
        long = rows.melt(id_vars=["date"], var_name="series", value_name="value")
        fig = px.line(long, x="date", y="value", color="series", markers=False, height=420)
        fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 30})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Aggregate CV metrics**")
        agg = bt.get("aggregate_metrics") or {}
        if agg:
            tbl = pd.DataFrame(agg).T
            st.dataframe(tbl.style.format("{:.3f}"), use_container_width=True)

with tab_holiday:
    st.subheader("Historical holiday-week sales lift")
    h = fetch_holiday(state)
    cols = st.columns(3)
    cols[0].metric("Non-holiday avg", f"{h['non_holiday_avg']:,.0f}")
    cols[1].metric("Holiday avg", f"{h['holiday_avg']:,.0f}")
    cols[2].metric("Lift", f"{h['holiday_lift_pct']:.1f}%")
    per = h.get("per_holiday") or {}
    if per:
        df = (
            pd.DataFrame(per)
            .T.reset_index()
            .rename(columns={"index": "holiday"})
            .sort_values("lift_vs_non_holiday_pct", ascending=False)
        )
        st.dataframe(df, use_container_width=True)
        fig = px.bar(df, x="holiday", y="lift_vs_non_holiday_pct", height=360)
        fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 30})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No holidays detected in this state's history.")

with tab_metrics:
    st.subheader("Model registry — per-state metrics")
    m = fetch_metrics()
    if m and m.get("states"):
        recs = []
        for entry in m["states"]:
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
        st.dataframe(df.sort_values(["state", "rmse"]), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.box(df, x="model", y="rmse", height=380)
            fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 30})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            sel = df[df["state"] == state]
            if not sel.empty:
                fig = px.bar(sel, x="model", y="rmse", title=f"{state} RMSE per model")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No metrics available yet.")
