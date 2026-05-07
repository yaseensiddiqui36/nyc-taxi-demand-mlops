"""
Streamlit dashboard — NYC Taxi Demand Forecasting.
Shows live predictions on an interactive NYC choropleth map,
top demand zones, monitoring badges, and time-series charts.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="NYC Taxi Demand — Live Forecasting",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helpers ──────────────────────────────────────────────────


@st.cache_data(ttl=60)
def fetch_predictions() -> pd.DataFrame:
    try:
        resp = requests.post(f"{API_URL}/predict", json={}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["predictions"])
        df["meta_model_version"] = data.get("model_version", "unknown")
        df["meta_latency_ms"] = data.get("latency_ms", 0)
        return df
    except Exception as e:
        st.error(f"API error: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_monitoring_metrics() -> dict:
    try:
        resp = requests.get(f"{API_URL}/monitoring/metrics", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


@st.cache_data(ttl=30)
def api_health() -> dict:
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except Exception:
        return {"status": "unreachable"}


# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.title("🚕 NYC Taxi Demand")
    st.caption("Real-time hourly demand forecasting")
    st.divider()

    health = api_health()
    status_color = "🟢" if health.get("status") == "healthy" else "🔴"
    st.markdown(f"**API** {status_color} `{health.get('status', 'unknown')}`")
    st.markdown(
        f"**DB** {'🟢' if health.get('database') == 'ok' else '🔴'} `{health.get('database', 'unknown')}`"
    )
    st.markdown(f"**Model** {'🟢' if health.get('model_loaded') else '🔴'} loaded")
    st.divider()

    top_n = st.slider("Top N zones to highlight", 5, 30, 10)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.markdown("**Quick links**")
    st.markdown("[Grafana Dashboard](http://localhost:3000)")
    st.markdown("[MLflow Experiments](http://localhost:5000)")
    st.markdown("[Airflow DAGs](http://localhost:8080)")

if auto_refresh:
    import time

    time.sleep(60)
    st.cache_data.clear()
    st.rerun()

# ── Main content ─────────────────────────────────────────────

st.title("🚕 NYC Taxi Demand — Live Forecast")
st.caption(
    f"Predictions for: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:00 UTC')}"
)

predictions_df = fetch_predictions()
monitoring = fetch_monitoring_metrics()

if predictions_df.empty:
    st.warning(
        "No predictions available. Make sure the API is running and a model is registered."
    )
    st.stop()

# ── KPI row ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_predicted = predictions_df["predicted_rides"].sum()
    st.metric("Total Predicted Rides (all zones)", f"{total_predicted:,.0f}")

with col2:
    mae_val = monitoring.get("model_mae", {}).get("value", "N/A")
    st.metric("Model MAE", f"{mae_val:.1f}" if isinstance(mae_val, float) else mae_val)

with col3:
    drift_val = monitoring.get("data_drift_score", {}).get("value", "N/A")
    drift_label = f"{drift_val:.3f}" if isinstance(drift_val, float) else drift_val
    drift_delta = (
        "⚠️ Drifting"
        if isinstance(drift_val, float) and drift_val > 0.25
        else "✅ Stable"
    )
    st.metric("Data Drift Score", drift_label, delta=drift_delta)

with col4:
    latency = (
        predictions_df["meta_latency_ms"].iloc[0] if not predictions_df.empty else "N/A"
    )
    st.metric(
        "Prediction Latency",
        f"{latency:.0f} ms" if isinstance(latency, float) else latency,
    )

st.divider()

# ── Top zones bar chart ───────────────────────────────────────
top_zones = predictions_df.nlargest(top_n, "predicted_rides")
col_chart, col_table = st.columns([2, 1])

with col_chart:
    st.subheader(f"Top {top_n} Zones by Predicted Demand")
    fig = px.bar(
        top_zones.sort_values("predicted_rides"),
        x="predicted_rides",
        y="pickup_location_id",
        orientation="h",
        labels={"predicted_rides": "Predicted Rides", "pickup_location_id": "Zone ID"},
        color="predicted_rides",
        color_continuous_scale="Oranges",
    )
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

with col_table:
    st.subheader("Zone Rankings")
    display_df = top_zones[["pickup_location_id", "predicted_rides"]].copy()
    display_df.columns = ["Zone ID", "Predicted Rides"]
    display_df["Predicted Rides"] = display_df["Predicted Rides"].round(1)
    display_df = display_df.sort_values("Predicted Rides", ascending=False).reset_index(
        drop=True
    )
    display_df.index += 1
    st.dataframe(display_df, use_container_width=True)

st.divider()

# ── Distribution chart ────────────────────────────────────────
st.subheader("Demand Distribution Across All Zones")
fig2 = px.histogram(
    predictions_df,
    x="predicted_rides",
    nbins=40,
    labels={"predicted_rides": "Predicted Rides"},
    color_discrete_sequence=["#FF8C00"],
)
fig2.update_layout(height=300)
st.plotly_chart(fig2, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data source: NYC TLC | Model: MLflow Registry | "
    "Feature store: Feast + Redis | Monitoring: Evidently AI + Grafana"
)
