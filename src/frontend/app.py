"""
NYC Taxi Demand — Live Forecast Dashboard
Interactive choropleth map with zone selection, demand analytics, and monitoring.
"""

from __future__ import annotations

import io
import os
import zipfile
from datetime import datetime, timezone

import folium
import geopandas as gpd
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from shapely.geometry import Point
from streamlit_folium import st_folium

API_URL = os.getenv("API_URL", "http://localhost:8000")
ZONE_LOOKUP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZONE_SHAPE_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
GEOJSON_CACHE = "/tmp/nyc_taxi_zones.geojson"

st.set_page_config(
    page_title="NYC Taxi Demand",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state init ────────────────────────────────────────
if "selected_zone" not in st.session_state:
    st.session_state.selected_zone = None


# ── Data loaders ─────────────────────────────────────────────

@st.cache_data(ttl=86400)
def load_zone_lookup() -> pd.DataFrame:
    try:
        df = pd.read_csv(ZONE_LOOKUP_URL, timeout=15)
        df.columns = ["location_id", "borough", "zone_name", "service_zone"]
        df["location_id"] = df["location_id"].astype(int)
        return df
    except Exception:
        return pd.DataFrame(
            columns=["location_id", "borough", "zone_name", "service_zone"]
        )


@st.cache_resource
def load_zone_geodata():
    if os.path.exists(GEOJSON_CACHE):
        return gpd.read_file(GEOJSON_CACHE)
    try:
        resp = requests.get(ZONE_SHAPE_URL, timeout=45)
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            z.extractall("/tmp/nyc_zones_shp")
        gdf = gpd.read_file("/tmp/nyc_zones_shp/taxi_zones.shp")
        gdf = gdf.to_crs(epsg=4326)
        gdf["LocationID"] = gdf["LocationID"].astype(int)
        gdf.to_file(GEOJSON_CACHE, driver="GeoJSON")
        return gdf
    except Exception:
        return None


@st.cache_data(ttl=60)
def fetch_predictions() -> pd.DataFrame:
    try:
        resp = requests.post(f"{API_URL}/predict", json={}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        df = pd.DataFrame(data["predictions"])
        df["meta_latency_ms"] = data.get("latency_ms", 0)
        df["predicted_hour"] = data["predictions"][0].get("predicted_hour", "")
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
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except Exception:
        return {"status": "unreachable"}


# ── Map builder ───────────────────────────────────────────────

def build_choropleth(
    predictions_df: pd.DataFrame,
    zone_lookup: pd.DataFrame,
    gdf,
    selected_zone_id: int | None,
) -> folium.Map:
    import branca.colormap as cm

    merged = gdf.copy()
    merged = merged.merge(
        predictions_df[["pickup_location_id", "predicted_rides"]],
        left_on="LocationID", right_on="pickup_location_id", how="left",
    )
    merged = merged.merge(
        zone_lookup[["location_id", "zone_name", "borough"]],
        left_on="LocationID", right_on="location_id", how="left",
    )
    merged["predicted_rides"] = merged["predicted_rides"].fillna(0).round(1)
    merged["zone_name"] = merged["zone_name"].fillna("Unknown")
    merged["borough"] = merged["borough"].fillna("Unknown")

    m = folium.Map(
        location=[40.71, -73.94],
        zoom_start=11,
        tiles="CartoDB positron",
        prefer_canvas=True,
    )

    max_rides = max(merged["predicted_rides"].max(), 1)
    colormap = cm.LinearColormap(
        colors=["#ffffb2", "#fecc5c", "#fd8d3c", "#f03b20", "#bd0026"],
        vmin=0,
        vmax=max_rides,
        caption="Predicted Rides / Hour",
    )

    def style_fn(feature):
        loc_id = feature["properties"].get("LocationID")
        rides = feature["properties"].get("predicted_rides", 0) or 0
        is_selected = loc_id == selected_zone_id
        return {
            "fillColor": colormap(rides),
            "fillOpacity": 0.75,
            "color": "#2563eb" if is_selected else "#555555",
            "weight": 3 if is_selected else 0.4,
        }

    folium.GeoJson(
        data=merged.__geo_interface__,
        style_function=style_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=["zone_name", "borough", "predicted_rides"],
            aliases=["Zone:", "Borough:", "Predicted Rides:"],
            style="font-size:13px; padding:6px;",
            sticky=True,
        ),
    ).add_to(m)

    colormap.add_to(m)
    return m


# ── Zone detail panel ─────────────────────────────────────────

def render_zone_details(zone_id: int, zone_lookup: pd.DataFrame, predictions_df: pd.DataFrame):
    zone_row = zone_lookup[zone_lookup["location_id"] == zone_id]
    pred_row = predictions_df[predictions_df["pickup_location_id"] == zone_id]

    if zone_row.empty:
        st.warning(f"Zone {zone_id} not found in lookup.")
        return

    z = zone_row.iloc[0]
    p = pred_row.iloc[0] if not pred_row.empty else None

    c1, c2 = st.columns([3, 1])
    with c1:
        st.subheader(f"📍 {z['zone_name']}")
        st.caption(f"**Borough:** {z['borough']}  ·  **Zone ID:** {zone_id}  ·  **Service zone:** {z.get('service_zone', 'N/A')}")
    with c2:
        if st.button("✕ Clear selection", key="clear_zone"):
            st.session_state.selected_zone = None
            st.rerun()

    if p is not None:
        city_avg = predictions_df["predicted_rides"].mean()
        zone_rides = float(p["predicted_rides"])
        pct_vs_avg = ((zone_rides - city_avg) / city_avg * 100) if city_avg else 0
        delta_str = f"{pct_vs_avg:+.0f}% vs city avg"

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Predicted Rides (next hour)", f"{zone_rides:,.0f}", delta=delta_str)
        mc2.metric("City Average (all zones)", f"{city_avg:,.1f}")
        city_rank = (predictions_df["predicted_rides"] >= zone_rides).sum()
        mc3.metric("City Rank", f"#{city_rank} of {len(predictions_df)}")
    else:
        st.info("No prediction available for this zone.")

    st.info(
        "Origin-Destination analytics (top destinations, avg fare, avg duration) "
        "will be available in the next phase when O-D data is loaded."
    )


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🚕 NYC Taxi Demand")
    st.caption("Real-time hourly demand forecasting")
    st.divider()

    health = api_health()
    st.markdown(f"**API** {'🟢' if health.get('status') == 'healthy' else '🔴'} `{health.get('status', 'unknown')}`")
    st.markdown(f"**DB** {'🟢' if health.get('database') == 'ok' else '🔴'} `{health.get('database', 'unknown')}`")
    st.markdown(f"**Model** {'🟢' if health.get('model_loaded') else '🔴'} `{'loaded' if health.get('model_loaded') else 'not loaded'}`")
    st.divider()

    zone_lookup = load_zone_lookup()

    if not zone_lookup.empty:
        zone_options = {
            f"{row['zone_name']} ({row['borough']})": int(row["location_id"])
            for _, row in zone_lookup.iterrows()
        }
        zone_options_sorted = dict(sorted(zone_options.items()))
        zone_labels = ["— Select a zone —"] + list(zone_options_sorted.keys())

        current_label = "— Select a zone —"
        if st.session_state.selected_zone:
            for label, zid in zone_options_sorted.items():
                if zid == st.session_state.selected_zone:
                    current_label = label
                    break

        selected_label = st.selectbox(
            "Select Zone",
            zone_labels,
            index=zone_labels.index(current_label),
            help="Search and select a zone by name, or click on the map",
        )
        if selected_label != "— Select a zone —":
            new_id = zone_options_sorted[selected_label]
            if new_id != st.session_state.selected_zone:
                st.session_state.selected_zone = new_id
                st.rerun()
        elif st.session_state.selected_zone is not None and selected_label == "— Select a zone —":
            st.session_state.selected_zone = None
            st.rerun()

    st.divider()
    top_n = st.slider("Top N zones to highlight", 5, 30, 10)
    auto_refresh = st.checkbox("Auto-refresh (60s)", value=False)
    if st.button("🔄 Refresh now"):
        st.cache_data.clear()
        st.rerun()

    st.divider()
    ec2_host = os.getenv("EC2_HOST", "100.49.225.43")
    st.markdown("**External links**")
    st.markdown(f"[Grafana](http://{ec2_host}:3000)  ·  [MLflow](http://{ec2_host}:5000)  ·  [Airflow](http://{ec2_host}:8081)")

if auto_refresh:
    import time
    time.sleep(60)
    st.cache_data.clear()
    st.rerun()

# ════════════════════════════════════════════════════════════
# MAIN CONTENT
# ════════════════════════════════════════════════════════════

predictions_df = fetch_predictions()
monitoring = fetch_monitoring_metrics()

predicted_hour_str = ""
if not predictions_df.empty and "predicted_hour" in predictions_df.columns:
    try:
        ph = pd.to_datetime(predictions_df["predicted_hour"].iloc[0])
        predicted_hour_str = ph.strftime("%Y-%m-%d %H:00 UTC")
    except Exception:
        predicted_hour_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:00 UTC")
else:
    predicted_hour_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:00 UTC")

st.title("🚕 NYC Taxi Demand — Live Forecast")
st.caption(f"Predictions for: **{predicted_hour_str}**")

if predictions_df.empty:
    st.warning("No predictions available. Make sure the API is running and the inference pipeline has run.")
    st.stop()

# Enrich predictions with zone names
if not zone_lookup.empty:
    predictions_df = predictions_df.merge(
        zone_lookup[["location_id", "zone_name", "borough"]],
        left_on="pickup_location_id", right_on="location_id", how="left",
    )
    predictions_df["zone_name"] = predictions_df["zone_name"].fillna(
        predictions_df["pickup_location_id"].astype(str)
    )
    predictions_df["borough"] = predictions_df["borough"].fillna("Unknown")
else:
    predictions_df["zone_name"] = predictions_df["pickup_location_id"].astype(str)
    predictions_df["borough"] = "Unknown"

# ── KPI row ──────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Total Predicted Rides", f"{predictions_df['predicted_rides'].sum():,.0f}")
with k2:
    mae_val = monitoring.get("model_mae", {}).get("value", "N/A")
    st.metric("Model MAE", f"{mae_val:.1f}" if isinstance(mae_val, float) else mae_val)
with k3:
    drift_val = monitoring.get("data_drift_score", {}).get("value", "N/A")
    drift_lbl = f"{drift_val:.3f}" if isinstance(drift_val, float) else drift_val
    drift_delta = "⚠️ Drifting" if isinstance(drift_val, float) and drift_val > 0.25 else "✅ Stable"
    st.metric("Data Drift Score", drift_lbl, delta=drift_delta)
with k4:
    latency = predictions_df["meta_latency_ms"].iloc[0] if not predictions_df.empty else "N/A"
    st.metric("Prediction Latency", f"{latency:.0f} ms" if isinstance(latency, float) else latency)

st.divider()

# ── Choropleth Map ────────────────────────────────────────────
st.subheader("🗺️ Demand Heatmap — Click a zone to explore")

gdf = load_zone_geodata()

if gdf is not None:
    map_col, detail_col = st.columns([2, 1])

    with map_col:
        folium_map = build_choropleth(
            predictions_df, zone_lookup, gdf, st.session_state.selected_zone
        )
        map_output = st_folium(
            folium_map,
            width="100%",
            height=520,
            returned_objects=[],
            key="main_map",
        )

        # Detect map click → find zone by point-in-polygon
        if map_output and map_output.get("last_clicked"):
            lat = map_output["last_clicked"]["lat"]
            lng = map_output["last_clicked"]["lng"]
            click_pt = Point(lng, lat)
            clicked_id = None
            for _, row in gdf.iterrows():
                if row.geometry and row.geometry.contains(click_pt):
                    clicked_id = int(row["LocationID"])
                    break
            if clicked_id and clicked_id != st.session_state.selected_zone:
                st.session_state.selected_zone = clicked_id
                st.rerun()

    with detail_col:
        st.markdown("**Zone Details**")
        if st.session_state.selected_zone:
            z_id = st.session_state.selected_zone
            z_row = zone_lookup[zone_lookup["location_id"] == z_id]
            p_row = predictions_df[predictions_df["pickup_location_id"] == z_id]

            if not z_row.empty:
                z = z_row.iloc[0]
                st.markdown(f"#### 📍 {z['zone_name']}")
                st.markdown(f"**Borough:** {z['borough']}")
                st.markdown(f"**Zone ID:** {z_id}")
                st.markdown(f"**Service Zone:** {z.get('service_zone', 'N/A')}")
                st.divider()

            if not p_row.empty:
                rides = float(p_row["predicted_rides"].iloc[0])
                city_avg = predictions_df["predicted_rides"].mean()
                pct = ((rides - city_avg) / city_avg * 100) if city_avg else 0
                rank = int((predictions_df["predicted_rides"] >= rides).sum())

                st.metric("Predicted Rides", f"{rides:,.0f}", delta=f"{pct:+.0f}% vs avg")
                st.metric("City Rank", f"#{rank} of {len(predictions_df)}")
                st.metric("City Avg", f"{city_avg:,.1f}")

                # Borough comparison
                borough_avg = predictions_df[
                    predictions_df["borough"] == z.get("borough", "Unknown")
                ]["predicted_rides"].mean()
                if borough_avg:
                    st.metric(f"{z.get('borough','Borough')} Avg", f"{borough_avg:,.1f}")
            else:
                st.info("No prediction for this zone.")

            st.divider()
            if st.button("✕ Clear selection", use_container_width=True):
                st.session_state.selected_zone = None
                st.rerun()
        else:
            st.info("Click any zone on the map or use the sidebar dropdown to explore zone details.")
            st.markdown("---")
            st.markdown("**Top 3 zones right now:**")
            top3 = predictions_df.nlargest(3, "predicted_rides")[
                ["zone_name", "predicted_rides"]
            ]
            for _, row in top3.iterrows():
                st.markdown(f"- **{row['zone_name']}** — {row['predicted_rides']:,.0f} rides")

else:
    st.warning("Zone shapefile could not be loaded. Showing table view instead.")

st.divider()

# ── Zone selected → expanded detail row ──────────────────────
if st.session_state.selected_zone and gdf is not None:
    render_zone_details(st.session_state.selected_zone, zone_lookup, predictions_df)
    st.divider()

# ── Top N Zones (with zone names) ────────────────────────────
top_zones = predictions_df.nlargest(top_n, "predicted_rides")
st.subheader(f"Top {top_n} Zones by Predicted Demand")

chart_col, table_col = st.columns([2, 1])

with chart_col:
    fig = px.bar(
        top_zones.sort_values("predicted_rides"),
        x="predicted_rides",
        y="zone_name",
        orientation="h",
        color="predicted_rides",
        color_continuous_scale="Oranges",
        labels={"predicted_rides": "Predicted Rides", "zone_name": "Zone"},
        hover_data={"borough": True, "pickup_location_id": True},
    )
    fig.update_layout(height=420, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with table_col:
    st.markdown("**Rankings**")
    tbl = top_zones[["zone_name", "borough", "predicted_rides"]].copy()
    tbl.columns = ["Zone", "Borough", "Rides"]
    tbl["Rides"] = tbl["Rides"].round(1)
    tbl = tbl.sort_values("Rides", ascending=False).reset_index(drop=True)
    tbl.index += 1
    st.dataframe(tbl, use_container_width=True, height=400)

st.divider()

# ── Borough breakdown ─────────────────────────────────────────
st.subheader("Demand by Borough")
borough_df = (
    predictions_df.groupby("borough")["predicted_rides"]
    .agg(["sum", "mean", "count"])
    .reset_index()
)
borough_df.columns = ["Borough", "Total Rides", "Avg per Zone", "Zones"]
borough_df = borough_df.sort_values("Total Rides", ascending=False)

b1, b2 = st.columns(2)
with b1:
    fig_b = px.pie(
        borough_df,
        names="Borough",
        values="Total Rides",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hole=0.4,
    )
    fig_b.update_layout(height=300, margin=dict(t=20, b=20))
    st.plotly_chart(fig_b, use_container_width=True)

with b2:
    st.dataframe(
        borough_df.set_index("Borough").style.format(
            {"Total Rides": "{:,.0f}", "Avg per Zone": "{:,.1f}"}
        ),
        use_container_width=True,
        height=300,
    )

st.divider()

# ── Distribution ──────────────────────────────────────────────
st.subheader("Demand Distribution Across All Zones")
fig_hist = px.histogram(
    predictions_df,
    x="predicted_rides",
    nbins=40,
    labels={"predicted_rides": "Predicted Rides"},
    color_discrete_sequence=["#fd8d3c"],
)
fig_hist.update_layout(height=280)
st.plotly_chart(fig_hist, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data: NYC TLC  ·  Model: LightGBM via MLflow  ·  "
    "Features: Feast + Redis  ·  Monitoring: Evidently AI + Grafana"
)
