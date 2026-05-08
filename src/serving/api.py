"""
FastAPI prediction service.
Exposes /predict, /health, /metrics (Prometheus), and /monitoring endpoints.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from src.config import settings
from src.utils.db import health_check as db_health_check
from src.utils.logging_utils import logger

# ── Prometheus custom metrics ────────────────────────────────
PREDICTIONS_TOTAL = Counter(
    "taxi_demand_predictions_total",
    "Total number of demand predictions served",
    ["location_id"],
)
MODEL_MAE = Gauge("taxi_demand_model_mae", "Current production model MAE")
DATA_DRIFT_SCORE = Gauge(
    "taxi_demand_data_drift_score", "Latest data drift score (0–1)"
)
PREDICTION_LATENCY = Histogram(
    "taxi_demand_prediction_latency_seconds",
    "Time to generate predictions batch",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
)

# ── App state ────────────────────────────────────────────────
_model = None
_model_loaded_at: str = ""


def _load_model():
    """Load production model from MLflow registry at startup."""
    global _model, _model_loaded_at
    try:
        from src.training.registry import load_production_model

        _model = load_production_model()
        _model_loaded_at = datetime.now(tz=timezone.utc).isoformat()
        logger.info("Production model loaded successfully")
    except Exception as e:
        logger.warning(
            f"Could not load model at startup (will retry on first request): {e}"
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


# ── FastAPI app ───────────────────────────────────────────────
app = FastAPI(
    title="NYC Taxi Demand API",
    description="Real-time hourly taxi demand predictions per NYC zone",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Auto-instrument all routes with Prometheus metrics
Instrumentator().instrument(app).expose(app)


# ── Request / Response schemas ────────────────────────────────


class PredictionRequest(BaseModel):
    location_ids: list[int] = Field(
        default_factory=list,
        description="List of NYC taxi zone IDs to predict. Empty = all zones.",
        example=[1, 132, 161],
    )
    target_hour: str = Field(
        default="",
        description="ISO8601 datetime for prediction target. Defaults to next full hour.",
        example="2024-03-15T14:00:00Z",
    )


class ZonePrediction(BaseModel):
    pickup_location_id: int
    predicted_rides: float
    predicted_hour: str


class PredictionResponse(BaseModel):
    predictions: list[ZonePrediction]
    model_version: str
    predicted_at: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    database: str
    model_loaded: bool
    model_loaded_at: str
    environment: str


# ── Endpoints ────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    db_ok = db_health_check()
    return HealthResponse(
        status="healthy" if db_ok and _model is not None else "degraded",
        database="ok" if db_ok else "unavailable",
        model_loaded=_model is not None,
        model_loaded_at=_model_loaded_at,
        environment=settings.environment,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
async def predict(request: PredictionRequest):
    global _model

    # Lazy-load model if startup failed
    if _model is None:
        _load_model()
        if _model is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Check MLflow registry.",
            )

    # Default: all 263 NYC zones
    location_ids = request.location_ids or list(range(1, 264))

    t0 = time.perf_counter()

    try:
        # Fetch latest feature window from PostgreSQL / Redis
        X = _build_feature_df(location_ids)
        # Drop non-numeric columns the model was not trained on
        feature_cols = [c for c in X.columns if c != "pickup_hour"]
        raw_preds = _model.predict(X[feature_cols]).clip(0)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000
    PREDICTION_LATENCY.observe(latency_ms / 1000)

    target_hour = _resolve_target_hour(request.target_hour)

    predictions = []
    for loc_id, pred in zip(location_ids, raw_preds):
        predictions.append(
            ZonePrediction(
                pickup_location_id=loc_id,
                predicted_rides=round(float(pred), 2),
                predicted_hour=target_hour,
            )
        )
        PREDICTIONS_TOTAL.labels(location_id=str(loc_id)).inc()

    return PredictionResponse(
        predictions=predictions,
        model_version=_model_loaded_at,
        predicted_at=datetime.now(tz=timezone.utc).isoformat(),
        latency_ms=round(latency_ms, 2),
    )


@app.get("/monitoring/metrics", tags=["monitoring"])
async def monitoring_metrics() -> dict[str, Any]:
    """Return the latest monitoring metrics from the database."""
    try:
        from src.utils.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT DISTINCT ON (metric_name)
                        metric_name, metric_value, recorded_at
                    FROM model_monitoring
                    ORDER BY metric_name, recorded_at DESC
                """)
            ).fetchall()
        return {r[0]: {"value": r[1], "recorded_at": str(r[2])} for r in rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Helpers ───────────────────────────────────────────────────


def _build_feature_df(location_ids: list[int]) -> pd.DataFrame:
    """
    Build the inference feature DataFrame.
    Reads the most recent window_hours of ride counts from PostgreSQL.
    In production this reads from the Redis online store via Feast.
    """
    from sqlalchemy import text
    from src.utils.db import get_session

    window = settings.feature_window_hours
    records = []
    with get_session() as session:
        for loc_id in location_ids:
            rows = session.execute(
                text("""
                    SELECT ride_count
                    FROM hourly_rides
                    WHERE pickup_location_id = :loc_id
                    ORDER BY pickup_hour DESC
                    LIMIT :window
                """),
                {"loc_id": loc_id, "window": window},
            ).fetchall()
            if len(rows) < window:
                # Pad with zeros for locations with insufficient history
                counts = [0] * (window - len(rows)) + [r[0] for r in reversed(rows)]
            else:
                counts = [r[0] for r in reversed(rows)]

            lags = {f"lag_{j}": counts[-(j)] for j in range(1, window + 1)}
            lags["pickup_location_id"] = loc_id
            lags["pickup_hour"] = datetime.now(tz=timezone.utc).replace(
                minute=0, second=0, microsecond=0
            )
            records.append(lags)

    return pd.DataFrame(records)


def _resolve_target_hour(target_hour_str: str) -> str:
    if target_hour_str:
        return target_hour_str
    now = datetime.now(tz=timezone.utc)
    next_hour = now.replace(minute=0, second=0, microsecond=0)
    return next_hour.isoformat()
