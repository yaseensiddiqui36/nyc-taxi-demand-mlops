-- ─────────────────────────────────────────────────────────────
-- NYC Taxi Demand MLOps — Database Initialization
-- Runs once on first PostgreSQL container start
-- ─────────────────────────────────────────────────────────────

-- Create additional databases (taxi_demand already created via POSTGRES_DB)
CREATE DATABASE airflow;
CREATE DATABASE mlflow;

-- ── taxi_demand database setup ────────────────────────────────
\c taxi_demand

-- Enable TimescaleDB
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- Enable uuid
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Raw hourly rides (time-series) ───────────────────────────
CREATE TABLE IF NOT EXISTS hourly_rides (
    pickup_location_id  SMALLINT     NOT NULL,
    pickup_hour         TIMESTAMPTZ  NOT NULL,
    ride_count          INTEGER      NOT NULL DEFAULT 0,
    ingested_at         TIMESTAMPTZ  DEFAULT NOW(),
    PRIMARY KEY (pickup_location_id, pickup_hour)
);
SELECT create_hypertable('hourly_rides', 'pickup_hour', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_hourly_rides_loc_hour
    ON hourly_rides (pickup_location_id, pickup_hour DESC);

-- ── Model predictions ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS predictions (
    id                  UUID         DEFAULT uuid_generate_v4() PRIMARY KEY,
    pickup_location_id  SMALLINT     NOT NULL,
    predicted_hour      TIMESTAMPTZ  NOT NULL,
    predicted_rides     FLOAT        NOT NULL,
    model_name          VARCHAR(100),
    model_version       VARCHAR(50),
    predicted_at        TIMESTAMPTZ  DEFAULT NOW()
);
SELECT create_hypertable('predictions', 'predicted_hour', if_not_exists => TRUE);
CREATE UNIQUE INDEX IF NOT EXISTS predictions_location_hour_unique
    ON predictions (pickup_location_id, predicted_hour);
CREATE INDEX IF NOT EXISTS idx_predictions_loc_hour
    ON predictions (pickup_location_id, predicted_hour DESC);

-- ── Model monitoring (MAE, drift scores, etc.) ───────────────
CREATE TABLE IF NOT EXISTS model_monitoring (
    id           BIGSERIAL    PRIMARY KEY,
    metric_name  VARCHAR(100) NOT NULL,
    metric_value FLOAT        NOT NULL,
    model_name   VARCHAR(100),
    model_version VARCHAR(50),
    window_start TIMESTAMPTZ,
    window_end   TIMESTAMPTZ,
    recorded_at  TIMESTAMPTZ  DEFAULT NOW(),
    metadata     JSONB
);
CREATE INDEX IF NOT EXISTS idx_monitoring_metric_time
    ON model_monitoring (metric_name, recorded_at DESC);

-- ── Data pipeline runs (audit log) ───────────────────────────
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id             BIGSERIAL    PRIMARY KEY,
    pipeline_name  VARCHAR(100) NOT NULL,
    status         VARCHAR(20)  NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    started_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    finished_at    TIMESTAMPTZ,
    rows_processed INTEGER,
    error_message  TEXT,
    metadata       JSONB
);

-- ── Feast schema (offline store) ─────────────────────────────
CREATE SCHEMA IF NOT EXISTS feast;
