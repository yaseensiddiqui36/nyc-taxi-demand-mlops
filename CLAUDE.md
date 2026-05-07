# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

Production-grade MLOps system for real-time NYC taxi demand forecasting. Predicts hourly ride demand per NYC zone using LightGBM. Full automation via Apache Airflow, feature store via Feast + Redis, experiment tracking via MLflow (hosted on DagsHub), drift detection via Evidently AI, monitoring via Prometheus + Grafana, served by FastAPI + Streamlit. Deployed on AWS Lightsail with GitHub Actions CI/CD.

---

## Common Commands

```bash
# Start always-on services (API, frontend, DB, monitoring)
make up

# Start full local dev stack (+ Airflow + MLflow)
make dev-up

# Stop everything
make dev-down

# Trigger pipelines manually (requires dev-up first)
make data-pipeline    # fetch latest TLC data → feature store
make train            # run weekly training experiment
make infer            # generate hourly predictions

# Feast
make feast-apply           # register feature definitions
make feast-materialize     # sync offline → Redis online store

# Code quality
make test             # pytest all tests
make lint             # ruff check
make format           # ruff format + autofix

# Deploy to AWS
make deploy           # rsync + docker compose pull + up on AWS host
```

---

## Service URLs (local dev)

| Service     | URL                         | Credentials           |
|-------------|-----------------------------|-----------------------|
| Airflow     | http://localhost:8081       | admin / see .env      |
| MLflow      | http://localhost:5000       | —                     |
| FastAPI     | http://localhost:8000/docs  | —                     |
| Streamlit   | http://localhost:8501       | —                     |
| Grafana     | http://localhost:3000       | admin / see .env      |
| Prometheus  | http://localhost:9090       | —                     |
| PostgreSQL  | localhost:5433              | see .env              |
| Redis       | localhost:6379              | —                     |

---

## Architecture

```
NYC TLC Data (AWS CloudFront, monthly parquet)
  ↓
Airflow DAG: data_pipeline_dag (daily 06:00 UTC)
  → validate_and_clean  (src/data/validation.py)
  → to_hourly_time_series  (src/data/transformation.py)
  → PostgreSQL + TimescaleDB  (hourly_rides table)
  → Feast offline store (PostgreSQL feast schema)
  → Feast online store (Redis)
  ↓
Airflow DAG: training_pipeline_dag (every Monday 02:00 UTC)
  → builds 180-day sliding window features
  → trains baseline, LightGBM, XGBoost in parallel
  → selects best by CV MAE
  → registers to MLflow Model Registry (only if MAE improves)
  ↓
Airflow DAG: inference_pipeline_dag (every hour :05)
  → builds 28-day feature windows from PostgreSQL
  → loads Production model from MLflow registry
  → stores predictions → predictions table
  → runs model drift check vs MLflow Production MAE
  → Slack alert if degradation > 20%
  ↓
FastAPI (src/serving/api.py)  →  reads predictions + model
Streamlit (src/frontend/app.py)  →  calls FastAPI
Evidently AI (src/monitoring/drift.py)  →  data + model drift HTML reports
Prometheus + Grafana  →  scrape /metrics, dashboards
```

---

## Key Source Modules

| Module                          | Purpose                                                                         |
|---------------------------------|---------------------------------------------------------------------------------|
| `src/config.py`                 | All settings via Pydantic BaseSettings, loaded from .env                        |
| `src/data/ingestion.py`         | Fetch parquet from NYC TLC; detect latest available month                       |
| `src/data/validation.py`        | 7-rule quality checks; returns ValidationReport                                 |
| `src/data/transformation.py`    | Raw trips → hourly time-series; sliding-window features                         |
| `src/features/engineering.py`   | Temporal features, rolling stats, US holiday flags, cyclical encoding           |
| `src/features/feature_store.py` | Feast read/write helpers (offline + online)                                     |
| `src/training/models.py`        | All model pipelines (baseline, LightGBM, XGBoost) in sklearn Pipeline          |
| `src/training/train.py`         | Time-series CV training loop, MLflow logging, Optuna tuning                    |
| `src/training/registry.py`      | MLflow Model Registry: promote to Production only if MAE improves              |
| `src/monitoring/drift.py`       | Evidently AI data drift + model performance reports                             |
| `src/monitoring/alerts.py`      | Slack webhook alerts for drift, degradation, failures                           |
| `src/serving/api.py`            | FastAPI: /predict, /health, /metrics, /monitoring/metrics                       |
| `src/frontend/app.py`           | Streamlit: live demand map, KPI cards, monitoring badges                        |
| `src/utils/db.py`               | SQLAlchemy + psycopg2 helpers; raw connection for COPY                          |
| `src/utils/logging_utils.py`    | Loguru: colored dev logs, JSON prod logs, file rotation                         |

---

## Docker Compose Profiles

- **No profile** (`make up`): postgres, redis, api, frontend, prometheus, grafana — always-on services
- **`--profile local`** (`make dev-up`): adds airflow-init, airflow-webserver, airflow-scheduler, mlflow — on-demand only

The always-on profile is what runs on AWS. Airflow + MLflow run locally.

---

## Database Schema (PostgreSQL + TimescaleDB)

- `hourly_rides` — hypertable, partitioned by `pickup_hour`. Primary key: `(pickup_location_id, pickup_hour)`
- `predictions` — hypertable, partitioned by `predicted_hour`
- `model_monitoring` — scalar metrics (MAE, drift scores) with timestamps
- `pipeline_runs` — audit log of pipeline executions
- `feast.*` — Feast offline store schema (managed by `feast apply`)

---

## MLflow Model Registry

- Experiment: `taxi_demand_forecasting`
- Model name: `taxi_demand_predictor` (from `settings.model_name`)
- Tracking URI: DagsHub (`https://dagshub.com/yaseensiddiqui36/nyc-taxi-demand-mlops.mlflow`)
- Stages: Staging → Production → Archived
- New models only promoted if CV MAE < current Production MAE

---

## Environment Setup

```bash
# 1. Copy and fill in secrets
cp .env.example .env

# 2. Generate Airflow Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# Paste into AIRFLOW_FERNET_KEY in .env

# 3. Generate Airflow secret key
python -c "import secrets; print(secrets.token_hex(32))"
# Paste into AIRFLOW_SECRET_KEY in .env

# 4. Start services
make dev-up

# 5. Apply Feast feature definitions
make feast-apply
```

---

## AWS Deployment (always-on tier)

Target: AWS Lightsail $10/month plan (2 vCPU, 2GB RAM) in `us-east-1`.

Required GitHub Secrets:
- `AWS_HOST` — public IP or hostname of Lightsail instance
- `AWS_SSH_PRIVATE_KEY` — SSH private key for ubuntu user
- `POSTGRES_PASSWORD`, `SLACK_WEBHOOK_URL`, `GRAFANA_ADMIN_PASSWORD`
- `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `S3_BUCKET_NAME`
- `MLFLOW_TRACKING_URI` (DagsHub URI for the always-on stack)

---

## Google Colab + MLflow (GPU training)

```python
import os
os.environ["MLFLOW_TRACKING_URI"]      = "https://dagshub.com/yaseensiddiqui36/nyc-taxi-demand-mlops.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "yaseensiddiqui36"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<DAGSHUB_TOKEN>"
# Then run any src/training/train.py experiment normally
```

---

## Model Features

- **Lag features**: lag_1 … lag_672 (28 days × 24 hours of ride counts)
- **Temporal**: hour_of_day, day_of_week, month, is_weekend, is_holiday (cyclically encoded)
- **Rolling**: avg_rides_last_{1,2,3,4}w (same-hour average from prior weeks)
- **Global model**: one model for all 263 zones; `pickup_location_id` is a feature
- **Target**: next hour's ride count for that zone

---

# ══════════════════════════════════════════════════════════════
# PROJECT STATUS, ERRORS, AND DECISIONS (Added 2026-05-07)
# ══════════════════════════════════════════════════════════════

## Current Development Stage

### ✅ Completed (Phases 1–4)
1. **Data Ingestion & Validation** — `src/data/ingestion.py`, `validation.py`, `transformation.py` are fully implemented. Backfill DAG working. Data from NYC TLC (May 2025 → Feb 2026) is loading successfully.
2. **Database Layer** — PostgreSQL + TimescaleDB is configured. `hourly_rides`, `predictions`, `model_monitoring`, `pipeline_runs` tables in place. Feast offline schema via PostgreSQL (`feast.*`).
3. **Feature Store** — Feast configured with PostgreSQL offline store + Redis online store. `feast apply` registers the `hourly_rides_view` feature view. `materialize_to_online_store` syncs offline → Redis.
4. **Training Infrastructure** — `src/training/train.py`, `models.py`, `registry.py` are complete. Time-series CV with 3 folds, parallel training of Baseline/LightGBM/XGBoost, MLflow experiment logging to DagsHub. Conditional model promotion (only promotes if MAE improves).
5. **Airflow DAGs** — All 4 DAGs scaffolded and partially working: `data_pipeline_dag`, `training_pipeline_dag`, `inference_pipeline_dag`, `backfill_dag`.
6. **Monitoring** — Evidently AI drift reports (`drift.py`), Slack alerts (`alerts.py`), Prometheus + Grafana stack running.
7. **Serving** — FastAPI + Streamlit containers defined in docker-compose.

### 🔄 In Progress / Partially Complete (Phase 5)
- **Drift check in `data_pipeline_dag`** (`_run_drift_check` task) is a **placeholder** — just prints a message. Real drift detection calling `run_data_drift_report()` needs to be wired in.
- **Training DAG** runs locally but the `build_training_dataset` step requires 180-day feature matrix construction (672 lag features × ~180k rows per zone) which OOMs on 8GB RAM locally. Currently solved by running training on Google Colab.
- **Inference DAG** — `_check_model_drift` references `_record_metric` from `src/monitoring/drift.py` but imports it with underscore-prefixed private name. This works but is fragile.
- **Feast registry** — uses SQLite file (`feature_repo/data/registry.db`). This breaks in multi-container Docker because each container gets its own copy of the file. Migration to PostgreSQL registry is pending.
- **`alert_new_model_registered`** in `training_pipeline_dag.py` passes `"latest"` as version string instead of the actual version number from MLflow. This is cosmetic but should be fixed.

### ❌ Not Yet Started (Phase 6+)
- **AWS Lightsail deployment** — GitHub Actions CI/CD workflow not yet created/triggered.
- **Grafana dashboards** — `grafana/provisioning` directory exists but dashboards are placeholder config.
- **Optuna hyperparameter tuning** — code exists in `train.py` but not wired into any DAG task.
- **Full end-to-end live inference** — requires a Production model in MLflow registry first.
- **`predictions` table ON CONFLICT clause** — `_store_predictions` in the inference DAG does plain INSERTs with no upsert; duplicate runs will cause constraint errors.

---

## DAG Errors Analysis (from logs/app_2026-05-06.log)

### What the logs show
The log file (`logs/app_2026-05-06.log`) contains **only INFO-level messages** — no ERROR or WARNING entries. All 41 log entries are from two modules:
- `src.data.validation:validate_and_clean` — reporting drop rates per month
- `src.data.transformation:to_hourly_time_series` — reporting row/location/hour counts

These are from the **backfill run** processing months 2025-05 through 2026-02. The data is processing correctly.

### Drop rate analysis (normal, expected)
| Month     | In rows    | Out rows   | Drop rate |
|-----------|------------|------------|-----------|
| 2025-05   | 4,591,845  | 3,163,446  | 31.1%     |
| 2025-06   | 4,322,960  | 2,882,349  | 33.3%     |
| 2025-07   | 3,898,963  | 2,646,923  | 32.1%     |
| 2025-08   | 3,574,091  | 2,483,325  | 30.5%     |
| 2025-09   | 4,251,015  | 2,961,723  | 30.3%     |
| 2025-10   | 4,428,699  | 3,211,888  | 27.5%     |
| 2025-11   | 4,181,444  | 2,975,592  | 28.8%     |
| 2025-12   | 4,305,006  | 2,920,083  | 32.2%     |
| 2026-01   | 3,724,889  | 2,486,175  | 33.3%     |
| 2026-02   | 3,399,866  | 2,251,842  | 33.8%     |

The ~30% drop rate is **normal for NYC TLC data** — the dominant issue is `null_key_columns` (trips where pickup/dropoff location IDs are null), which is a known data quality characteristic of the TLC dataset. This is not a bug.

### Observed/Known DAG Errors (from conversation history)

These are errors that have been encountered in previous sessions (not in today's log, which ran clean):

#### 1. `materialize_feast_online` task — Feast SQLite registry multi-container conflict
- **Error**: `OperationalError: database is locked` or `registry.db not found` inside Airflow container
- **Root cause**: `feature_repo/data/registry.db` is a SQLite file. When `feast apply` runs on the host and Airflow runs in Docker, the containers don't share the registry state reliably.
- **Current workaround**: Mount the `feature_repo` directory as a Docker volume (already in docker-compose.yml).
- **Proper fix**: Migrate Feast registry to PostgreSQL (see Planned Changes section).

#### 2. `run_drift_check` task — placeholder, always logs but does nothing
- **Error type**: Not a crash — it's a logic gap. The task runs a `print()` statement and exits successfully without performing any actual drift analysis.
- **Root cause**: Implementation deferred to Phase 6. No reference dataset has been defined yet.
- **Fix required**: Wire `run_data_drift_report()` from `src/monitoring/drift.py` with a reference snapshot vs current batch.

#### 3. `build_training_dataset` / training tasks — OOM on local 8GB RAM
- **Error**: Python process killed (`MemoryError` or OS kill) during `to_features_and_target()`.
- **Root cause**: Feature matrix for 180 days × 263 zones × 672 lag columns is ~180k rows × 672 columns = ~120M float values ≈ ~1GB RAM for the matrix alone, plus intermediate DataFrames can spike to 4–6GB total.
- **Current workaround**: Train on Google Colab (free GPU tier). See Training Compute section below.

#### 4. `_record_metric` called as private import in inference DAG
- **Location**: `airflow/dags/inference_pipeline_dag.py` line 170
- **Code**: `from src.monitoring.drift import _record_metric`
- **Issue**: Importing underscore-prefixed (private) function directly. Works but violates encapsulation. If drift.py is refactored, this silently breaks.
- **Fix**: Expose a public `record_monitoring_metric()` wrapper in `drift.py`.

#### 5. `_store_predictions` — no upsert / ON CONFLICT handling
- **Location**: `airflow/dags/inference_pipeline_dag.py` lines 124–132
- **Issue**: Plain `INSERT` without `ON CONFLICT DO UPDATE`. If the inference DAG is re-triggered or retried for the same hour, it will fail with a unique constraint violation on `(pickup_location_id, predicted_hour)`.
- **Fix**: Add `ON CONFLICT (pickup_location_id, predicted_hour) DO UPDATE SET predicted_rides = EXCLUDED.predicted_rides`.

#### 6. `alert_new_model_registered` — hardcoded `"latest"` version string
- **Location**: `airflow/dags/training_pipeline_dag.py` line 139
- **Issue**: `alert_new_model_registered(best_name, "latest", best_mae)` — version is always `"latest"` instead of the actual MLflow model version number.
- **Fix**: Pull version from the `promoted` result or from the `register_model_if_better()` return value.

#### 7. `session.execute()` with raw string SQL in `drift.py`
- **Location**: `src/monitoring/drift.py` line 117
- **Issue**: `session.execute("""INSERT INTO model_monitoring...""", {...})` — passing a raw string instead of `sqlalchemy.text()`. In SQLAlchemy 2.x, this raises `RemovedIn20Warning` and will break in future versions.
- **Fix**: Wrap with `text()`: `session.execute(text("INSERT INTO..."), {...})`.

---

## Training Compute Strategy

### Current State
Training is currently done on **Google Colab** (free tier with GPU) because the local machine has only **8GB RAM**, which is insufficient for building the full 672-feature lag matrix across 263 zones × 6+ months of history.

### Colab Workflow
```python
# In Colab, set DagsHub MLflow tracking:
import os
os.environ["MLFLOW_TRACKING_URI"]      = "https://dagshub.com/yaseensiddiqui36/nyc-taxi-demand-mlops.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "yaseensiddiqui36"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "<DAGSHUB_TOKEN>"
# Run training notebook / train.py
# Model is logged to DagsHub MLflow
# Promote to Production via MLflow UI or registry.py
```

### Going Live — Training Compute Options

Once the project goes live, training must be automated (no manual Colab). Options ranked by cost-effectiveness:

#### Option A: AWS EC2 Spot Instance (Recommended for this project)
- **Instance**: `r5.large` (2 vCPU, 16GB RAM) Spot → ~$0.05/hr
- **Trigger**: Airflow `training_pipeline_dag` provisions an EC2 Spot via `boto3`, runs training, terminates instance
- **Cost**: Training takes ~30–60 min weekly → ~$0.05–0.10/week → **<$6/year**
- **Artifact storage**: MLflow logs to DagsHub (already configured)
- **Implementation**: Add a `provision_and_train` task that calls `boto3.client('ec2').run_instances()` with a user-data bootstrap script, waits for completion via SSM or S3 sentinel file
- **Risk**: Spot interruption. Use `r5.large` Spot with On-Demand fallback or use SageMaker Training Jobs

#### Option B: AWS SageMaker Training Jobs
- **Cost**: `ml.m5.xlarge` (4 vCPU, 16GB) → ~$0.23/hr → ~$0.12–0.23/run
- **Advantage**: Fully managed, no Spot interruption risk, native MLflow integration
- **Disadvantage**: More expensive, requires containerizing the training script
- **Good if**: project scales to multiple model experiments or needs GPU

#### Option C: Modal Labs (Serverless GPU)
- **Cost**: ~$0.0002/GPU-second, free tier 30 GPU-hours/month
- **Advantage**: Zero infra management, scales to 0 when not training
- **Disadvantage**: Requires rewriting training entry point as a Modal function

#### Option D: Keep Colab (Current, not scalable)
- **Advantage**: Free, no infra changes
- **Disadvantage**: Manual, not automatable for production, session timeouts
- **Use case**: Only acceptable for initial model training / experimentation phase

#### Option E: Increase Local RAM
- Upgrade local dev machine to 32GB RAM → $60–100 cost
- Simplest change but doesn't solve the cloud deployment scenario

### Recommended Path
1. **Short-term**: Continue Colab for initial model training until a Production model exists in MLflow.
2. **Medium-term (going live)**: Implement **EC2 Spot training** triggered by `training_pipeline_dag`. Add a `launch_spot_training` task using `boto3` before the `build_training_dataset` task, or replace the in-DAG training with an EC2 runner.
3. **If scale increases**: Migrate to SageMaker Training Jobs.

---

## What Must Change If Structure Changes

### If you change the PostgreSQL schema
- Update `scripts/init_db.sql`
- Update `src/utils/db.py` (any raw SQL)
- Update all DAG tasks that reference table/column names
- Update Feast `data_sources.py` if the source table/columns change
- Re-run `feast apply` after `data_sources.py` changes

### If you change feature engineering (lag windows, new features)
- Update `to_features_and_target()` in `src/data/transformation.py`
- Update `to_inference_features()` in the same file
- Update `settings.feature_window_hours` in `src/config.py` if window changes
- **RETRAIN the model** — feature set must match between training and inference
- Update `src/features/engineering.py` if adding temporal/rolling features
- The model stored in MLflow must be re-registered with the new feature set

### If you change the Feast feature view or entities
- Edit `feature_repo/features.py` and `feature_repo/data_sources.py`
- Run `feast apply` inside the Airflow container or on the host with the correct env
- Feast registry must be accessible from all containers (see SQLite migration note)

### If you change MLflow tracking URI
- Update `MLFLOW_TRACKING_URI` in `.env`
- Update `docker-compose.yml` (passed to all containers)
- Update `src/config.py` default (if relevant)
- Update Colab notebook instructions in this file

### If you add a new DAG
- Place in `airflow/dags/`
- All `sys.path.insert(0, "/opt/airflow")` calls are needed for Docker import resolution
- Add a Makefile target if it should be triggerable via `make`

### If you change Docker service names (postgres, redis, mlflow...)
- Update `docker-compose.yml` service names
- Update all `POSTGRES_HOST=postgres`, `REDIS_URL=redis://redis:...` references in `.env`, `docker-compose.yml`, and `src/config.py` defaults
- Update `feature_repo/feature_store.yaml` environment variable references

### If you change the model algorithm or sklearn Pipeline structure
- Update `src/training/models.py` (`MODEL_REGISTRY`)
- The trained pipeline must remain `sklearn`-compatible (uses `mlflow.sklearn.log_model`)
- Update inference DAG's `feature_cols` extraction logic if column names change
- Consider versioning the model schema separately

---

## Change Log

### 2026-05-06 — Backfill Pipeline Run
- Successfully ran `backfill_dag` loading NYC TLC data for 2025-05 through 2026-02 (10 months).
- Average drop rate ~31% — normal for TLC data (null location IDs are the dominant cause).
- No errors in data pipeline execution. All INFO logs clean.

### 2026-05-07 — Architecture Analysis & Documentation Update (this session)

**Reason**: User requested a comprehensive understanding of the project state, DAG errors, training compute decisions, and documentation of what must change if structure changes.

**Findings**:
1. The `app_2026-05-06.log` contains only clean INFO messages from the backfill run — no runtime errors. All previously reported DAG errors are from earlier sessions (Hopsworks integration, datetime parsing, Evidently `save_html` crashes — all resolved in prior sessions).
2. Seven known code issues identified in the current DAG codebase (documented above in "Known DAG Errors").
3. Training compute bottleneck confirmed: 672-feature lag matrix across 263 zones requires >8GB RAM.
4. Feast SQLite registry is a known multi-container liability.

**No code was changed in this session** — documentation only.

**Pending decisions**:
- [ ] Migrate Feast registry from SQLite to PostgreSQL
- [ ] Implement EC2 Spot training trigger in `training_pipeline_dag`
- [ ] Fix `_store_predictions` upsert logic
- [ ] Wire actual drift detection in `data_pipeline_dag`
- [ ] Fix `sqlalchemy.text()` wrapping in `drift.py`
- [ ] Fix private `_record_metric` import in inference DAG
- [ ] Fix `alert_new_model_registered` version string

---

## Previous Session Context (from conversation history)

### Sessions summary
- **2026-04-26**: Fixed Hopsworks connection (app.hopsworks.ai endpoint), Python 3.10 venv setup.
- **2026-04-29 (session 1)**: Switched from Hopsworks to Feast+PostgreSQL feature store. Resolved datetime parsing (`format='mixed'`), Avro schema type issues, Evidently `save_html` crash handling.
- **2026-04-29 (session 2)**: Installed `confluent-kafka`, added `safe_insert` helper, resolved Hopsworks schema alignment. *(Note: These Hopsworks references are obsolete — feature store was migrated to Feast.)*
- **2026-04-29 (session 3)**: Verified WandB/MLflow config, generated preprocessing artifacts.
- **2026-05-06 (session 1)**: Migrated Feast registry to PostgreSQL design, planned EC2 Spot training, fixed Feast materialization and drift detection logic, replaced placeholder DAG tasks with production implementations. Codified changes in CLAUDE.md.
- **2026-05-07 (this session)**: Full project audit, log analysis, error cataloging, training compute strategy, structural change guide.
