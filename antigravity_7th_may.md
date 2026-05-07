# Session Log — 2026-05-07
# NYC Taxi Demand MLOps — Antigravity AI Session

> **Session Date**: 2026-05-07 (EDT)
> **Session Duration**: ~3 hours (~20:44 → ~00:44)
> **Primary Goal**: Fix known DAG bugs (Phase 5), provision AWS infrastructure, and deploy the project to production (Phase 6).
> **Status at end of session**: ⚠️ PARTIALLY COMPLETE — GitHub Actions CI still failing. Deployment blocked.

---

## Current Infrastructure State

| Component | Status | Details |
|---|---|---|
| **AWS EC2** | ✅ Running | `m7i-flex.large` (2 vCPU, 8GB RAM), Ubuntu 26.04 AMI |
| **Public IP** | ✅ Static | `54.87.148.188` |
| **SSH Key** | ✅ On disk | `nyc-taxi-demand-mlops/nyc-taxi-demand-key-pair.pem` (gitignored) |
| **Docker** | ✅ Installed | On EC2 server via `curl -fsSL https://get.docker.com | sudo sh` |
| **EC2 Disk** | ⚠️ Tight | 8GB root volume (AMI default). 3.9GB free after `docker system prune`. Full 40GB not applied. |
| **GitHub Repo** | ✅ Created | `https://github.com/yaseensiddiqui36/nyc-taxi-demand-mlops` |
| **GitHub Secrets** | ✅ All added | See secrets checklist below |
| **GitHub Actions** | ❌ CI Failing | `ModuleNotFoundError: No module named 'src.data'` — fix pushed, may need new run |
| **Deployment** | ❌ Not yet live | Blocked by CI + disk space errors |
| **S3 Bucket** | ✅ Created | `nyc-taxi-demand-mlops` (us-east-1) |
| **IAM Keys** | ✅ In .env | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` configured |

---

## GitHub Secrets (All Added ✅)

| Secret Name | Value / Source |
|---|---|
| `AWS_HOST` | `54.87.148.188` |
| `AWS_SSH_PRIVATE_KEY` | Contents of `nyc-taxi-demand-key-pair.pem` |
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `S3_BUCKET_NAME` | `nyc-taxi-demand-mlops` |
| `POSTGRES_PASSWORD` | `Yaseen#321` |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/yaseensiddiqui36/nyc-taxi-demand-mlops.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | `yaseensiddiqui36` |
| `MLFLOW_TRACKING_PASSWORD` | `42980097cf8a1a8a00d853c961352e8592d24089` |
| `SLACK_WEBHOOK_URL` | From `.env` line 54 |
| `GRAFANA_ADMIN_PASSWORD` | `Yaseen#321` |
| `AIRFLOW_PASSWORD` | `Yaseen#321` |
| `AIRFLOW_FERNET_KEY` | `2awRih1wNhODAM8Yu-B9U_-QOOiTUoSGrefXOvv1cVA=` |
| `AIRFLOW_SECRET_KEY` | `be9e4391c06edaf20f07996b1386a7c7dafb39101b6a2b46e0a844ec93000081` |

---

## Git Commit History (Today)

```
ca21cec  Fix deploy: add disk cleanup step, complete .env vars, use single compose file
5b0f2db  Fix CI: add PYTHONPATH and use plain postgres to fix disk space error
c70f27d  Re-trigger CI and deploy after Docker install
0fe2349  Initial commit
```

---

## All Code Changes Made Today

### 1. `.gitignore` — Added `*.pem`
**Why**: Protect the SSH private key (`nyc-taxi-demand-key-pair.pem`) from being accidentally committed.
```diff
+ *.pem
```

### 2. `src/monitoring/drift.py` — Three fixes
**Why**: SQLAlchemy 2.x compatibility, public API, remove private import dependency.
- Added `from sqlalchemy import text`
- Renamed `_record_metric()` → `record_monitoring_metric()` (public name)
- Wrapped all raw SQL strings in `text()` to fix SQLAlchemy 2.x `RemovedIn20Warning`

### 3. `airflow/dags/inference_pipeline_dag.py` — Two fixes
**Why**: Prevent crash on DAG retry + fix fragile private import.
- `_store_predictions`: Added `ON CONFLICT (pickup_location_id, predicted_hour) DO UPDATE SET predicted_rides = EXCLUDED.predicted_rides` — prevents unique key violation on retries
- `_check_model_drift`: Changed `from src.monitoring.drift import _record_metric` → `from src.monitoring.drift import record_monitoring_metric`

### 4. `airflow/dags/data_pipeline_dag.py` — Wired real drift detection
**Why**: `_run_drift_check` was a placeholder that only printed a message.
- Now queries last 30 days (current window) and prior 30 days (reference window) from `hourly_rides`
- Calls `run_data_drift_report()` from `src/monitoring/drift.py`
- Gracefully skips if not enough data (<1000 rows)

### 5. `airflow/dags/training_pipeline_dag.py` — Fixed version string
**Why**: Alert was always saying `"latest"` instead of actual model version.
- Changed `alert_new_model_registered(best_name, "latest", best_mae)` → `alert_new_model_registered(best_name, promoted, best_mae)`
- `promoted` now contains the actual version string (see registry.py change below)

### 6. `src/training/registry.py` — Return version string
**Why**: `register_model_if_better()` was returning `True/False` which loses the version number.
- Changed return type from `bool` to `str | None`
- Now returns `mv.version` (the actual version string) on success
- Returns `None` instead of `False` on no-improvement

### 7. `feature_repo/feature_store.yaml` — Migrated Feast registry SQLite → PostgreSQL
**Why**: SQLite file (`data/registry.db`) causes `database is locked` errors in multi-container Docker because each container gets its own file copy.
```diff
- registry:
-   path: data/registry.db
+ registry:
+   registry_type: sql
+   path: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/taxi_demand
```
**Action required**: Run `feast apply` once after deploying to register feature views in the new PostgreSQL registry.

### 8. `docker-compose.yml` — Moved Airflow to always-on
**Why**: Decision made to run Airflow on the AWS server 24/7 (enabled by 8GB RAM). Previously under `--profile local`.
- Removed `profiles: [local]` from `x-airflow-common` and `mlflow` service blocks
- Updated header comments

### 9. `.env` — Set Grafana password
```diff
- GRAFANA_ADMIN_PASSWORD=<CHANGE_ME>
+ GRAFANA_ADMIN_PASSWORD=Yaseen#321
```

### 10. `.github/workflows/ci.yml` — Fixed CI failures
**Why**: Tests fail with `ModuleNotFoundError: No module named 'src.data'` because pytest can't find the `src` package.
- Added `PYTHONPATH: ${{ github.workspace }}` to the test job's env section
- Replaced `timescale/timescaledb:latest-pg15` service image with `postgres:15` — TimescaleDB image is ~500MB and causes `no space left on device` on GitHub Actions runners

### 11. `.github/workflows/deploy.yml` — Fixed deployment workflow
**Why**: Deploy was failing with `no space left on device` on the EC2 server + missing env vars.
- Added **"Free disk space on server"** step that runs `docker system prune -f` before pulling images
- Removed `-f docker-compose.prod.yml` override (not needed, uses single `docker-compose.yml`)
- Added all missing `.env` variables: `POSTGRES_HOST`, `POSTGRES_PORT`, `REDIS_HOST`, `REDIS_PORT`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD`, all Airflow vars, `BACKFILL_MONTHS`
- Added `mkdir -p` to ensure deploy directory exists before writing `.env`

### 12. `conftest.py` (NEW FILE) — Root-level pytest config
**Why**: Belt-and-suspenders fix for CI import errors. `PYTHONPATH` env var works in most cases but conftest.py is more reliable because pytest loads it before any test module import.
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

---

## Current Failing GitHub Actions

### CI — `ModuleNotFoundError: No module named 'src.data'`

**Root cause**: pytest on GitHub Actions runner can't find the `src` package because the project root isn't on Python's import path.

**Fixes applied**:
1. Added `PYTHONPATH: ${{ github.workspace }}` in `ci.yml` (commit `5b0f2db`)
2. Added root-level `conftest.py` with `sys.path.insert` (current session, not yet committed)

**Status**: Fix is in code but the error log the user pasted is from an OLD run (before commit `5b0f2db`). The env vars in the log don't show `PYTHONPATH`, confirming it's a stale run. **A new GitHub Actions run triggered by the next push should pass CI.**

### Deploy — `no space left on device`

**Root cause**: EC2 instance has only 8GB root disk (AMI default). The TimescaleDB Docker image (~400-500MB compressed) filled up the disk during `docker compose pull`.

**Fixes applied**:
1. Manual `docker system prune -af` on server — freed 516MB (disk went from 71% → 42% used)
2. Added automatic `docker system prune -f` step in `deploy.yml` before every pull (commit `ca21cec`)

**Remaining risk**: The full stack (PostgreSQL/TimescaleDB, Redis, Airflow, FastAPI, Streamlit, Prometheus, Grafana, MLflow) may still exceed 8GB after all images are pulled. If this happens:
- Go to AWS EC2 Console → Volumes → Select the volume → Actions → Modify Volume → Set to 30GB
- Then SSH in and run: `sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1`

---

## Architecture Decisions Made Today

| Decision | Choice | Reason |
|---|---|---|
| Lightsail plan | ❌ Rejected | Pricing was higher than expected ($44/mo for 8GB, not $20) |
| EC2 instance type | `m7i-flex.large` | Free Tier eligible, 8GB RAM, ~$3.20/month (storage only) |
| Airflow location | On EC2 server (always-on) | 8GB RAM is sufficient; enables fully automated 24/7 scheduling |
| Training | Use existing DagsHub model | Model already trained on Colab and registered. EC2 Spot automation deferred to Phase 7 |
| Feast registry | PostgreSQL (migrated from SQLite) | SQLite causes multi-container file locking issues |

---

## Next Steps for Next Session

1. **Verify CI passes** — Check if the `conftest.py` + `PYTHONPATH` fix resolves the import errors
2. **Verify Deploy passes** — Check if the disk cleanup step gives enough room for all Docker images
3. **If disk still full** — Resize the EC2 EBS volume from 8GB → 30GB from AWS console
4. **Post-deployment** — SSH into server, run `feast apply` to register Feast feature views in PostgreSQL registry
5. **Backfill** — Trigger `backfill_dag` from Airflow UI to load 10 months of TLC data into production PostgreSQL
6. **Verify inference** — Check that the inference DAG can load the existing Production model from DagsHub MLflow

---

## Service URLs (once deployed)

| Service | URL | Credentials |
|---|---|---|
| FastAPI | http://54.87.148.188:8000/docs | — |
| Streamlit | http://54.87.148.188:8501 | — |
| Airflow | http://54.87.148.188:8081 | admin / Yaseen#321 |
| Grafana | http://54.87.148.188:3000 | admin / Yaseen#321 |
| Prometheus | http://54.87.148.188:9090 | — |

---

## Key File Locations

| File | Purpose |
|---|---|
| `CLAUDE.md` | Full project architecture, commands, and change log |
| `conftest.py` | Root pytest config — fixes CI import path |
| `.github/workflows/ci.yml` | CI lint + test workflow |
| `.github/workflows/deploy.yml` | Automated deployment to EC2 |
| `.github/workflows/build-push.yml` | Docker image build + push |
| `docker-compose.yml` | Full stack definition (all services always-on) |
| `feature_repo/feature_store.yaml` | Feast config (now using PostgreSQL registry) |
| `nyc-taxi-demand-key-pair.pem` | EC2 SSH key (GITIGNORED — do not commit) |
| `.env` | All secrets (GITIGNORED — do not commit) |
