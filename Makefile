.PHONY: help up down dev-up dev-down logs build test lint format \
        data-pipeline train infer mlflow-up mlflow-down \
        feast-apply feast-materialize deploy

DOCKER_COMPOSE = docker compose
PROFILE_LOCAL  = --profile local

# ─────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "NYC Taxi Demand MLOps"
	@echo "────────────────────────────────────────────────────"
	@echo "  make up              Start always-on services (API, frontend, DB, monitoring)"
	@echo "  make down            Stop all services"
	@echo "  make dev-up          Start full local stack (+ Airflow + MLflow)"
	@echo "  make dev-down        Stop full local stack"
	@echo "  make build           Rebuild all Docker images"
	@echo "  make logs            Tail all service logs"
	@echo "  make logs-api        Tail API logs only"
	@echo ""
	@echo "  make data-pipeline   Trigger Airflow data pipeline DAG"
	@echo "  make train           Trigger Airflow training pipeline DAG"
	@echo "  make infer           Trigger Airflow inference pipeline DAG"
	@echo ""
	@echo "  make mlflow-up       Start MLflow server only"
	@echo "  make feast-apply     Apply Feast feature definitions"
	@echo "  make feast-materialize  Materialize features to Redis online store"
	@echo ""
	@echo "  make test            Run all tests"
	@echo "  make lint            Run ruff linter"
	@echo "  make format          Auto-format with ruff"
	@echo ""
	@echo "  make deploy          Deploy to AWS (production)"
	@echo ""

# ─── Service management ───────────────────────────────────────

up:
	$(DOCKER_COMPOSE) up -d
	@echo "Services: API=http://localhost:8000  Frontend=http://localhost:8501  Grafana=http://localhost:3000"

down:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) down

dev-up:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) up -d
	@echo "Full stack running:"
	@echo "  Airflow  → http://localhost:8081  (admin / see .env)"
	@echo "  MLflow   → http://localhost:5000"
	@echo "  API      → http://localhost:8000"
	@echo "  Frontend → http://localhost:8501"
	@echo "  Grafana  → http://localhost:3000"

dev-down:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) down

build:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) build --no-cache

logs:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) logs -f

logs-api:
	$(DOCKER_COMPOSE) logs -f api

logs-airflow:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) logs -f airflow-webserver airflow-scheduler

# ─── Airflow DAG triggers ─────────────────────────────────────

data-pipeline:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) exec airflow-webserver \
		airflow dags trigger data_pipeline_dag

train:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) exec airflow-webserver \
		airflow dags trigger training_pipeline_dag

infer:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) exec airflow-webserver \
		airflow dags trigger inference_pipeline_dag

# ─── MLflow ───────────────────────────────────────────────────

mlflow-up:
	$(DOCKER_COMPOSE) $(PROFILE_LOCAL) up -d mlflow postgres
	@echo "MLflow UI → http://localhost:5000"

# ─── Feast ────────────────────────────────────────────────────

feast-apply:
	cd feature_repo && feast apply

feast-materialize:
	cd feature_repo && feast materialize-incremental $$(date -u +"%Y-%m-%dT%H:%M:%S")

# ─── Code quality ─────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

lint:
	ruff check src/ tests/ airflow/

format:
	ruff format src/ tests/ airflow/
	ruff check --fix src/ tests/ airflow/

# ─── Database ─────────────────────────────────────────────────

db-shell:
	$(DOCKER_COMPOSE) exec postgres psql -U taxi -d taxi_demand

# ─── Deployment ───────────────────────────────────────────────

deploy:
	@echo "Deploying to AWS..."
	@echo "Make sure you have set AWS_HOST, AWS_KEY_PATH in your environment"
	rsync -avz --exclude='.env' --exclude='data/' --exclude='*.pyc' \
		./ ubuntu@$$AWS_HOST:/home/ubuntu/nyc-taxi-demand-mlops/
	ssh ubuntu@$$AWS_HOST \
		"cd /home/ubuntu/nyc-taxi-demand-mlops && \
		 docker compose -f docker-compose.yml -f docker-compose.prod.yml pull && \
		 docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d"
