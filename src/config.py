from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── Environment ──────────────────────────────────────────
    environment: str = "development"
    log_level: str = "INFO"

    # ── PostgreSQL ───────────────────────────────────────────
    postgres_user: str = "taxi"
    postgres_password: str
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "taxi_demand"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def async_database_url(self) -> str:
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")

    # ── Redis ────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379"
    redis_host: str = "redis"
    redis_port: int = 6379

    # ── MLflow ───────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_artifact_root: str = "/mlflow/artifacts"

    # ── DagsHub (for Colab / remote experiments) ─────────────
    dagshub_user: str = ""
    dagshub_token: str = ""
    dagshub_mlflow_uri: str = ""

    # ── AWS ──────────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    s3_bucket_name: str = "nyc-taxi-demand-mlops"

    # ── Slack ────────────────────────────────────────────────
    slack_webhook_url: str = ""
    slack_channel: str = "#taxi-demand-alerts"

    # ── Data ─────────────────────────────────────────────────
    nyc_tlc_base_url: str = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    nyc_taxi_zones_url: str = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    backfill_months: int = 24

    # ── Model ────────────────────────────────────────────────
    model_name: str = "taxi_demand_predictor"
    feature_window_hours: int = 672   # 28 days × 24 hours
    prediction_horizon_hours: int = 1

    # ── Paths ────────────────────────────────────────────────
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def feature_repo_dir(self) -> Path:
        return self.project_root / "feature_repo"


settings = Settings()
