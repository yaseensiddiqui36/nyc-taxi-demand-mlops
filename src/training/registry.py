"""
MLflow Model Registry helpers.
Handles promoting new models from Staging → Production only when MAE improves.
"""

from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient

from src.config import settings
from src.utils.logging_utils import logger


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


def get_production_mae(model_name: str = settings.model_name) -> float | None:
    """Return the MAE of the current Production model, or None if no model exists."""
    client = get_client()
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if not versions:
            return None
        run = client.get_run(versions[0].run_id)
        return float(run.data.metrics.get("cv_mae", float("inf")))
    except Exception as e:
        logger.warning(f"Could not fetch production MAE: {e}")
        return None


def register_model_if_better(
    run_id: str,
    new_mae: float,
    model_name: str = settings.model_name,
) -> str | None:
    """
    Register the model from `run_id` and promote to Production if new_mae
    is strictly better than the current Production model.

    Returns the new version string if promoted, None otherwise.
    """
    client = get_client()
    current_mae = get_production_mae(model_name)

    if current_mae is not None and new_mae >= current_mae:
        logger.info(
            f"New model MAE={new_mae:.2f} is not better than current={current_mae:.2f}. "
            "Skipping registration."
        )
        return None

    # Register in Staging first
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(f"Registered model '{model_name}' version {mv.version} in Staging")

    # Transition old Production models to Archived
    if current_mae is not None:
        for old_version in client.get_latest_versions(
            model_name, stages=["Production"]
        ):
            client.transition_model_version_stage(
                name=model_name,
                version=old_version.version,
                stage="Archived",
            )
            logger.info(f"Archived previous production version {old_version.version}")

    # Promote new version to Production
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    logger.info(
        f"Promoted model '{model_name}' v{mv.version} to Production. "
        f"MAE: {current_mae} → {new_mae:.2f}"
    )
    return mv.version


def load_production_model(model_name: str = settings.model_name):
    """
    Load the current Production model.
    Checks local joblib file first (instant, no network), then MLflow registry.
    """
    import joblib
    from pathlib import Path

    local_path = (
        Path(__file__).parent.parent.parent / "models" / "taxi_demand_model.joblib"
    )

    # Attempt 1: local file (fast, no network dependency)
    if local_path.exists():
        model = joblib.load(local_path)
        logger.info(f"Loaded production model from local file: {local_path}")
        return model

    # Attempt 2: MLflow registry
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        logger.info(f"Loaded production model from MLflow registry: {model_name}")
        return model
    except Exception as e:
        logger.warning(f"Registry URI load failed: {e}")

    # Attempt 3: direct artifact URI (MLflow 2.x LoggedModel path)
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        prod = next((v for v in versions if v.current_stage == "Production"), None)
        if prod:
            source = prod.source
            artifact_dir = (
                source if not source.endswith(".pkl") else source.rsplit("/", 1)[0]
            )
            model = mlflow.sklearn.load_model(artifact_dir)
            logger.info(f"Loaded production model from artifact URI: {artifact_dir}")
            return model
    except Exception as e:
        logger.warning(f"Direct artifact URI load failed: {e}")

    raise RuntimeError(
        f"No production model available. Place a model at {local_path}"
    )

    raise RuntimeError(
        "No production model available. Either fix MLflow artifact upload "
        f"or place a model at {local_path}"
    )
