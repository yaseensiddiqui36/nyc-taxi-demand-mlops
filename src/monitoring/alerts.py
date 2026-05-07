"""
Alert system using Slack webhooks.
Called by Airflow DAGs and the drift monitoring module.
"""

from __future__ import annotations

import requests

from src.config import settings
from src.utils.logging_utils import logger


def _post_slack(message: str, color: str = "#36a64f") -> bool:
    """Send a formatted Slack message. Returns True on success."""
    if not settings.slack_webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set — alert skipped")
        return False

    payload = {
        "attachments": [
            {
                "color": color,
                "text": message,
                "footer": "NYC Taxi Demand MLOps",
                "ts": __import__("time").time(),
            }
        ]
    }
    try:
        resp = requests.post(settings.slack_webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Slack alert failed: {e}")
        return False


def alert_pipeline_success(pipeline_name: str, details: str = "") -> None:
    msg = f":white_check_mark: *{pipeline_name}* completed successfully.\n{details}"
    _post_slack(msg, color="#36a64f")


def alert_pipeline_failure(pipeline_name: str, error: str) -> None:
    msg = f":red_circle: *{pipeline_name}* FAILED.\n```{error}```"
    _post_slack(msg, color="#e01e5a")


def alert_data_drift(drift_score: float, threshold: float = 0.25) -> None:
    if drift_score >= threshold:
        msg = (
            f":warning: *Data Drift Detected* — score `{drift_score:.3f}` "
            f"exceeds threshold `{threshold}`.\n"
            "Consider triggering model retraining."
        )
        _post_slack(msg, color="#ecb22e")
        logger.warning(f"Data drift alert sent: score={drift_score:.3f}")


def alert_model_degradation(current_mae: float, reference_mae: float, threshold_pct: float = 0.15) -> None:
    degradation = (current_mae - reference_mae) / reference_mae if reference_mae else 0
    if degradation >= threshold_pct:
        msg = (
            f":rotating_light: *Model Degradation Alert*\n"
            f"Current MAE: `{current_mae:.2f}` | Reference MAE: `{reference_mae:.2f}` "
            f"| Degradation: `{degradation:.1%}` (threshold: {threshold_pct:.0%})\n"
            "Retraining recommended."
        )
        _post_slack(msg, color="#e01e5a")
        logger.warning(f"Model degradation alert sent: {degradation:.1%} degradation")


def alert_validation_failure(year: int, month: int, drop_rate: float, details: dict) -> None:
    msg = (
        f":x: *Data Validation Failed* for {year}-{month:02d}\n"
        f"Drop rate: `{drop_rate:.1%}`\nDetails: `{details}`"
    )
    _post_slack(msg, color="#e01e5a")


def alert_new_model_registered(model_name: str, version: str, mae: float) -> None:
    msg = (
        f":rocket: *New Model Registered*\n"
        f"Model: `{model_name}` | Version: `{version}` | MAE: `{mae:.2f}`\n"
        "Promoted to Production."
    )
    _post_slack(msg, color="#36a64f")
