"""
Data drift and model drift detection using Evidently AI.
Generates HTML reports and structured drift scores logged to PostgreSQL.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report

from sqlalchemy import text

from src.utils.db import get_session
from src.utils.logging_utils import logger

REPORTS_DIR = Path("reports/drift")


def _ensure_reports_dir():
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def run_data_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_cols: list[str],
    report_name: str = "data_drift",
) -> dict:
    """
    Compare current data distribution against reference (training data).
    Returns a dict with drift_detected (bool) and drift_score (float).
    Saves an HTML report to reports/drift/.
    """
    _ensure_reports_dir()

    column_mapping = ColumnMapping(numerical_features=feature_cols)
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[feature_cols],
        current_data=current[feature_cols],
        column_mapping=column_mapping,
    )

    report_path = (
        REPORTS_DIR / f"{report_name}_{datetime.now(tz=timezone.utc):%Y%m%d_%H%M}.html"
    )
    report.save_html(str(report_path))

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    # share_of_drifted_columns is a good scalar drift score
    drift_score = result["metrics"][0]["result"].get("share_of_drifted_columns", 0.0)

    logger.info(
        f"Data drift report: detected={drift_detected}, score={drift_score:.3f}, "
        f"saved to {report_path}"
    )
    record_monitoring_metric("data_drift_score", drift_score)
    record_monitoring_metric("data_drift_detected", float(drift_detected))

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "report_path": str(report_path),
    }


def run_model_performance_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_col: str = "target_rides",
    prediction_col: str = "predicted_rides",
    report_name: str = "model_performance",
) -> dict:
    """
    Compare model performance on current data vs reference window.
    Returns dict with MAE degradation flag and current MAE.
    """
    _ensure_reports_dir()

    column_mapping = ColumnMapping(
        target=target_col,
        prediction=prediction_col,
    )
    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data=reference[[target_col, prediction_col]],
        current_data=current[[target_col, prediction_col]],
        column_mapping=column_mapping,
    )

    report_path = (
        REPORTS_DIR / f"{report_name}_{datetime.now(tz=timezone.utc):%Y%m%d_%H%M}.html"
    )
    report.save_html(str(report_path))

    result = report.as_dict()
    current_mae = result["metrics"][0]["result"]["current"]["mean_abs_error"]
    ref_mae = result["metrics"][0]["result"]["reference"]["mean_abs_error"]
    degradation = (current_mae - ref_mae) / ref_mae if ref_mae else 0

    logger.info(
        f"Model performance: current_mae={current_mae:.2f}, "
        f"ref_mae={ref_mae:.2f}, degradation={degradation:.1%}"
    )
    record_monitoring_metric("model_mae", current_mae)
    record_monitoring_metric("model_mae_degradation", degradation)

    return {
        "current_mae": current_mae,
        "reference_mae": ref_mae,
        "degradation": degradation,
        "report_path": str(report_path),
    }


def record_monitoring_metric(
    metric_name: str, value: float, metadata: dict | None = None
) -> None:
    """Persist a monitoring metric to the model_monitoring table."""
    try:
        with get_session() as session:
            session.execute(
                text("""
                INSERT INTO model_monitoring (metric_name, metric_value, metadata)
                VALUES (:name, :value, :meta)
                """),
                {
                    "name": metric_name,
                    "value": value,
                    "meta": json.dumps(metadata or {}),
                },
            )
    except Exception as e:
        logger.warning(f"Failed to record metric {metric_name}: {e}")
