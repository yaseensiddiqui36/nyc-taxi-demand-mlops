"""
Model Training Pipeline DAG
────────────────────────────
Schedule: every Monday at 02:00 UTC
Purpose:  Fetch 180 days of features, run model experiments, register
          the winner to MLflow Model Registry if it beats current Production.

Task flow:
  check_enough_data
       ↓
  build_training_dataset
       ↓
  train_baseline  ──┐
  train_lgbm      ──┼── (parallel)
  train_xgboost   ──┘
       ↓
  select_best_model
       ↓
  register_if_better
       ↓
  notify_result
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
}

with DAG(
    dag_id="training_pipeline_dag",
    description="Weekly model training, evaluation, and conditional registry promotion",
    schedule="0 2 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["training", "mlflow"],
) as dag:

    def _check_enough_data(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.utils.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            row_count = session.execute(
                text("SELECT COUNT(*) FROM hourly_rides WHERE pickup_hour >= NOW() - INTERVAL '180 days'")
            ).scalar()

        if row_count < 10_000:
            raise ValueError(f"Insufficient training data: {row_count:,} rows (need ≥10,000)")
        print(f"Training data available: {row_count:,} rows")

    def _build_training_dataset(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.utils.db import get_session
        from src.data.transformation import to_features_and_target
        from sqlalchemy import text
        from src.config import settings

        with get_session() as session:
            ts = pd.read_sql(
                """
                SELECT pickup_location_id, pickup_hour, ride_count
                FROM hourly_rides
                WHERE pickup_hour >= NOW() - INTERVAL '180 days'
                ORDER BY pickup_location_id, pickup_hour
                """,
                session.bind,
            )

        X, y = to_features_and_target(ts, window_hours=settings.feature_window_hours)
        X.to_parquet("/tmp/train_X.parquet", index=False)
        y.to_frame("target").to_parquet("/tmp/train_y.parquet", index=False)
        print(f"Training dataset: {len(X):,} samples, {len(X.columns)} features")

    def _train_model(model_name: str, **ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.training.train import train_and_evaluate

        X = pd.read_parquet("/tmp/train_X.parquet")
        y = pd.read_parquet("/tmp/train_y.parquet")["target"]
        result = train_and_evaluate(X, y, model_name=model_name)
        ctx["ti"].xcom_push(key=f"{model_name}_mae", value=result["mae"])
        ctx["ti"].xcom_push(key=f"{model_name}_run_id", value=result["run_id"])
        print(f"{model_name} MAE={result['mae']:.2f}")

    def _select_best_model(**ctx) -> None:
        ti = ctx["ti"]
        candidates = {}
        for name in ["baseline", "lgbm", "xgboost"]:
            mae = ti.xcom_pull(key=f"{name}_mae", task_ids=f"train_{name}")
            if mae is not None:
                candidates[name] = mae

        best_name = min(candidates, key=candidates.get)
        best_mae = candidates[best_name]
        best_run_id = ti.xcom_pull(key=f"{best_name}_run_id", task_ids=f"train_{best_name}")
        ti.xcom_push(key="best_model_name", value=best_name)
        ti.xcom_push(key="best_mae", value=best_mae)
        ti.xcom_push(key="best_run_id", value=best_run_id)
        print(f"Best model: {best_name} (MAE={best_mae:.2f})")

    def _register_if_better(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.training.registry import register_model_if_better

        ti = ctx["ti"]
        run_id = ti.xcom_pull(key="best_run_id", task_ids="select_best_model")
        new_mae = ti.xcom_pull(key="best_mae", task_ids="select_best_model")
        promoted = register_model_if_better(run_id, new_mae)
        ti.xcom_push(key="promoted", value=promoted)

    def _notify_result(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.monitoring.alerts import alert_pipeline_success, alert_new_model_registered

        ti = ctx["ti"]
        promoted = ti.xcom_pull(key="promoted", task_ids="register_if_better")
        best_name = ti.xcom_pull(key="best_model_name", task_ids="select_best_model")
        best_mae = ti.xcom_pull(key="best_mae", task_ids="select_best_model")

        if promoted:
            alert_new_model_registered(best_name, promoted, best_mae)
        else:
            alert_pipeline_success(
                "training_pipeline_dag",
                f"Training complete. Best={best_name} MAE={best_mae:.2f}. No improvement over Production."
            )

    # ── Task graph ────────────────────────────────────────────

    check_data   = PythonOperator(task_id="check_enough_data", python_callable=_check_enough_data)
    build_data   = PythonOperator(task_id="build_training_dataset", python_callable=_build_training_dataset)

    train_base   = PythonOperator(
        task_id="train_baseline", python_callable=_train_model,
        op_kwargs={"model_name": "baseline"}
    )
    train_lgbm   = PythonOperator(
        task_id="train_lgbm", python_callable=_train_model,
        op_kwargs={"model_name": "lgbm"}
    )
    train_xgb    = PythonOperator(
        task_id="train_xgboost", python_callable=_train_model,
        op_kwargs={"model_name": "xgboost"}
    )
    select_best  = PythonOperator(task_id="select_best_model", python_callable=_select_best_model)
    register     = PythonOperator(task_id="register_if_better", python_callable=_register_if_better)
    notify       = PythonOperator(task_id="notify_result", python_callable=_notify_result)

    check_data >> build_data >> [train_base, train_lgbm, train_xgb] >> select_best >> register >> notify
