"""
Inference Pipeline DAG
───────────────────────
Schedule: every hour at :05
Purpose:  Build features for the coming hour, generate predictions for all
          NYC zones, store to PostgreSQL, run model drift check.

Task flow:
  build_inference_features
          ↓
  generate_predictions
          ↓
  store_predictions
          ↓
  check_model_drift
          ↓
  notify_on_drift  (only if drift detected)
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "mlops",
    "retries": 3,
    "retry_delay": timedelta(minutes=2),
    "email_on_failure": False,
}

with DAG(
    dag_id="inference_pipeline_dag",
    description="Hourly: generate taxi demand predictions for all NYC zones",
    schedule="5 * * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["inference", "serving"],
) as dag:

    def _build_inference_features(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.utils.db import get_session
        from src.data.transformation import to_inference_features
        from src.config import settings
        from sqlalchemy import text

        with get_session() as session:
            # Anchor on the latest available data, not wall-clock time.
            # This handles the case where TLC data lags behind the current date.
            ts = pd.read_sql(
                """
                SELECT pickup_location_id, pickup_hour, ride_count
                FROM hourly_rides
                WHERE pickup_hour >= (
                    SELECT MAX(pickup_hour) FROM hourly_rides
                ) - INTERVAL '28 days'
                ORDER BY pickup_location_id, pickup_hour
                """,
                session.bind,
            )
            latest_hour = session.execute(
                text("SELECT MAX(pickup_hour) FROM hourly_rides")
            ).scalar()

        predicted_hour = pd.Timestamp(latest_hour) + pd.Timedelta(hours=1)
        ctx["ti"].xcom_push(key="predicted_hour", value=predicted_hour.isoformat())

        X = to_inference_features(ts, window_hours=settings.feature_window_hours)
        X.to_parquet("/tmp/inference_X.parquet", index=False)
        ctx["ti"].xcom_push(key="n_locations", value=len(X))
        print(f"Built inference features for {len(X)} locations")
        print(f"Predicting for hour: {predicted_hour}")

    def _generate_predictions(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        import numpy as np
        from src.training.registry import load_production_model

        X = pd.read_parquet("/tmp/inference_X.parquet")
        model = load_production_model()
        feature_cols = [c for c in X.columns if c not in ("pickup_hour",)]
        preds = model.predict(X[feature_cols]).clip(0)

        results = pd.DataFrame({
            "pickup_location_id": X["pickup_location_id"].values,
            "predicted_rides": np.round(preds, 2),
        })
        results.to_parquet("/tmp/predictions.parquet", index=False)
        print(f"Generated {len(results)} predictions")

    def _store_predictions(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.utils.db import get_raw_connection
        from src.training.registry import get_client
        from src.config import settings
        from datetime import datetime, timezone
        from io import StringIO

        ti = ctx["ti"]
        preds = pd.read_parquet("/tmp/predictions.parquet")
        predicted_hour_str = ti.xcom_pull(key="predicted_hour", task_ids="build_inference_features")
        if predicted_hour_str:
            predicted_hour = datetime.fromisoformat(predicted_hour_str)
        else:
            predicted_hour = datetime.now(tz=timezone.utc).replace(minute=0, second=0, microsecond=0)
        preds["predicted_hour"] = predicted_hour
        preds["model_name"] = settings.model_name

        conn = get_raw_connection()
        try:
            with conn.cursor() as cur:
                for _, row in preds.iterrows():
                    cur.execute(
                        """
                        INSERT INTO predictions
                            (pickup_location_id, predicted_hour, predicted_rides, model_name)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (pickup_location_id, predicted_hour) 
                        DO UPDATE SET predicted_rides = EXCLUDED.predicted_rides
                        """,
                        (row["pickup_location_id"], row["predicted_hour"],
                         row["predicted_rides"], row["model_name"]),
                    )
            conn.commit()
            print(f"Stored {len(preds)} predictions for {predicted_hour.isoformat()}")
        finally:
            conn.close()

    def _check_model_drift(**ctx) -> str:
        """
        Compare recent prediction errors against reference window.
        Returns 'notify_on_drift' if degradation detected, else 'no_drift'.
        """
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.utils.db import get_session
        from sqlalchemy import text

        with get_session() as session:
            rows = session.execute(
                text("""
                    SELECT p.predicted_rides, h.ride_count AS actual_rides
                    FROM predictions p
                    JOIN hourly_rides h
                        ON h.pickup_location_id = p.pickup_location_id
                        AND h.pickup_hour = p.predicted_hour
                    WHERE p.predicted_at >= NOW() - INTERVAL '7 days'
                """)
            ).fetchall()

        if len(rows) < 100:
            print("Not enough actuals yet for drift check")
            return "no_drift"

        import numpy as np
        preds_arr = np.array([r[0] for r in rows])
        actuals_arr = np.array([r[1] for r in rows])
        recent_mae = float(np.mean(np.abs(preds_arr - actuals_arr)))

        # Log to monitoring table
        from src.monitoring.drift import record_monitoring_metric
        record_monitoring_metric("model_mae", recent_mae)

        # Pull reference MAE from MLflow
        from src.training.registry import get_production_mae
        prod_mae = get_production_mae()
        if prod_mae and recent_mae > prod_mae * 1.20:
            print(f"Drift detected: recent_mae={recent_mae:.2f}, prod_mae={prod_mae:.2f}")
            ctx["ti"].xcom_push(key="recent_mae", value=recent_mae)
            ctx["ti"].xcom_push(key="reference_mae", value=prod_mae)
            return "notify_on_drift"

        print(f"No drift: recent_mae={recent_mae:.2f}")
        return "no_drift"

    def _notify_on_drift(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.monitoring.alerts import alert_model_degradation

        ti = ctx["ti"]
        recent_mae = ti.xcom_pull(key="recent_mae", task_ids="check_model_drift")
        ref_mae = ti.xcom_pull(key="reference_mae", task_ids="check_model_drift")
        alert_model_degradation(recent_mae, ref_mae)

    # ── Task graph ──────────────────────────────────────────���─

    build_features = PythonOperator(
        task_id="build_inference_features", python_callable=_build_inference_features
    )
    gen_preds = PythonOperator(
        task_id="generate_predictions", python_callable=_generate_predictions
    )
    store_preds = PythonOperator(
        task_id="store_predictions", python_callable=_store_predictions
    )
    drift_check = BranchPythonOperator(
        task_id="check_model_drift", python_callable=_check_model_drift
    )
    no_drift = EmptyOperator(task_id="no_drift")
    notify_drift = PythonOperator(
        task_id="notify_on_drift", python_callable=_notify_on_drift
    )

    build_features >> gen_preds >> store_preds >> drift_check >> [no_drift, notify_drift]
