"""
Data Pipeline DAG
─────────────────
Schedule: daily at 06:00 UTC
Purpose:  Check for new NYC TLC data, validate it, transform to hourly
          time-series, load to PostgreSQL, push to Feast feature store.

Task flow:
  check_new_data
       ↓
  fetch_and_validate          (branches to notify_validation_failure if bad)
       ↓
  transform_to_timeseries
       ↓
  load_to_postgres
       ↓
  push_to_feast_offline
       ↓
  materialize_feast_online
       ↓
  run_drift_check
       ↓
  notify_success
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator

default_args = {
    "owner": "mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="data_pipeline_dag",
    description="Fetch, validate, and store NYC taxi hourly time-series",
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["data", "ingestion"],
) as dag:

    def _check_new_data(**ctx) -> str:
        """Return 'fetch_and_validate' if new data is available, else 'no_new_data'."""
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.data.ingestion import latest_available_month
        from src.utils.db import get_session
        from sqlalchemy import text

        year, month = latest_available_month()
        with get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*) FROM hourly_rides
                    WHERE DATE_TRUNC('month', pickup_hour) =
                          DATE_TRUNC('month', MAKE_DATE(:year, :month, 1)::TIMESTAMPTZ)
                """),
                {"year": year, "month": month},
            ).scalar()

        if result and result > 0:
            print(f"Data for {year}-{month:02d} already loaded ({result:,} rows). Skipping.")
            return "no_new_data"

        ctx["ti"].xcom_push(key="year", value=year)
        ctx["ti"].xcom_push(key="month", value=month)
        print(f"New data found: {year}-{month:02d}")
        return "fetch_and_validate"

    def _fetch_and_validate(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.data.ingestion import fetch_raw_trip_data
        from src.data.validation import validate_and_clean
        from src.monitoring.alerts import alert_validation_failure

        ti = ctx["ti"]
        year = ti.xcom_pull(key="year", task_ids="check_new_data")
        month = ti.xcom_pull(key="month", task_ids="check_new_data")

        raw_df = fetch_raw_trip_data(year, month)
        clean_df, report = validate_and_clean(raw_df, year, month)

        if not report.passed:
            alert_validation_failure(year, month, report.drop_rate, report.failure_reasons)
            raise ValueError(
                f"Validation failed for {year}-{month:02d}: drop_rate={report.drop_rate:.1%}"
            )

        ti.xcom_push(key="clean_df_path", value=f"/tmp/clean_{year}_{month:02d}.parquet")
        clean_df.to_parquet(f"/tmp/clean_{year}_{month:02d}.parquet", index=False)
        print(f"Validated: {report.rows_out:,} rows (dropped {report.drop_rate:.1%})")

    def _transform_to_timeseries(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.data.transformation import to_hourly_time_series

        ti = ctx["ti"]
        parquet_path = ti.xcom_pull(key="clean_df_path", task_ids="fetch_and_validate")
        year = ti.xcom_pull(key="year", task_ids="check_new_data")
        month = ti.xcom_pull(key="month", task_ids="check_new_data")

        df = pd.read_parquet(parquet_path)
        ts = to_hourly_time_series(df)
        ts_path = f"/tmp/ts_{year}_{month:02d}.parquet"
        ts.to_parquet(ts_path, index=False)
        ti.xcom_push(key="ts_path", value=ts_path)
        print(f"Time-series: {len(ts):,} rows")

    def _load_to_postgres(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.utils.db import get_raw_connection

        ti = ctx["ti"]
        ts_path = ti.xcom_pull(key="ts_path", task_ids="transform_to_timeseries")
        ts = pd.read_parquet(ts_path)

        conn = get_raw_connection()
        try:
            with conn.cursor() as cur:
                from io import StringIO
                buf = StringIO()
                ts[["pickup_location_id", "pickup_hour", "ride_count"]].to_csv(buf, index=False, header=False)
                buf.seek(0)
                cur.execute("CREATE TEMP TABLE tmp_rides (LIKE hourly_rides INCLUDING ALL) ON COMMIT DROP")
                cur.copy_from(buf, "tmp_rides", sep=",", columns=["pickup_location_id", "pickup_hour", "ride_count"])
                cur.execute("""
                    INSERT INTO hourly_rides (pickup_location_id, pickup_hour, ride_count)
                    SELECT pickup_location_id, pickup_hour, ride_count FROM tmp_rides
                    ON CONFLICT (pickup_location_id, pickup_hour)
                    DO UPDATE SET ride_count = EXCLUDED.ride_count
                """)
            conn.commit()
            print(f"Loaded {len(ts):,} rows to hourly_rides")
        finally:
            conn.close()

    def _push_to_feast_offline(**ctx) -> None:
        # PostgreSQL offline store reads directly from public.hourly_rides via
        # the PostgreSQLSource query — no separate push step needed.
        print("Offline store is PostgreSQL (public.hourly_rides) — skipping push, proceeding to materialize.")

    def _materialize_feast_online(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.features.feature_store import materialize_to_online_store
        materialize_to_online_store()

    def _run_drift_check(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        import pandas as pd
        from src.utils.db import get_session
        from sqlalchemy import text
        from src.monitoring.drift import run_data_drift_report

        with get_session() as session:
            # Current distribution: last 30 days
            current = pd.read_sql(
                "SELECT ride_count FROM hourly_rides WHERE pickup_hour >= NOW() - INTERVAL '30 days'",
                session.bind
            )
            # Reference distribution: 30 days before that
            reference = pd.read_sql(
                "SELECT ride_count FROM hourly_rides WHERE pickup_hour >= NOW() - INTERVAL '60 days' AND pickup_hour < NOW() - INTERVAL '30 days'",
                session.bind
            )

        if len(current) > 1000 and len(reference) > 1000:
            run_data_drift_report(reference, current, feature_cols=["ride_count"])
            print(f"Drift report generated. Current rows: {len(current)}, Ref rows: {len(reference)}")
        else:
            print("Not enough data to run drift check yet.")

    def _notify_success(**ctx) -> None:
        import sys
        sys.path.insert(0, "/opt/airflow")
        from src.monitoring.alerts import alert_pipeline_success
        ti = ctx["ti"]
        year = ti.xcom_pull(key="year", task_ids="check_new_data")
        month = ti.xcom_pull(key="month", task_ids="check_new_data")
        alert_pipeline_success("data_pipeline_dag", f"Loaded data for {year}-{month:02d}")

    # ── Task definitions ──────────────────────────────────────

    check = BranchPythonOperator(task_id="check_new_data", python_callable=_check_new_data)
    no_new = EmptyOperator(task_id="no_new_data")
    fetch  = PythonOperator(task_id="fetch_and_validate", python_callable=_fetch_and_validate)
    transform = PythonOperator(task_id="transform_to_timeseries", python_callable=_transform_to_timeseries)
    load_pg   = PythonOperator(task_id="load_to_postgres", python_callable=_load_to_postgres)
    feast_off = PythonOperator(task_id="push_to_feast_offline", python_callable=_push_to_feast_offline)
    feast_on  = PythonOperator(task_id="materialize_feast_online", python_callable=_materialize_feast_online)
    drift     = PythonOperator(task_id="run_drift_check", python_callable=_run_drift_check)
    notify    = PythonOperator(task_id="notify_success", python_callable=_notify_success)

    check >> [fetch, no_new]
    fetch >> transform >> load_pg >> feast_off >> feast_on >> drift >> notify
