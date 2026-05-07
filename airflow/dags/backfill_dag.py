"""
Backfill DAG
────────────
One-time DAG to load N months of historical NYC TLC data into PostgreSQL.
Triggered manually — will skip months already present in hourly_rides.

Usage:
  Trigger from Airflow UI or:
    airflow dags trigger backfill_dag
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="backfill_dag",
    description="One-time backfill of N months of historical TLC data",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["data", "backfill"],
) as dag:

    def _run_backfill(**ctx) -> None:
        import gc
        import sys
        sys.path.insert(0, "/opt/airflow")
        from io import StringIO

        from src.config import settings
        from src.data.ingestion import fetch_raw_trip_data, months_to_backfill
        from src.data.validation import validate_and_clean
        from src.data.transformation import to_hourly_time_series
        from src.utils.db import get_session, get_raw_connection
        from sqlalchemy import text

        months = months_to_backfill(12)  # 1 year of training data
        print(f"Backfill plan: {len(months)} months from {months[0]} to {months[-1]}")

        with get_session() as session:
            existing = set(
                session.execute(
                    text("SELECT DISTINCT DATE_TRUNC('month', pickup_hour)::DATE FROM hourly_rides")
                ).scalars()
            )
        print(f"Already loaded: {len(existing)} months")

        to_load = [(y, m) for y, m in months if datetime(y, m, 1).date() not in existing]
        print(f"Need to load: {len(to_load)} months: {to_load}")

        for i, (year, month) in enumerate(to_load, 1):
            print(f"\n[{i}/{len(to_load)}] Fetching {year}-{month:02d}...")
            raw_df = clean_df = ts = None
            try:
                raw_df = fetch_raw_trip_data(year, month)
                clean_df, report = validate_and_clean(raw_df, year, month)
                del raw_df; raw_df = None; gc.collect()

                if not report.passed:
                    print(f"  WARNING: drop_rate={report.drop_rate:.1%}, loading anyway")

                ts = to_hourly_time_series(clean_df)
                del clean_df; clean_df = None; gc.collect()

                conn = get_raw_connection()
                try:
                    with conn.cursor() as cur:
                        buf = StringIO()
                        ts[["pickup_location_id", "pickup_hour", "ride_count"]].to_csv(
                            buf, index=False, header=False
                        )
                        buf.seek(0)
                        cur.execute(
                            "CREATE TEMP TABLE tmp_rides (LIKE hourly_rides INCLUDING ALL) ON COMMIT DROP"
                        )
                        cur.copy_from(buf, "tmp_rides", sep=",", columns=["pickup_location_id", "pickup_hour", "ride_count"])
                        cur.execute("""
                            INSERT INTO hourly_rides (pickup_location_id, pickup_hour, ride_count)
                            SELECT pickup_location_id, pickup_hour, ride_count FROM tmp_rides
                            ON CONFLICT (pickup_location_id, pickup_hour)
                            DO UPDATE SET ride_count = EXCLUDED.ride_count
                        """)
                    conn.commit()
                    print(f"  Loaded {len(ts):,} rows for {year}-{month:02d}")
                finally:
                    conn.close()

            except Exception as e:
                print(f"  ERROR for {year}-{month:02d}: {e} — skipping")
            finally:
                del raw_df, clean_df, ts
                gc.collect()

        with get_session() as session:
            total = session.execute(text("SELECT COUNT(*) FROM hourly_rides")).scalar()
            date_range = session.execute(
                text("SELECT MIN(pickup_hour), MAX(pickup_hour) FROM hourly_rides")
            ).fetchone()
        print(f"\nBackfill complete. DB now has {total:,} rows from {date_range[0]} to {date_range[1]}")

    backfill = PythonOperator(
        task_id="backfill_historical_data",
        python_callable=_run_backfill,
        execution_timeout=timedelta(hours=6),
    )
