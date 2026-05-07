"""
Local Backfill Script
─────────────────────
Run this OUTSIDE Docker using your local .venv. It downloads TLC parquet
files to data/raw/, processes them, and loads them into PostgreSQL.

Usage (from project root, with .venv activated):
    python scripts/backfill_local.py

Months already in the DB are skipped automatically.
"""

from __future__ import annotations

import gc
import os
import sys
from datetime import date, datetime
from io import StringIO
from pathlib import Path

# ── resolve project root so src.* imports work ───────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env manually (pydantic-settings looks for .env relative to cwd)
os.chdir(PROJECT_ROOT)

import requests
import pandas as pd
from sqlalchemy import create_engine, text

# ── Config ────────────────────────────────────────────────────────────────────
BACKFILL_MONTHS   = 12
TLC_BASE_URL      = "https://d37ci6vzurychx.cloudfront.net/trip-data"
RAW_DIR           = PROJECT_ROOT / "data" / "raw"
POSTGRES_URL      = "postgresql://taxi:Yaseen#321@localhost:5433/taxi_demand"
DOWNLOAD_TIMEOUT  = 600   # 10 min — large files on slow connections
CHUNK_SIZE        = 4 * 1024 * 1024  # 4 MB chunks


def months_to_load(n: int) -> list[tuple[int, int]]:
    today = date.today()
    result = []
    for i in range(n, 0, -1):
        m = today.month - i
        y = today.year
        while m <= 0:
            m += 12
            y -= 1
        result.append((y, m))
    return result


def already_loaded(engine) -> set[date]:
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT DISTINCT DATE_TRUNC('month', pickup_hour)::DATE FROM hourly_rides")
        ).fetchall()
    return {r[0] for r in rows}


def download_month(year: int, month: int) -> Path:
    """Stream-download parquet to data/raw/. Returns path. Skips if exists."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / f"yellow_tripdata_{year}-{month:02d}.parquet"
    if dest.exists():
        print(f"  Already downloaded: {dest.name}")
        return dest
    url = f"{TLC_BASE_URL}/yellow_tripdata_{year}-{month:02d}.parquet"
    print(f"  Downloading {url}")
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"  {pct:.0f}%  ({downloaded/1e6:.0f} / {total/1e6:.0f} MB)", end="\r")
    print(f"  Downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)          ")
    return dest


def validate_and_transform(path: Path, year: int, month: int) -> pd.DataFrame:
    from src.data.validation import validate_and_clean
    from src.data.transformation import to_hourly_time_series

    print(f"  Reading {path.name}...")
    raw_df = pd.read_parquet(path)
    print(f"  {len(raw_df):,} raw rows")

    clean_df, report = validate_and_clean(raw_df, year, month)
    del raw_df; gc.collect()
    print(f"  Validated: {len(clean_df):,} rows ({report.drop_rate:.1%} dropped)")

    ts = to_hourly_time_series(clean_df)
    del clean_df; gc.collect()
    print(f"  Time-series: {len(ts):,} rows")
    return ts


def load_to_postgres(ts: pd.DataFrame, engine) -> None:
    import psycopg2
    raw_conn = psycopg2.connect(
        host="localhost", port=5433, dbname="taxi_demand",
        user="taxi", password="Yaseen#321"
    )
    try:
        with raw_conn.cursor() as cur:
            buf = StringIO()
            ts[["pickup_location_id", "pickup_hour", "ride_count"]].to_csv(
                buf, index=False, header=False
            )
            buf.seek(0)
            cur.execute(
                "CREATE TEMP TABLE tmp_rides (LIKE hourly_rides INCLUDING ALL) ON COMMIT DROP"
            )
            cur.copy_from(
                buf, "tmp_rides", sep=",",
                columns=["pickup_location_id", "pickup_hour", "ride_count"]
            )
            cur.execute("""
                INSERT INTO hourly_rides (pickup_location_id, pickup_hour, ride_count)
                SELECT pickup_location_id, pickup_hour, ride_count FROM tmp_rides
                ON CONFLICT (pickup_location_id, pickup_hour)
                DO UPDATE SET ride_count = EXCLUDED.ride_count
            """)
        raw_conn.commit()
    finally:
        raw_conn.close()


def main() -> None:
    engine = create_engine(POSTGRES_URL)

    months = months_to_load(BACKFILL_MONTHS)
    print(f"Target: {len(months)} months  ({months[0][0]}-{months[0][1]:02d} to {months[-1][0]}-{months[-1][1]:02d})")

    loaded = already_loaded(engine)
    to_do = [(y, m) for y, m in months if datetime(y, m, 1).date() not in loaded]
    print(f"Already in DB: {len(loaded)} months")
    print(f"To load: {len(to_do)} months: {to_do}\n")

    for i, (year, month) in enumerate(to_do, 1):
        print(f"[{i}/{len(to_do)}] {year}-{month:02d}")
        ts = None
        try:
            path = download_month(year, month)
            ts = validate_and_transform(path, year, month)
            load_to_postgres(ts, engine)
            print(f"  OK Loaded {len(ts):,} rows for {year}-{month:02d}\n")
        except Exception as e:
            print(f"  ERROR: {e} -- skipping\n")
        finally:
            del ts; gc.collect()

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM hourly_rides")).scalar()
        row = conn.execute(
            text("SELECT MIN(pickup_hour), MAX(pickup_hour) FROM hourly_rides")
        ).fetchone()
    print(f"Done. DB has {total:,} rows  |  {row[0]}  →  {row[1]}")


if __name__ == "__main__":
    main()
