"""Fetches raw NYC TLC Yellow Taxi parquet files from the public AWS endpoint."""

from __future__ import annotations

import io
from datetime import date, datetime
from typing import Optional

import pandas as pd
import requests

from src.config import settings
from src.utils.logging_utils import logger


def _parquet_url(year: int, month: int) -> str:
    return f"{settings.nyc_tlc_base_url}/yellow_tripdata_{year}-{month:02d}.parquet"


def fetch_raw_trip_data(year: int, month: int) -> pd.DataFrame:
    """Download one month of raw yellow taxi trip data from NYC TLC."""
    import tempfile, os
    url = _parquet_url(year, month)
    logger.info(f"Fetching {year}-{month:02d} from {url}")
    # Stream to a temp file so we never hold the full ~200MB response in RAM twice
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with requests.get(url, timeout=300, stream=True) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    f.write(chunk)
        df = pd.read_parquet(tmp_path)
    finally:
        os.unlink(tmp_path)
    logger.info(f"Fetched {len(df):,} rows for {year}-{month:02d}")
    return df


def latest_available_month() -> tuple[int, int]:
    """
    Probe the TLC endpoint to find the most recent month with available data.
    TLC typically lags ~2 months behind the current date.
    """
    today = date.today()
    for lag in range(1, 6):
        month = today.month - lag
        year = today.year
        if month <= 0:
            month += 12
            year -= 1
        url = _parquet_url(year, month)
        try:
            resp = requests.head(url, timeout=15)
            if resp.status_code == 200:
                logger.info(f"Latest available month: {year}-{month:02d}")
                return year, month
        except requests.RequestException:
            continue
    # Fallback: 2 months ago
    month = today.month - 2
    year = today.year
    if month <= 0:
        month += 12
        year -= 1
    return year, month


def months_to_backfill(n_months: int) -> list[tuple[int, int]]:
    """Return the last n_months (year, month) pairs in ascending order."""
    today = date.today()
    result = []
    for i in range(n_months, 0, -1):
        month = today.month - i
        year = today.year
        while month <= 0:
            month += 12
            year -= 1
        result.append((year, month))
    return result
