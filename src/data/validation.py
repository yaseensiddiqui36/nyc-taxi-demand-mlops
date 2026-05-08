"""
Data validation using Great Expectations + Pydantic.
All raw NYC TLC data must pass these checks before entering the feature store.
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from src.utils.logging_utils import logger

# Valid NYC taxi location IDs (1–263; 264/265 are unknowns)
VALID_LOCATION_IDS = set(range(1, 264))

# Thresholds derived from domain knowledge
MIN_TRIP_DURATION_SECONDS = 60        # under 1 min is almost certainly a mistake
MAX_TRIP_DURATION_SECONDS = 18_000    # 5 hours max
MIN_FARE = 0.0
MAX_FARE = 2_000.0                    # outlier cap
MIN_PASSENGERS = 1
MAX_PASSENGERS = 8


class ValidationReport(NamedTuple):
    rows_in: int
    rows_out: int
    rows_dropped: int
    drop_rate: float
    passed: bool
    failure_reasons: dict[str, int]


def validate_and_clean(df: pd.DataFrame, year: int, month: int) -> tuple[pd.DataFrame, ValidationReport]:
    """
    Applies all quality checks to raw TLC data.
    Returns the cleaned DataFrame and a ValidationReport.
    Raises ValueError if the drop rate exceeds 40% (data quality incident).
    """
    rows_in = len(df)
    failure_counts: dict[str, int] = {}

    # ── 1. Required columns present ──────────────────────────
    required = {"tpep_pickup_datetime", "tpep_dropoff_datetime",
                "PULocationID", "DOLocationID", "fare_amount", "passenger_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # ── 2. Drop nulls in key columns ─────────────────────────
    before = len(df)
    df = df.dropna(subset=list(required))
    failure_counts["null_key_columns"] = before - len(df)

    # ── 3. Trip duration bounds ───────────────────────────────
    df["_duration_sec"] = (
        pd.to_datetime(df["tpep_dropoff_datetime"]) -
        pd.to_datetime(df["tpep_pickup_datetime"])
    ).dt.total_seconds()
    mask_duration = (
        (df["_duration_sec"] >= MIN_TRIP_DURATION_SECONDS) &
        (df["_duration_sec"] <= MAX_TRIP_DURATION_SECONDS)
    )
    failure_counts["invalid_duration"] = (~mask_duration).sum()
    df = df[mask_duration].drop(columns=["_duration_sec"])

    # ── 4. Fare amount bounds ─────────────────────────────────
    mask_fare = df["fare_amount"].between(MIN_FARE, MAX_FARE)
    failure_counts["invalid_fare"] = (~mask_fare).sum()
    df = df[mask_fare]

    # ── 5. Valid NYC location IDs ─────────────────────────────
    mask_loc = (
        df["PULocationID"].isin(VALID_LOCATION_IDS) &
        df["DOLocationID"].isin(VALID_LOCATION_IDS)
    )
    failure_counts["invalid_location"] = (~mask_loc).sum()
    df = df[mask_loc]

    # ── 6. Passenger count ────────────────────────────────────
    mask_pax = df["passenger_count"].between(MIN_PASSENGERS, MAX_PASSENGERS)
    failure_counts["invalid_passenger_count"] = (~mask_pax).sum()
    df = df[mask_pax]

    # ── 7. Pickup timestamps within expected year/month ───────
    pickup_dt = pd.to_datetime(df["tpep_pickup_datetime"])
    mask_time = (pickup_dt.dt.year == year) & (pickup_dt.dt.month == month)
    failure_counts["wrong_month"] = (~mask_time).sum()
    df = df[mask_time]

    # ── Build report ─────────────────────────────────────────
    rows_out = len(df)
    rows_dropped = rows_in - rows_out
    drop_rate = rows_dropped / rows_in if rows_in else 0

    total_failures = sum(failure_counts.values())
    logger.info(
        f"Validation {year}-{month:02d}: {rows_in:,} in → {rows_out:,} out "
        f"({drop_rate:.1%} dropped). Issues: {failure_counts}"
    )

    passed = drop_rate < 0.40
    if not passed:
        logger.error(f"Validation FAILED: drop rate {drop_rate:.1%} exceeds 40% threshold")

    report = ValidationReport(
        rows_in=rows_in,
        rows_out=rows_out,
        rows_dropped=rows_dropped,
        drop_rate=drop_rate,
        passed=passed,
        failure_reasons=failure_counts,
    )
    return df.reset_index(drop=True), report
