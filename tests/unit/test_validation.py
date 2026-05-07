"""Unit tests for data validation logic."""

import pandas as pd
import pytest
from src.data.validation import validate_and_clean


def _make_df(**overrides):
    """Build a minimal valid trip DataFrame."""
    base = {
        "tpep_pickup_datetime": ["2024-03-15 10:00:00"] * 10,
        "tpep_dropoff_datetime": ["2024-03-15 10:15:00"] * 10,
        "PULocationID": [132] * 10,
        "DOLocationID": [161] * 10,
        "fare_amount": [12.5] * 10,
        "passenger_count": [1] * 10,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_clean_data_passes():
    df, report = validate_and_clean(_make_df(), 2024, 3)
    assert report.passed
    assert report.rows_out == 10
    assert report.rows_dropped == 0


def test_invalid_location_dropped():
    df = _make_df(PULocationID=[999] * 10)
    _, report = validate_and_clean(df, 2024, 3)
    assert report.failure_reasons["invalid_location"] == 10
    assert report.rows_out == 0


def test_negative_fare_dropped():
    df = _make_df(fare_amount=[-5.0] * 10)
    _, report = validate_and_clean(df, 2024, 3)
    assert report.failure_reasons["invalid_fare"] == 10


def test_zero_duration_dropped():
    df = _make_df(tpep_dropoff_datetime=["2024-03-15 10:00:00"] * 10)
    _, report = validate_and_clean(df, 2024, 3)
    assert report.failure_reasons["invalid_duration"] == 10


def test_wrong_month_dropped():
    df = _make_df(tpep_pickup_datetime=["2024-01-15 10:00:00"] * 10)
    _, report = validate_and_clean(df, 2024, 3)
    assert report.failure_reasons["wrong_month"] == 10


def test_missing_columns_raises():
    df = pd.DataFrame({"fare_amount": [10.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_and_clean(df, 2024, 3)
