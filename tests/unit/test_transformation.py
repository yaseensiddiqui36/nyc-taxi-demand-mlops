"""Unit tests for time-series transformation logic."""

import pandas as pd
from src.data.transformation import to_hourly_time_series, to_features_and_target


def _make_clean_df(n_trips: int = 50):
    import numpy as np

    rng = pd.date_range("2024-03-01 00:00:00", periods=n_trips, freq="30min", tz="UTC")
    return pd.DataFrame(
        {
            "tpep_pickup_datetime": rng,
            "PULocationID": np.random.choice([132, 161, 1], size=n_trips),
        }
    )


def test_to_hourly_produces_complete_grid():
    df = _make_clean_df(100)
    ts = to_hourly_time_series(df)
    assert "pickup_location_id" in ts.columns
    assert "pickup_hour" in ts.columns
    assert "ride_count" in ts.columns
    # Every location should have the same number of hours
    counts = ts.groupby("pickup_location_id")["pickup_hour"].count()
    assert counts.nunique() == 1, "All locations must have the same number of hours"


def test_to_hourly_no_negative_counts():
    df = _make_clean_df(50)
    ts = to_hourly_time_series(df)
    assert (ts["ride_count"] >= 0).all()


def test_to_features_and_target_shape():
    # Build a simple 3-location, 700-hour time-series
    n_hours = 700
    locations = [1, 2, 3]
    hours = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    rows = [
        {"pickup_location_id": loc, "pickup_hour": h, "ride_count": 5}
        for loc in locations
        for h in hours
    ]
    ts = pd.DataFrame(rows)
    X, y = to_features_and_target(ts, window_hours=672)
    assert len(X) == len(y)
    assert len(X) > 0
    assert "lag_1" in X.columns
    assert "lag_672" in X.columns
    assert (y >= 0).all()
