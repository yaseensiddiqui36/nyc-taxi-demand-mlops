"""
Transforms validated raw trip records into hourly ride-count time-series.
One row per (pickup_location_id, pickup_hour) with 0-filled gaps.
"""

from __future__ import annotations

import pandas as pd

from src.utils.logging_utils import logger


def to_hourly_time_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate validated trips to hourly counts per location.
    Fills missing (location, hour) combinations with ride_count = 0.

    Returns DataFrame with columns:
        pickup_location_id  int
        pickup_hour         datetime64[ns, UTC]
        ride_count          int
    """
    df = df.copy()
    df["pickup_hour"] = (
        pd.to_datetime(df["tpep_pickup_datetime"], utc=True)
        .dt.floor("h")
    )

    agg = (
        df.groupby(["PULocationID", "pickup_hour"], observed=True)
        .size()
        .reset_index(name="ride_count")
        .rename(columns={"PULocationID": "pickup_location_id"})
    )

    # Build a complete grid: every location × every hour in the range
    all_locations = agg["pickup_location_id"].unique()
    all_hours = pd.date_range(
        start=agg["pickup_hour"].min(),
        end=agg["pickup_hour"].max(),
        freq="h",
        tz="UTC",
    )

    grid = pd.MultiIndex.from_product(
        [all_locations, all_hours],
        names=["pickup_location_id", "pickup_hour"],
    ).to_frame(index=False)

    full = grid.merge(agg, on=["pickup_location_id", "pickup_hour"], how="left")
    full["ride_count"] = full["ride_count"].fillna(0).astype(int)

    logger.info(
        f"Time-series: {len(full):,} rows, "
        f"{full['pickup_location_id'].nunique()} locations, "
        f"{full['pickup_hour'].nunique()} hours"
    )
    return full.sort_values(["pickup_location_id", "pickup_hour"]).reset_index(drop=True)


def to_features_and_target(
    ts: pd.DataFrame,
    window_hours: int = 672,
    step_hours: int = 1,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Create tabular features from time-series using a sliding window.

    For each row, the feature vector contains the previous `window_hours`
    ride counts for that location, and the target is the next hour's count.

    Returns (X, y) where X has columns lag_1 … lag_{window_hours}.
    """
    records = []
    for loc_id, group in ts.groupby("pickup_location_id", observed=True):
        group = group.sort_values("pickup_hour").reset_index(drop=True)
        rides = group["ride_count"].values
        hours = group["pickup_hour"].values

        for i in range(window_hours, len(rides) - step_hours + 1, step_hours):
            lags = {f"lag_{j}": rides[i - j] for j in range(1, window_hours + 1)}
            lags["pickup_location_id"] = loc_id
            lags["pickup_hour"] = hours[i]
            lags["target_rides"] = rides[i]
            records.append(lags)

    if not records:
        return pd.DataFrame(), pd.Series(dtype=float)

    df_out = pd.DataFrame(records)
    y = df_out.pop("target_rides")
    return df_out, y


def to_inference_features(
    ts: pd.DataFrame,
    window_hours: int = 672,
) -> pd.DataFrame:
    """
    Build the latest feature row for each location (for real-time inference).
    Returns one row per location with the most recent window_hours of lags.
    """
    records = []
    for loc_id, group in ts.groupby("pickup_location_id", observed=True):
        group = group.sort_values("pickup_hour").reset_index(drop=True)
        rides = group["ride_count"].values
        if len(rides) < window_hours:
            continue
        lags = {f"lag_{j}": rides[-(j)] for j in range(1, window_hours + 1)}
        lags["pickup_location_id"] = loc_id
        lags["pickup_hour"] = group["pickup_hour"].iloc[-1]
        records.append(lags)
    return pd.DataFrame(records)
