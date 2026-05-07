"""
Feast feature store helpers: write to offline store, materialize to online store,
and read features for training and inference.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from feast import FeatureStore

from src.config import settings
from src.utils.logging_utils import logger


def get_feature_store() -> FeatureStore:
    return FeatureStore(repo_path=str(settings.feature_repo_dir))


def write_to_offline_store(df: pd.DataFrame, feature_view_name: str = "hourly_rides_view") -> None:
    """Push a batch of hourly ride counts into the Feast offline store (PostgreSQL)."""
    fs = get_feature_store()
    fs.write_to_offline_store(feature_view_name=feature_view_name, df=df)
    logger.info(f"Wrote {len(df):,} rows to offline store [{feature_view_name}]")


def materialize_to_online_store(start_dt: datetime | None = None) -> None:
    """Sync the most recent features from offline → Redis online store."""
    fs = get_feature_store()
    end_dt = datetime.now(tz=timezone.utc)
    fs.materialize_incremental(end_date=end_dt)
    logger.info(f"Materialized features to online store up to {end_dt.isoformat()}")


def get_online_features(location_ids: list[int]) -> pd.DataFrame:
    """
    Fetch the latest feature vector for each location from the Redis online store.
    Used by the inference pipeline for real-time predictions.
    """
    fs = get_feature_store()
    entity_rows = [{"pickup_location_id": loc_id} for loc_id in location_ids]
    feature_vector = fs.get_online_features(
        features=["hourly_rides_view:ride_count"],
        entity_rows=entity_rows,
    ).to_df()
    logger.info(f"Fetched online features for {len(location_ids)} locations")
    return feature_vector


def get_historical_features(entity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Point-in-time correct feature retrieval for model training.
    entity_df must have columns: pickup_location_id, event_timestamp.
    """
    fs = get_feature_store()
    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=["hourly_rides_view:ride_count"],
    ).to_df()
    logger.info(f"Retrieved historical features: {len(training_df):,} rows")
    return training_df
