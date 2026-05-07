"""
Feature engineering: temporal features, rolling statistics, holiday flags.
Applied on top of the raw lag features before model training/inference.

Planned additions (Phase 4):
  - Location cluster features (borough, zone type: airport/downtown/residential)
  - Cross-zone features (demand at adjacent/correlated zones)
  - Weather features (temperature, precipitation via Open-Meteo free API)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as holidays_lib  # pip install holidays
    _HOLIDAYS_AVAILABLE = True
except ImportError:
    _HOLIDAYS_AVAILABLE = False

from sklearn.base import BaseEstimator, TransformerMixin


def _is_us_holiday(dt: pd.Timestamp) -> int:
    """Return 1 if the date is a US federal holiday (handles floating holidays)."""
    if _HOLIDAYS_AVAILABLE:
        us = holidays_lib.US(years=dt.year)
        return int(dt.date() in us)
    # Fallback: fixed-date federal holidays only
    _FIXED = {(1, 1), (7, 4), (11, 11), (12, 25)}
    return int((dt.month, dt.day) in _FIXED)


class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds hour-of-day, day-of-week, month, is_weekend, is_holiday columns."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dt = pd.to_datetime(X["pickup_hour"], utc=True)
        X["hour_of_day"] = dt.dt.hour
        X["day_of_week"] = dt.dt.dayofweek          # 0=Monday
        X["month"] = dt.dt.month
        X["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        X["is_holiday"] = dt.apply(_is_us_holiday)
        # Cyclical encoding for hour and day (avoids discontinuity at midnight/week-boundary)
        X["hour_sin"] = np.sin(2 * np.pi * X["hour_of_day"] / 24)
        X["hour_cos"] = np.cos(2 * np.pi * X["hour_of_day"] / 24)
        X["dow_sin"] = np.sin(2 * np.pi * X["day_of_week"] / 7)
        X["dow_cos"] = np.cos(2 * np.pi * X["day_of_week"] / 7)
        return X.drop(columns=["pickup_hour"])


class RollingStatsEngineer(BaseEstimator, TransformerMixin):
    """
    Adds rolling-mean features: avg rides over last 1/2/4 weeks
    at the same hour (uses lag_168, lag_336, lag_672 columns).
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # lag_168 = same hour 1 week ago, lag_336 = 2 weeks, lag_504 = 3, lag_672 = 4
        for weeks, lag in [(1, 168), (2, 336), (3, 504), (4, 672)]:
            col = f"lag_{lag}"
            if col in X.columns:
                X[f"avg_rides_{weeks}w_ago"] = X[col]

        # Simple 4-week average
        week_cols = [f"lag_{w * 168}" for w in range(1, 5) if f"lag_{w * 168}" in X.columns]
        if week_cols:
            X["avg_rides_last_4w"] = X[week_cols].mean(axis=1)

        return X


def add_all_features(X: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering steps in sequence."""
    X = TemporalFeatureEngineer().transform(X)
    X = RollingStatsEngineer().transform(X)
    return X
