"""
All model definitions for the NYC Taxi Demand forecasting project.
Each model is wrapped in a sklearn Pipeline for consistent train/predict API.
"""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.features.engineering import TemporalFeatureEngineer, RollingStatsEngineer


def _drop_id_cols(X):
    """Drop non-feature identifier columns before fitting."""
    drop_cols = [c for c in ["pickup_location_id", "pickup_hour"] if c in X.columns]
    return X.drop(columns=drop_cols)


_id_dropper = FunctionTransformer(_drop_id_cols)


def build_lgbm_pipeline(params: dict | None = None) -> Pipeline:
    """LightGBM with full feature engineering — primary production model."""
    lgbm_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    if params:
        lgbm_params.update(params)
    return Pipeline(
        [
            ("temporal", TemporalFeatureEngineer()),
            ("rolling", RollingStatsEngineer()),
            ("drop_ids", _id_dropper),
            ("model", LGBMRegressor(**lgbm_params)),
        ]
    )


def build_xgb_pipeline(params: dict | None = None) -> Pipeline:
    """XGBoost — challenger model for comparison experiments."""
    xgb_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
        "tree_method": "hist",  # fast on CPU
    }
    if params:
        xgb_params.update(params)
    return Pipeline(
        [
            ("temporal", TemporalFeatureEngineer()),
            ("rolling", RollingStatsEngineer()),
            ("drop_ids", _id_dropper),
            ("model", XGBRegressor(**xgb_params)),
        ]
    )


def build_baseline_pipeline() -> Pipeline:
    """
    Naive seasonal baseline: predict the average of the same hour
    from the last 4 weeks. Uses only the lag_168/336/504/672 columns.
    """
    from sklearn.base import BaseEstimator, RegressorMixin
    import numpy as np

    class SeasonalMeanRegressor(BaseEstimator, RegressorMixin):
        def fit(self, X, y=None):
            return self

        def predict(self, X):
            week_cols = [
                f"lag_{w * 168}" for w in range(1, 5) if f"lag_{w * 168}" in X.columns
            ]
            if not week_cols:
                return np.zeros(len(X))
            return X[week_cols].mean(axis=1).clip(lower=0).values

    return Pipeline(
        [
            ("drop_ids", _id_dropper),
            ("model", SeasonalMeanRegressor()),
        ]
    )


# Registry of all models available for experiments
MODEL_REGISTRY = {
    "lgbm": build_lgbm_pipeline,
    "xgboost": build_xgb_pipeline,
    "baseline": build_baseline_pipeline,
}
