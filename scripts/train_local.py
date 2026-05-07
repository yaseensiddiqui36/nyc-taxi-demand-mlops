"""
Local Training Script
─────────────────────
Runs outside Docker so it uses the full Windows host RAM, not the 6 GB
Docker container limit. Reads from PostgreSQL, builds features, trains
LightGBM, logs to the local MLflow server (localhost:5000).

Usage (from project root, .venv activated):
    python scripts/train_local.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np
import pandas as pd
import mlflow
from sqlalchemy import create_engine, text

POSTGRES_URL      = "postgresql://taxi:Yaseen#321@localhost:5433/taxi_demand"
MLFLOW_URI        = "http://localhost:5000"
TRAINING_DAYS     = 180   # how many days of history to train on
FEATURE_WINDOW    = 672   # lag hours (28 days)
EXPERIMENT_NAME   = "taxi_demand_forecasting"


def load_timeseries(engine) -> pd.DataFrame:
    print(f"Loading last {TRAINING_DAYS} days from hourly_rides...")
    with engine.connect() as conn:
        df = pd.read_sql(
            text(f"""
                SELECT pickup_location_id, pickup_hour, ride_count
                FROM hourly_rides
                WHERE pickup_hour >= NOW() - INTERVAL '{TRAINING_DAYS} days'
                ORDER BY pickup_location_id, pickup_hour
            """),
            conn,
        )
    print(f"  {len(df):,} rows, {df['pickup_location_id'].nunique()} zones")
    return df


def build_features(ts: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    print(f"Building lag features (window={FEATURE_WINDOW}h)...")
    print("  This may take several minutes on a large dataset.")

    records = []
    zones = ts["pickup_location_id"].unique()
    total = len(zones)

    for idx, (loc_id, group) in enumerate(ts.groupby("pickup_location_id", observed=True), 1):
        if idx % 50 == 0 or idx == total:
            print(f"  Zone {idx}/{total}  ({len(records):,} samples so far)")

        group = group.sort_values("pickup_hour").reset_index(drop=True)
        rides = group["ride_count"].values

        if len(rides) <= FEATURE_WINDOW:
            continue

        for i in range(FEATURE_WINDOW, len(rides)):
            lags = rides[i - FEATURE_WINDOW:i][::-1]   # lag_1 … lag_672
            records.append(
                [loc_id, group["pickup_hour"].iloc[i], rides[i]] + lags.tolist()
            )

        gc.collect()

    print(f"  Built {len(records):,} training samples")
    cols = ["pickup_location_id", "pickup_hour", "target"] + [f"lag_{j}" for j in range(1, FEATURE_WINDOW + 1)]
    df_out = pd.DataFrame(records, columns=cols)
    y = df_out.pop("target")
    df_out.drop(columns=["pickup_hour"], inplace=True)
    return df_out, y


def train(X: pd.DataFrame, y: pd.Series) -> dict:
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import TimeSeriesSplit
    import lightgbm as lgb

    print(f"\nTraining LightGBM on {len(X):,} samples x {len(X.columns)} features...")
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    tscv = TimeSeriesSplit(n_splits=3)
    cv_maes = []

    with mlflow.start_run(run_name="lgbm_local") as run:
        mlflow.log_params({
            "model": "lgbm",
            "training_days": TRAINING_DAYS,
            "feature_window": FEATURE_WINDOW,
            "n_cv_splits": 3,
        })

        for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
            model = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                n_jobs=-1,
                random_state=42,
                verbose=-1,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            preds = model.predict(X.iloc[val_idx]).clip(0)
            mae = mean_absolute_error(y.iloc[val_idx], preds)
            cv_maes.append(mae)
            print(f"  Fold {fold}/3  MAE={mae:.2f}")

        mean_mae = float(np.mean(cv_maes))
        mlflow.log_metric("cv_mae", mean_mae)

        # Final model on all data
        print("  Training final model on full dataset...")
        final = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            n_jobs=-1, random_state=42, verbose=-1,
        )
        final.fit(X, y)
        mlflow.sklearn.log_model(final, artifact_path="model")

        print(f"\nDone. CV MAE={mean_mae:.2f}  |  run_id={run.info.run_id}")
        return {"mae": mean_mae, "run_id": run.info.run_id, "model": final}


def main() -> None:
    import psutil
    available_gb = psutil.virtual_memory().available / 1e9
    print(f"Available RAM: {available_gb:.1f} GB")
    if available_gb < 6:
        print("WARNING: Less than 6 GB free — consider closing other apps or using Colab.")

    engine = create_engine(POSTGRES_URL)
    ts = load_timeseries(engine)

    X, y = build_features(ts)
    del ts; gc.collect()
    print(f"Feature matrix: {X.memory_usage(deep=True).sum() / 1e9:.2f} GB in RAM")

    result = train(X, y)
    del X, y; gc.collect()

    print(f"\nView in MLflow: http://localhost:5000")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Run ID: {result['run_id']}")
    print(f"CV MAE: {result['mae']:.2f} rides/hour")


if __name__ == "__main__":
    main()
