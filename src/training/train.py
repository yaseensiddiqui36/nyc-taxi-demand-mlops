"""
Main training entry point.
Trains a model, evaluates it, and logs everything to MLflow.
"""

from __future__ import annotations

import time
from typing import Literal

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.config import settings
from src.training.models import MODEL_REGISTRY
from src.utils.logging_utils import logger

ModelName = Literal["lgbm", "xgboost", "baseline"]


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: ModelName = "lgbm",
    model_params: dict | None = None,
    n_cv_splits: int = 3,
    experiment_name: str = "taxi_demand_forecasting",
) -> dict:
    """
    Train a model with time-series cross-validation and log to MLflow.

    Returns a dict with: mae, rmse, model_name, run_id, model_pipeline.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    build_fn = MODEL_REGISTRY[model_name]
    tscv = TimeSeriesSplit(n_splits=n_cv_splits)

    cv_maes, cv_rmses = [], []
    logger.info(f"Starting {n_cv_splits}-fold time-series CV for model='{model_name}'")

    with mlflow.start_run(run_name=f"{model_name}_{int(time.time())}") as run:
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_cv_splits", n_cv_splits)
        if model_params:
            mlflow.log_params(model_params)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline = build_fn(model_params)
            pipeline.fit(X_train, y_train)

            preds = pipeline.predict(X_val).clip(0)
            mae = mean_absolute_error(y_val, preds)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            cv_maes.append(mae)
            cv_rmses.append(rmse)
            logger.info(f"  Fold {fold + 1}: MAE={mae:.2f}, RMSE={rmse:.2f}")

        mean_mae = float(np.mean(cv_maes))
        mean_rmse = float(np.mean(cv_rmses))
        mlflow.log_metric("cv_mae", mean_mae)
        mlflow.log_metric("cv_rmse", mean_rmse)

        # Train final model on all data
        final_pipeline = build_fn(model_params)
        final_pipeline.fit(X, y)
        mlflow.sklearn.log_model(final_pipeline, artifact_path="model")

        logger.info(f"Training complete. CV MAE={mean_mae:.2f}, RMSE={mean_rmse:.2f}")

        return {
            "mae": mean_mae,
            "rmse": mean_rmse,
            "model_name": model_name,
            "run_id": run.info.run_id,
            "model_pipeline": final_pipeline,
        }


def run_optuna_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: ModelName = "lgbm",
    n_trials: int = 50,
    experiment_name: str = "taxi_demand_optuna",
) -> dict:
    """
    Hyperparameter tuning with Optuna. Returns best params and MAE.
    Requires `pip install optuna` (already in requirements.txt).
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tscv = TimeSeriesSplit(n_splits=3)
    build_fn = MODEL_REGISTRY[model_name]

    def objective(trial: optuna.Trial) -> float:
        params: dict = {}
        if model_name == "lgbm":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 16, 256),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            }
        elif model_name == "xgboost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            }

        maes = []
        for train_idx, val_idx in tscv.split(X):
            pipe = build_fn(params)
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = pipe.predict(X.iloc[val_idx]).clip(0)
            maes.append(mean_absolute_error(y.iloc[val_idx], preds))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize", study_name=f"tune_{model_name}")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(f"Optuna best MAE: {study.best_value:.2f}, params: {study.best_params}")
    return {"best_mae": study.best_value, "best_params": study.best_params}
