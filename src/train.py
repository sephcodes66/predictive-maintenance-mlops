
import mlflow
import yaml
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .visualize import plot_feature_importance, plot_actual_vs_predicted

def train_model(config: dict):
    """
    Trains a single XGBoost regression model based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.
    """

    # --- 1. Load Data ---
    features_df = pd.read_parquet("data/processed/turbofan_features.parquet")
    X = features_df.drop(columns=[config["data"]["target_column"]])
    y = features_df[config["data"]["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
    )

    # --- 2. Train Model ---
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        mlflow.log_params(config["model"]["params"])

        model = xgb.XGBRegressor(**config["model"]["params"])
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # --- 3. Evaluate Model ---
        y_pred = model.predict(X_test)
        signature = infer_signature(X_test, y_pred)

        # --- 4. Log Metrics ---
        mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
        mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
        mlflow.log_metric("r2", r2_score(y_test, y_pred))

        # --- 5. Log Artifacts ---
        output_dir = config["visualization"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        plot_feature_importance(model, f"{output_dir}/feature_importance.png")
        plot_actual_vs_predicted(y_test, y_pred, f"{output_dir}/actual_vs_predicted.png")
        mlflow.log_artifacts(output_dir)

        # --- 6. Log Model ---
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_test.iloc[[0]],
            registered_model_name=config["model"]["name"],
        )

        print(f"Model logged to experiment: {config['mlflow']['experiment_name']}")
        print(f"Run ID: {run.info.run_id}")
