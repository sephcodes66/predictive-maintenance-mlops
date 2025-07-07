
import mlflow
import yaml
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

from .visualize import plot_feature_importance, plot_actual_vs_predicted

from src.build_features import build_features

def train_model(config: dict):
    """
    Trains a single XGBoost regression model based on the provided configuration.

    Args:
        config (dict): A dictionary containing the configuration parameters.
    """

    # --- 1. Load Data ---
    processed_data_dir = config["data"]["processed_data_dir"]
    train_df = pd.read_csv(os.path.join(processed_data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_data_dir, "val.csv"))

    # --- 2. Apply Feature Engineering ---
    train_df_processed = build_features(train_df.copy(), is_training_data=True)
    val_df_processed = build_features(val_df.copy(), is_training_data=False)

    X_train = train_df_processed.drop(columns=[config["data"]["target_column"]])
    y_train = train_df_processed[config["data"]["target_column"]]

    X_val = val_df_processed.drop(columns=[config["data"]["target_column"]])
    y_val = val_df_processed[config["data"]["target_column"]]

    # --- 3. Train Model ---
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        mlflow.log_params(config["model"]["params"])

        model = xgb.XGBRegressor(**config["model"]["params"])
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # --- 4. Evaluate Model ---
        y_pred = model.predict(X_val)

        # Ensure specific columns are int64 for signature inference
        for col_name in ['unit_number', 'time_in_cycles', 'sensor_15', 'sensor_16', 'sensor_17', 'sensor_18']:
            if col_name in X_val.columns:
                X_val[col_name] = X_val[col_name].astype(int)

        signature = infer_signature(X_val, y_pred)

        # --- 5. Log Metrics ---
        mlflow.log_metric("mae", mean_absolute_error(y_val, y_pred))
        mlflow.log_metric("mse", mean_squared_error(y_val, y_pred))
        mlflow.log_metric("r2", r2_score(y_val, y_pred))

        # --- 6. Log Artifacts ---
        output_dir = os.path.abspath(config["visualization"]["output_dir"])
        print(f"[train.py] Plot output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        plot_feature_importance(model, os.path.join(output_dir, "feature_importance_baseline.png"), dataset_name="Validation Set")
        plot_actual_vs_predicted(
            y_val,
            y_pred,
            os.path.join(output_dir, "actual_vs_predicted_baseline.png"),
            dataset_name="Validation Set"
        )
        mlflow.log_artifacts(output_dir)

        # --- 7. Log Model ---
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_val.iloc[[0]],
            registered_model_name=config["model"]["name"],
        )

        print(f"Model logged to experiment: {config['mlflow']['experiment_name']}")
        print(f"Run ID: {run.info.run_id}")

        # --- 8. Save the run ID ---
        with open("mlruns/0/latest_run_id.txt", "w") as f:
            f.write(run.info.run_id)
