
import mlflow
import yaml
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature
import numpy as np
import os
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from .visualize import plot_feature_importance, plot_confusion_matrix, plot_roc_curve

def train_model(config_path: str):
    """
    Trains a single XGBoost model based on the provided configuration.

    Args:
        config_path (str): Path to the main configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 1. Load Data ---
    features_df = pd.read_parquet("data/processed/customer_features_realistic.parquet")
    X = features_df.drop(columns=[config["data"]["target_column"]])
    y = features_df[config["data"]["target_column"]]

    if len(y.unique()) < 2:
        print("Warning: The dataset has only one class. Skipping training.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"], stratify=y
    )

    if len(y_train.unique()) < 2:
        print("Warning: The training set has only one class. Skipping training.")
        return

    # --- 2. Train Model ---
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run() as run:
        mlflow.log_params(config["model"]["params"])

        model = xgb.XGBClassifier(**config["model"]["params"])
        if len(y_test.unique()) > 1:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train, verbose=False)

        # --- 3. Evaluate Model ---
        if len(y_test.unique()) > 1:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            signature = infer_signature(X_test, y_pred_proba)

            # --- 4. Log Metrics ---
            mlflow.log_metric("auc", roc_auc_score(y_test, y_pred_proba))
            mlflow.log_metric("f1_score", f1_score(y_test, y_pred))
            mlflow.log_metric("precision", precision_score(y_test, y_pred))
            mlflow.log_metric("recall", recall_score(y_test, y_pred))

            # --- 5. Log Artifacts ---
            output_dir = config["visualization"]["output_dir"]
            os.makedirs(output_dir, exist_ok=True)
            plot_feature_importance(model, f"{output_dir}/feature_importance.png")
            plot_confusion_matrix(y_test, y_pred, f"{output_dir}/confusion_matrix.png")
            plot_roc_curve(y_test, y_pred_proba, f"{output_dir}/roc_curve.png")
            mlflow.log_artifacts(output_dir)
        else:
            print("Warning: Test set has only one class. Skipping evaluation.")
            signature = infer_signature(X_test, model.predict_proba(X_test)[:, 1])

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

if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    from src.visualize import plot_feature_importance
    train_model("config/main_config.yaml")
