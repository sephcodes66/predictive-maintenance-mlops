

import mlflow
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def evaluate_models(config_path: str):
    """
    Loads the trained models and evaluates their accuracy on the test set.

    Args:
        config_path (str): Path to the main configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 1. Load Data ---
    features_df = pd.read_parquet(config["data"]["feature_path"])
    X = features_df.drop(columns=[config["data"]["target_column"], "customer_unique_id", "last_purchase_date"])
    y = features_df[config["data"]["target_column"]]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"], stratify=y
    )

    # --- 2. Load Models ---
    client = mlflow.tracking.MlflowClient()

    baseline_model_name = config['model']['name']
    tuned_model_name = f"{config['model']['name']}_tuned"

    latest_baseline_version = client.get_latest_versions(baseline_model_name, stages=["None"])[0].version
    latest_tuned_version = client.get_latest_versions(tuned_model_name, stages=["None"])[0].version

    baseline_model_uri = f"models:/{baseline_model_name}/{latest_baseline_version}"
    tuned_model_uri = f"models:/{tuned_model_name}/{latest_tuned_version}"

    baseline_model = mlflow.pyfunc.load_model(baseline_model_uri)
    tuned_model = mlflow.pyfunc.load_model(tuned_model_uri)

    # --- 3. Make Predictions ---
    baseline_preds = baseline_model.predict(X_test)
    tuned_preds = tuned_model.predict(X_test)

    # --- 4. Calculate and Print Accuracy ---
    baseline_accuracy = accuracy_score(y_test, baseline_preds)
    tuned_accuracy = accuracy_score(y_test, tuned_preds)

    print("\n--- Model Accuracy ---")
    print(f"Baseline Model Accuracy: {baseline_accuracy:.4f}")
    print(f"Tuned Model Accuracy:    {tuned_accuracy:.4f}")
    print("----------------------\n")

if __name__ == "__main__":
    evaluate_models("config/main_config.yaml")

