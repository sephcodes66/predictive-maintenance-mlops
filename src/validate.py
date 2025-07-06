import mlflow
import pandas as pd
import yaml
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

from src.build_features import build_features

def validate_model(config_path: str, output_dir: str):
    """
    Loads the latest tuned model, makes predictions on the test set,
    and validates the predictions against the ground truth.

    Args:
        config_path (str): Path to the main configuration file.
        output_dir (str): Directory to save validation results.
    """
    # --- 1. Load Config and Data ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data = build_features(db_path=config["data"]["database_path"])
    X = data.drop(columns=[config["data"]["target_column"]])
    y = data[config["data"]["target_column"]]

    # Use the same split to get the same test set used during training
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
    )

    # --- 2. Load Model from MLflow Registry ---
    model_name = f"{config['model']['name']}_tuned"
    model_uri = f"models:/{model_name}/latest"
    
    print(f"Loading latest model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # --- 3. Make Predictions on the Test Set ---
    y_pred = model.predict(X_test)

    # --- 4. Validate Predictions and Print Metrics ---
    print("\n--- Model Validation Results ---")
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R-squared: {r2:.4f}")

    # --- 5. Generate and Save Scatter Plot ---
    print("\nGenerating scatter plot...")
    os.makedirs(output_dir, exist_ok=True)
    scatter_path = os.path.join(output_dir, "validation_scatter_plot.png")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Actual RUL")
    plt.ylabel("Predicted RUL")
    plt.title("Actual vs. Predicted RUL")
    plt.savefig(scatter_path)
    plt.close()
    
    print(f"Scatter plot saved to: {scatter_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for churn model.")
    parser.add_argument("--config", type=str, default="config/main_config.yaml", help="Path to the main configuration file.")
    parser.add_argument("--output_dir", type=str, default="validation_results", help="Directory to save validation results.")
    args = parser.parse_args()

    validate_model(
        config_path=args.config,
        output_dir=args.output_dir
    )