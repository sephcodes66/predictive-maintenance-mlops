

import mlflow
import pandas as pd
import yaml
import os
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

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

    data = pd.read_parquet("data/processed/customer_features_realistic.parquet")
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
    y_pred_proba = model.predict(X_test) # Using predict instead of predict_proba for simplicity

    # --- 4. Validate Predictions and Print Metrics ---
    print("\n--- Model Validation Results ---")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")

    # --- 5. Generate and Save Confusion Matrix ---
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "validation_confusion_matrix.png")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('Validation Confusion Matrix')
    plt.savefig(cm_path)
    plt.close(fig)
    
    print(f"Confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validation script for churn model.")
    parser.add_argument("--config", type=str, default="config/main_config.yaml", help="Path to the main configuration file.")
    parser.add_argument("--output_dir", type=str, default="validation_results", help="Directory to save validation results.")
    args = parser.parse_args()

    validate_model(
        config_path=args.config,
        output_dir=args.output_dir
    )

