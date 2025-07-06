

import mlflow
import pandas as pd
import yaml
import os
import argparse
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

def monitor_model(config_path: str, holdout_data_path: str, output_dir: str):
    """
    Loads the latest tuned model, makes predictions on a holdout set,
    and logs performance metrics to a dedicated monitoring experiment.

    Args:
        config_path (str): Path to the main configuration file.
        holdout_data_path (str): Path to the holdout ground truth data.
        output_dir (str): Directory to save monitoring results.
    """
    # --- 1. Load Config and Holdout Data ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    holdout_df = pd.read_parquet(holdout_data_path)
    X_holdout = holdout_df.drop(columns=[config["data"]["target_column"], "customer_unique_id", "last_purchase_date", "first_purchase_date"])
    y_holdout = holdout_df[config["data"]["target_column"]]

    # --- 2. Load Model from MLflow Registry ---
    model_name = f"{config['model']['name']}_tuned"
    model_uri = f"models:/{model_name}/latest"
    
    print(f"Loading latest model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # --- 3. Make Predictions on the Holdout Set ---
    y_pred = model.predict(X_holdout)
    y_pred_proba = model.predict(X_holdout) # Using predict for simplicity

    # --- 4. Set up Monitoring Experiment and Log Metrics ---
    monitoring_experiment_name = "Model Monitoring"
    mlflow.set_experiment(monitoring_experiment_name)

    with mlflow.start_run() as run:
        print(f"Logging metrics to experiment: {monitoring_experiment_name}")
        
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred)
        recall = recall_score(y_holdout, y_pred)
        f1 = f1_score(y_holdout, y_pred)
        auc = roc_auc_score(y_holdout, y_pred_proba)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc", auc)

        print("\n--- Model Monitoring Results ---")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")

        # --- 5. Generate and Save Confusion Matrix ---
        print("\nGenerating confusion matrix for monitoring run...")
        cm = confusion_matrix(y_holdout, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        
        os.makedirs(output_dir, exist_ok=True)
        cm_path = os.path.join(output_dir, "monitoring_confusion_matrix.png")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues')
        ax.set_title('Monitoring Confusion Matrix')
        plt.savefig(cm_path)
        plt.close(fig)
        
        mlflow.log_artifact(cm_path)
        print(f"Confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model monitoring script.")
    parser.add_argument("--config", type=str, default="config/main_config.yaml", help="Path to the main configuration file.")
    parser.add_argument("--holdout_data", type=str, default="data/processed/holdout_ground_truth.parquet", help="Path to the holdout ground truth data.")
    parser.add_argument("--output_dir", type=str, default="monitoring_results", help="Directory to save monitoring results.")
    args = parser.parse_args()

    monitor_model(
        config_path=args.config,
        holdout_data_path=args.holdout_data,
        output_dir=args.output_dir
    )

