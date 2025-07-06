

import pandas as pd
import yaml
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def monitor_performance(config_path: str, predictions_path: str, ground_truth_path: str):
    """
    Monitors model performance on new data.

    Args:
        config_path (str): Path to the main configuration file.
        predictions_path (str): Path to the CSV file with predictions.
        ground_truth_path (str): Path to the CSV file with ground truth.
    """
    print("--- Monitoring Model Performance ---")

    # --- 1. Load Config, Predictions and Ground Truth ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    predictions_df = pd.read_csv(predictions_path)
    ground_truth_df = pd.read_csv(ground_truth_path)

    # --- 2. Extract Predictions and Ground Truth ---
    y_pred = predictions_df["RUL_prediction"]
    y_true = ground_truth_df["RUL"]

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # --- 3. Save Performance Report ---
    performance_report = {
        "mse": mse,
        "mae": mae,
        "r2_score": r2
    }
    output_dir = "monitoring_results"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "performance_report.json")
    with open(report_path, 'w') as f:
        json.dump(performance_report, f, indent=4)
    
    print(f"Performance report saved to: {report_path}")

    # --- 4. Alert on Performance Degradation ---
    r2_threshold = config.get("monitoring", {}).get("r2_threshold", 0.7)
    if r2 < r2_threshold:
        print("\n--- PERFORMANCE DEGRADATION DETECTED ---")
        print(f"  - R2 score of {r2:.4f} is below the threshold of {r2_threshold}")
        # In a real pipeline, you might exit with a non-zero status code
        # sys.exit(1)
    else:
        print("\n--- No significant performance degradation detected. ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/main_config.yaml", help="Path to the configuration file.")
    parser.add_argument("--predictions", required=True, help="Path to the predictions CSV file.")
    parser.add_argument("--ground_truth", required=True, help="Path to the ground truth CSV file.")
    args = parser.parse_args()
    monitor_performance(args.config, args.predictions, args.ground_truth)

