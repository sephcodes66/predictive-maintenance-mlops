import pandas as pd
import numpy as np
import yaml
import os
import json
from src.build_features import build_features

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    """
    Calculate the Population Stability Index (PSI) for a single column.

    Args:
        expected (pd.Series): The baseline series.
        actual (pd.Series): The series to compare.
        buckettype (str): 'bins' for equal-width bins, 'quantiles' for equal-frequency bins.
        buckets (int): Number of buckets to use.
        axis (int): The axis to operate on.

    Returns:
        float: The PSI value.
    """
    
    def get_buckets(data, buckettype, buckets):
        if buckettype == 'bins':
            return pd.cut(data, bins=buckets)
        elif buckettype == 'quantiles':
            return pd.qcut(data, q=buckets, duplicates='drop')

    expected_percents = get_buckets(expected, buckettype, buckets).value_counts(normalize=True)
    actual_percents = get_buckets(actual, buckettype, buckets).value_counts(normalize=True)

    # Align the series to handle missing buckets
    all_buckets = pd.Series(index=expected_percents.index.union(actual_percents.index)).fillna(0)
    expected_percents = expected_percents.add(all_buckets, fill_value=0)
    actual_percents = actual_percents.add(all_buckets, fill_value=0)

    # Replace 0s with a small number to avoid division by zero
    expected_percents = expected_percents.replace(0, 0.0001)
    actual_percents = actual_percents.replace(0, 0.0001)

    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return np.sum(psi_values)

def monitor_drift(config_path: str, new_data_path: str):
    """
    Monitors data drift by comparing new data to a baseline.

    Args:
        config_path (str): Path to the main configuration file.
        new_data_path (str): Path to the new data to be monitored.
    """
    print("--- Monitoring Data Drift ---")

    # --- 1. Load Config and Baseline ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    baseline_df = pd.read_csv("data/monitoring/baseline_dataset.csv")
    new_data_raw_df = pd.read_csv(new_data_path)

    # --- 2. Apply Feature Engineering for new data ---
    new_data_df = build_features(db_path=config["data"]["database_path"], input_df=new_data_raw_df)

    # --- 3. Calculate PSI for each feature ---
    drift_report = {}
    for col in baseline_df.columns:
        if col != 'RUL':
            baseline_series = baseline_df[col].dropna()
            new_data_series = new_data_df[col].dropna()
            if not baseline_series.empty and not new_data_series.empty:
                psi = calculate_psi(baseline_series, new_data_series)
                drift_report[col] = psi
                print(f"PSI for {col}: {psi:.4f}")

    # --- 3. Save Drift Report ---
    output_dir = "monitoring_results"
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "drift_report.json")
    with open(report_path, 'w') as f:
        json.dump(drift_report, f, indent=4)
    
    print(f"Drift report saved to: {report_path}")

    # --- 4. Alert on Drift ---
    drift_threshold = config.get("monitoring", {}).get("drift_threshold", 0.25)
    drift_detected = any(psi > drift_threshold for psi in drift_report.values())

    if drift_detected:
        print("\n--- DATA DRIFT DETECTED ---")
        for col, psi in drift_report.items():
            if psi > drift_threshold:
                print(f"  - {col} has a PSI of {psi:.4f}, which is above the threshold of {drift_threshold}")
        # In a real pipeline, you might exit with a non-zero status code
        # sys.exit(1)
    else:
        print("\n--- No significant data drift detected. ---")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/main_config.yaml", help="Path to the configuration file.")
    parser.add_argument("--new_data", required=True, help="Path to the new data CSV file.")
    args = parser.parse_args()
    monitor_drift(args.config, args.new_data)