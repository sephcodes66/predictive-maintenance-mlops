import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import yaml

def monitor_data_drift(new_data_path: str, reference_data_path: str, config_path: str):
    """
    Monitors for data drift by comparing the distribution of new data to a reference dataset.
    Uses the Kolmogorov-Smirnov (KS) test to compare the distributions of each sensor.

    Args:
        new_data_path (str): Path to the new data (CSV).
        reference_data_path (str): Path to the reference data (Parquet).
        config_path (str): Path to the main configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 1. Load Data ---
    new_df = pd.read_csv(new_data_path)
    ref_df = pd.read_parquet(reference_data_path)

    # --- 2. Compare Distributions ---
    for col in new_df.columns:
        if col in ref_df.columns and col.startswith("sensor"):
            ks_stat, p_value = ks_2samp(new_df[col], ref_df[col])
            print(f"Drift detection for {col}:")
            print(f"  KS Statistic: {ks_stat:.4f}")
            print(f"  P-value: {p_value:.4f}")
            if p_value < 0.05:
                print("  Drift detected!")
            else:
                print("  No drift detected.")

if __name__ == "__main__":
    # Example usage (replace with actual paths as needed)
    monitor_data_drift(
        "path/to/your/new_data.csv",
        "data/processed/turbofan_features.parquet",
        "config/main_config.yaml"
    )