import sqlite3
import pandas as pd
import os
from src.feature_engineering import apply_feature_engineering
from sklearn.preprocessing import PowerTransformer
import joblib

pd.set_option('future.no_silent_downcasting', True)

def build_features(df: pd.DataFrame, is_training_data: bool = False):
    """
    Builds features for the NASA Turbofan Engine Degradation dataset.
    This involves calculating the Remaining Useful Life (RUL) and creating
    time-series features from the sensor data.

    Args:
        df (pd.DataFrame): The input DataFrame to build features from.
        is_training_data (bool): If True, the RUL transformer will be fitted and saved.
                                 If False, the saved transformer will be loaded and applied.
    """
    # --- 1. Apply Feature Engineering (RUL calculation and rolling features) ---
    df = apply_feature_engineering(df)

    # --- 2. Apply/Save RUL Transformer ---
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)
    transformer_path = os.path.join(output_dir, "rul_transformer.joblib")

    if is_training_data:
        transformer = PowerTransformer()
        df['RUL'] = transformer.fit_transform(df[['RUL']])
        joblib.dump(transformer, transformer_path)
        print(f"RUL Transformer fitted and saved to: {transformer_path}")
    else:
        if not os.path.exists(transformer_path):
            raise FileNotFoundError(f"RUL Transformer not found at {transformer_path}. "
                                    "Ensure build_features was run with is_training_data=True first.")
        transformer = joblib.load(transformer_path)
        df['RUL'] = transformer.transform(df[['RUL']])
        print(f"RUL Transformer loaded from: {transformer_path} and applied.")

    return df