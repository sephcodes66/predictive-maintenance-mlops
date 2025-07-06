import sqlite3
import pandas as pd
import os
from src.feature_engineering import apply_feature_engineering
from sklearn.preprocessing import PowerTransformer
import joblib

pd.set_option('future.no_silent_downcasting', True)

def build_features(db_path: str = "turbofan.sqlite", input_df: pd.DataFrame = None):
    """
    Builds features for the NASA Turbofan Engine Degradation dataset.
    This involves calculating the Remaining Useful Life (RUL) and creating
    time-series features from the sensor data.
    """
    # --- 1. Read the training data or use provided DataFrame ---
    if input_df is not None:
        df = input_df.copy()
    else:
        conn = sqlite3.connect(db_path)
        df = pd.read_sql("SELECT * FROM train_fd001", conn)
        conn.close()

    # --- 2. Apply Feature Engineering ---
    df = apply_feature_engineering(df)

    # --- 3. Save the transformer (only if processing original training data) ---
    if input_df is None: # Only save transformer when processing the main training data
        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)
        transformer = PowerTransformer()
        df['RUL'] = transformer.fit_transform(df[['RUL']])
        transformer_path = os.path.join(output_dir, "rul_transformer.joblib")
        joblib.dump(transformer, transformer_path)
        print(f"Transformer saved to: {transformer_path}")

    return df