
import sqlite3
import pandas as pd
import os

def build_features(db_path: str = "turbofan.sqlite", output_dir: str = "data/processed"):
    """
    Builds features for the NASA Turbofan Engine Degradation dataset.
    This involves calculating the Remaining Useful Life (RUL) and creating
    time-series features from the sensor data.
    """
    # --- 1. Connect to the database ---
    conn = sqlite3.connect(db_path)

    # --- 2. Read the training data ---
    df = pd.read_sql("SELECT * FROM train_fd001", conn)

    # --- 3. Calculate RUL ---
    # The RUL is the number of cycles remaining before failure.
    # We can calculate this by subtracting the current cycle from the max cycle for each engine.
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    df = pd.merge(df, max_cycles, on='unit_number')
    df['RUL'] = df['max_cycles'] - df['time_in_cycles']
    df = df.drop(columns=['max_cycles'])

    # --- 4. Create time-series features ---
    # We can create rolling averages and standard deviations for the sensor data.
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    for window_size in [5, 10, 20]:
        for col in sensor_cols:
            df[f'{col}_rolling_mean_{window_size}'] = df.groupby('unit_number')[col].transform(lambda x: x.rolling(window_size).mean())
            df[f'{col}_rolling_std_{window_size}'] = df.groupby('unit_number')[col].transform(lambda x: x.rolling(window_size).std())

    # --- 5. Handle missing values ---
    # The rolling features will have missing values at the beginning of each time series.
    # We can fill these with the mean of the column.
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # --- 6. Save the feature table ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "turbofan_features.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Feature table saved to: {output_path}")

    conn.close()
