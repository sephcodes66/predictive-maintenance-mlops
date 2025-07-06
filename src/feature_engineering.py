import pandas as pd

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering steps to the input DataFrame.
    This includes calculating RUL (if not present) and creating
    time-series features from the sensor data.
    """
    # Calculate RUL if not already present (for training data)
    if 'RUL' not in df.columns:
        max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
        max_cycles.columns = ['unit_number', 'max_cycles']
        df = pd.merge(df, max_cycles, on='unit_number')
        df['RUL'] = df['max_cycles'] - df['time_in_cycles']
        df = df.drop(columns=['max_cycles'])

    # Create time-series features
    sensor_cols = [f'sensor_{i}' for i in range(1, 22)]
    new_features = []
    for window_size in [5, 10, 20]:
        for col in sensor_cols:
            mean_col = f'{col}_rolling_mean_{window_size}'
            std_col = f'{col}_rolling_std_{window_size}'
            
            mean_series = df.groupby('unit_number')[col].transform(lambda x: x.rolling(window_size).mean())
            std_series = df.groupby('unit_number')[col].transform(lambda x: x.rolling(window_size).std())
            
            mean_series.name = mean_col
            std_series.name = std_col
            
            new_features.append(mean_series)
            new_features.append(std_series)
            
    df = pd.concat([df] + new_features, axis=1)

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mean())

    # Convert all columns to numeric, handling specific types
    for col_name in ['sensor_15', 'sensor_16', 'sensor_17', 'sensor_18', 'time_in_cycles', 'unit_number']:
        if col_name in df.columns:
            df[col_name] = df[col_name].fillna(0).astype(int) # Fill NaN and convert to int

    for col in df.columns:
        # Convert remaining columns to numeric, coercing errors
        if df[col].dtype == 'object': # Only attempt conversion if not already numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df
