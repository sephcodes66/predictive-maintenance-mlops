
import os
import pandas as pd
from src.pipeline import Pipeline
from src.build_features import build_features
import sqlite3

def test_build_features():
    """
    Tests the build_features function to ensure it creates the feature file and the transformer file.
    """
    # --- 1. Setup ---
    db_path = "turbofan_test.sqlite"
    output_dir = "data/processed"
    feature_path = os.path.join(output_dir, "turbofan_features.parquet")
    transformer_path = os.path.join(output_dir, "rul_transformer.joblib")
    
    # Create a dummy database
    conn = sqlite3.connect(db_path)
    df = pd.DataFrame({
        'unit_number': [1, 1, 1, 2, 2],
        'time_in_cycles': [1, 2, 3, 1, 2],
        'op_setting_1': [1, 1, 1, 1, 1],
        'op_setting_2': [1, 1, 1, 1, 1],
        'op_setting_3': [1, 1, 1, 1, 1],
        'sensor_1': [1, 1, 1, 1, 1],
        'sensor_2': [1, 1, 1, 1, 1],
        'sensor_3': [1, 1, 1, 1, 1],
        'sensor_4': [1, 1, 1, 1, 1],
        'sensor_5': [1, 1, 1, 1, 1],
        'sensor_6': [1, 1, 1, 1, 1],
        'sensor_7': [1, 1, 1, 1, 1],
        'sensor_8': [1, 1, 1, 1, 1],
        'sensor_9': [1, 1, 1, 1, 1],
        'sensor_10': [1, 1, 1, 1, 1],
        'sensor_11': [1, 1, 1, 1, 1],
        'sensor_12': [1, 1, 1, 1, 1],
        'sensor_13': [1, 1, 1, 1, 1],
        'sensor_14': [1, 1, 1, 1, 1],
        'sensor_15': [1, 1, 1, 1, 1],
        'sensor_16': [1, 1, 1, 1, 1],
        'sensor_17': [1, 1, 1, 1, 1],
        'sensor_18': [1, 1, 1, 1, 1],
        'sensor_19': [1, 1, 1, 1, 1],
        'sensor_20': [1, 1, 1, 1, 1],
        'sensor_21': [1, 1, 1, 1, 1],
    })
    df.to_sql("train_fd001", conn, if_exists="replace", index=False)
    conn.close()

    # --- 2. Execution ---
    build_features(db_path=db_path)

    # --- 3. Assertion ---
    assert os.path.exists(feature_path), f"Feature file not found at {feature_path}"
    assert os.path.exists(transformer_path), f"Transformer file not found at {transformer_path}"
    
    # --- 4. Teardown ---
    os.remove(db_path)
    os.remove(feature_path)
    os.remove(transformer_path)

def test_pipeline():
    """
    Tests the full pipeline with the test configuration.
    """
    try:
        pipeline = Pipeline("config/test_config.yaml")
        pipeline.run()
    except Exception as e:
        assert False, f"Pipeline failed with exception: {e}"
