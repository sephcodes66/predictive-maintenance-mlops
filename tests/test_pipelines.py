
import os
import pandas as pd
from src.pipeline import Pipeline
from src.build_features import build_features
import sqlite3

def test_build_features():
    """
    Tests the build_features function with a dummy DataFrame.
    """
    # --- 1. Setup ---
    output_dir = "data/processed"
    transformer_path = os.path.join(output_dir, "rul_transformer.joblib")

    # Create a dummy DataFrame that mimics the structure of the raw data
    dummy_df = pd.DataFrame({
        'unit_number': [1, 1, 1, 2, 2],
        'time_in_cycles': [1, 2, 3, 1, 2],
        'op_setting_1': [0.1, 0.2, 0.3, 0.4, 0.5],
        'op_setting_2': [0.6, 0.7, 0.8, 0.9, 1.0],
        'op_setting_3': [100, 100, 100, 100, 100],
        'sensor_1': [641.8, 641.8, 641.8, 641.8, 641.8],
        'sensor_2': [641.8, 641.8, 641.8, 641.8, 641.8],
        'sensor_3': [1589.7, 1589.7, 1589.7, 1589.7, 1589.7],
        'sensor_4': [1400.0, 1400.0, 1400.0, 1400.0, 1400.0],
        'sensor_5': [14.62, 14.62, 14.62, 14.62, 14.62],
        'sensor_6': [21.61, 21.61, 21.61, 21.61, 21.61],
        'sensor_7': [554.36, 554.36, 554.36, 554.36, 554.36],
        'sensor_8': [2388.06, 2388.06, 2388.06, 2388.06, 2388.06],
        'sensor_9': [9046.19, 9046.19, 9046.19, 9046.19, 9046.19],
        'sensor_10': [1.3, 1.3, 1.3, 1.3, 1.3],
        'sensor_11': [47.2, 47.2, 47.2, 47.2, 47.2],
        'sensor_12': [521.7, 521.7, 521.7, 521.7, 521.7],
        'sensor_13': [2388.0, 2388.0, 2388.0, 2388.0, 2388.0],
        'sensor_14': [8138.6, 8138.6, 8138.6, 8138.6, 8138.6],
        'sensor_15': [8.4195, 8.4195, 8.4195, 8.4195, 8.4195],
        'sensor_16': [0.03, 0.03, 0.03, 0.03, 0.03],
        'sensor_17': [392, 392, 392, 392, 392],
        'sensor_18': [2388, 2388, 2388, 2388, 2388],
        'sensor_19': [100, 100, 100, 100, 100],
        'sensor_20': [38.86, 38.86, 38.86, 38.86, 38.86],
        'sensor_21': [23.3735, 23.3735, 23.3735, 23.3735, 23.3735],
    })

    # --- 2. Execution ---
    processed_df = build_features(dummy_df.copy(), is_training_data=True)

    # --- 3. Assertion ---
    assert 'RUL' in processed_df.columns, "RUL column not created"
    assert os.path.exists(transformer_path), f"Transformer file not found at {transformer_path}"
    
    # --- 4. Teardown ---
    os.remove(transformer_path)



def test_pipeline():
    """
    Tests the full pipeline with the test configuration.
    """
    # --- 1. Setup ---
    test_db_path = "turbofan_test.sqlite"
    test_processed_data_dir = "data/processed_test"
    test_visualizations_dir = "visualizations_test"

    # Create a dummy database for ingestion
    conn = sqlite3.connect(test_db_path)
    num_rows = 100 # Increased number of rows for splitting
    df = pd.DataFrame({
        'unit_number': [i % 10 + 1 for i in range(num_rows)], # More diverse unit numbers
        'time_in_cycles': [i % 20 + 1 for i in range(num_rows)],
        'op_setting_1': [0.1] * num_rows,
        'op_setting_2': [0.6] * num_rows,
        'op_setting_3': [100] * num_rows,
        'sensor_1': [641.8] * num_rows,
        'sensor_2': [641.8] * num_rows,
        'sensor_3': [1589.7] * num_rows,
        'sensor_4': [1400.0] * num_rows,
        'sensor_5': [14.62] * num_rows,
        'sensor_6': [21.61] * num_rows,
        'sensor_7': [554.36] * num_rows,
        'sensor_8': [2388.06] * num_rows,
        'sensor_9': [9046.19] * num_rows,
        'sensor_10': [1.3] * num_rows,
        'sensor_11': [47.2] * num_rows,
        'sensor_12': [521.7] * num_rows,
        'sensor_13': [2388.0] * num_rows,
        'sensor_14': [8138.6] * num_rows,
        'sensor_15': [8.4195] * num_rows,
        'sensor_16': [0.03] * num_rows,
        'sensor_17': [392] * num_rows,
        'sensor_18': [2388] * num_rows,
        'sensor_19': [100] * num_rows,
        'sensor_20': [38.86] * num_rows,
        'sensor_21': [23.3735] * num_rows,
    })
    df.to_sql("train_fd001", conn, if_exists="replace", index=False)
    conn.close()

    # --- 2. Execution ---
    try:
        pipeline = Pipeline("config/test_config.yaml")
        pipeline.run()
    except Exception as e:
        assert False, f"Pipeline failed with exception: {e}"
    finally:
        # --- 3. Teardown ---
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        if os.path.exists(os.path.join(test_processed_data_dir, "train.csv")):
            os.remove(os.path.join(test_processed_data_dir, "train.csv"))
        if os.path.exists(os.path.join(test_processed_data_dir, "val.csv")):
            os.remove(os.path.join(test_processed_data_dir, "val.csv"))
        if os.path.exists(os.path.join(test_processed_data_dir, "test.csv")):
            os.remove(os.path.join(test_processed_data_dir, "test.csv"))
        if os.path.exists(os.path.join(test_processed_data_dir, "rul_transformer.joblib")):
            os.remove(os.path.join(test_processed_data_dir, "rul_transformer.joblib"))
        
        # Clean up directories if empty
        if os.path.exists(test_processed_data_dir) and not os.listdir(test_processed_data_dir):
            os.rmdir(test_processed_data_dir)
        if os.path.exists(test_visualizations_dir) and not os.listdir(test_visualizations_dir):
            os.rmdir(test_visualizations_dir)
