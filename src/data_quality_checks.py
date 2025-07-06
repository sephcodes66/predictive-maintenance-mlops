import pandas as pd
import pandera as pa
import sqlite3
import sys
import yaml

# Schema for the 'train_fd001' table
train_fd001_schema = pa.DataFrameSchema({
    "unit_number": pa.Column(int, required=True, description="Identifier for the engine unit."),
    "time_in_cycles": pa.Column(int, required=True, description="Time in operational cycles."),
    "op_setting_1": pa.Column(float, required=True, description="Operational setting 1."),
    "op_setting_2": pa.Column(float, required=True, description="Operational setting 2."),
    "op_setting_3": pa.Column(float, required=True, description="Operational setting 3."),
    "sensor_1": pa.Column(float, required=True, description="Sensor measurement 1."),
    "sensor_2": pa.Column(float, required=True, description="Sensor measurement 2."),
    "sensor_3": pa.Column(float, required=True, description="Sensor measurement 3."),
    "sensor_4": pa.Column(float, required=True, description="Sensor measurement 4."),
    "sensor_5": pa.Column(float, required=True, description="Sensor measurement 5."),
    "sensor_6": pa.Column(float, required=True, description="Sensor measurement 6."),
    "sensor_7": pa.Column(float, required=True, description="Sensor measurement 7."),
    "sensor_8": pa.Column(float, required=True, description="Sensor measurement 8."),
    "sensor_9": pa.Column(float, required=True, description="Sensor measurement 9."),
    "sensor_10": pa.Column(float, required=True, description="Sensor measurement 10."),
    "sensor_11": pa.Column(float, required=True, description="Sensor measurement 11."),
    "sensor_12": pa.Column(float, required=True, description="Sensor measurement 12."),
    "sensor_13": pa.Column(float, required=True, description="Sensor measurement 13."),
    "sensor_14": pa.Column(float, required=True, description="Sensor measurement 14."),
    "sensor_15": pa.Column(int, required=True, description="Sensor measurement 15."),
    "sensor_16": pa.Column(int, required=True, description="Sensor measurement 16."),
    "sensor_17": pa.Column(int, required=True, description="Sensor measurement 17."),
    "sensor_18": pa.Column(int, required=True, description="Sensor measurement 18."),
    "sensor_19": pa.Column(float, required=True, description="Sensor measurement 19."),
    "sensor_20": pa.Column(float, required=True, description="Sensor measurement 20."),
    "sensor_21": pa.Column(float, required=True, description="Sensor measurement 21."),
})

def load_config(config_path="config/main_config.yaml"):
    """Loads the main configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_data_quality_checks():
    """
    Runs data quality checks on the raw data tables.
    """
    print("--- Running Data Quality Checks ---")
    
    config = load_config()
    db_path = config["data"]["database_path"]
    
    # --- 1. Connect to the database ---
    conn = sqlite3.connect(db_path)

    # --- 2. Load Data ---
    train_df = pd.read_sql("SELECT * FROM train_fd001", conn)

    # --- 3. Run Validations ---
    try:
        print("\nValidating 'train_fd001' table...")
        train_fd001_schema.validate(train_df, lazy=True)
        print("'train_fd001' table is valid.")

    except pa.errors.SchemaErrors as err:
        print("\n--- Data Quality Checks Failed ---")
        print(err.failure_cases)
        print("\n--- Data Quality Checks Failed ---")
        sys.exit(1)

    finally:
        conn.close()

    print("\n--- All Data Quality Checks Passed ---")


if __name__ == "__main__":
    run_data_quality_checks()
