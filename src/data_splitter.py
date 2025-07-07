import pandas as pd
import sqlite3
import yaml
import os
from sklearn.model_selection import train_test_split

def load_config(config_path="config/main_config.yaml"):
    """Loads the main configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def split_data(config: dict):
    """
    Loads raw data, splits it into train, validation, and test sets,
    and saves them to processed data directory.
    """
    print("--- Starting Data Splitting Script ---")

    db_path = config["data"]["database_path"]
    processed_data_dir = config["data"]["processed_data_dir"]
    test_size = config["data"]["test_size"]
    val_size = config["data"]["val_size"]
    random_state = config["data"]["random_state"]

    os.makedirs(processed_data_dir, exist_ok=True)

    # --- 1. Load Data ---
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM train_fd001", conn)
    conn.close()

    # --- 2. Split Data ---
    # First, split into training and temp (validation + test)
    train_df, temp_df = train_test_split(
        df, test_size=(test_size + val_size), random_state=random_state,
        stratify=df['unit_number'] if 'unit_number' in df.columns else None
    )

    # Then, split temp into validation and test
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (test_size + val_size), random_state=random_state,
        stratify=temp_df['unit_number'] if 'unit_number' in temp_df.columns else None
    )

    # --- 3. Save Split Data ---
    train_df.to_csv(os.path.join(processed_data_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(processed_data_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(processed_data_dir, "test.csv"), index=False)

    print(f"Train data saved to: {os.path.join(processed_data_dir, 'train.csv')}")
    print(f"Validation data saved to: {os.path.join(processed_data_dir, 'val.csv')}")
    print(f"Test data saved to: {os.path.join(processed_data_dir, 'test.csv')}")
    print("--- Data Splitting Complete ---")

if __name__ == "__main__":
    config = load_config()
    split_data(config)
