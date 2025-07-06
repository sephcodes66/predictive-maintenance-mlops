import os
import sqlite3
import pandas as pd
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi

def load_config(config_path="config/main_config.yaml"):
    """Loads the main configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ingest_data():
    """
    Downloads the NASA Turbofan Engine Degradation dataset from Kaggle,
    unzips it, and loads the data into a SQLite database.
    """
    print("--- Starting Data Ingestion Script ---")
    
    config = load_config()
    output_db_path = config["data"]["database_path"]
    
    print("--- 1. Authenticate and Download ---")
    try:
        print("Initializing Kaggle API...")
        api = KaggleApi()
        print("Authenticating with Kaggle...")
        api.authenticate()
        print("Authentication successful.")

        dataset = "behrad3d/nasa-cmaps"
        download_path = "data/raw"
        
        print(f"Downloading dataset: {dataset} to {download_path}...")
        api.dataset_download_files(dataset, path=download_path, unzip=True)
        print("Download complete.")

    except Exception as e:
        print(f"An error occurred during Kaggle API interaction: {e}")
        return

    print("--- 2. Load Data into SQLite ---")
    try:
        print(f"Connecting to database: {output_db_path}...")
        db_conn = sqlite3.connect(output_db_path)
        print("Database connection successful.")
        
        train_file = "CMaps/train_FD001.txt"
        train_file_path = os.path.join(download_path, train_file)
        print(f"Attempting to read file: {train_file_path}")
        
        if not os.path.exists(train_file_path):
            print(f"Error: File not found at {train_file_path}")
            db_conn.close()
            return
            
        print("Reading CSV file...")
        df = pd.read_csv(train_file_path, sep=r'''\s+''', header=None, engine='python', na_values=[''])
        print(f"Successfully read CSV. Initial DataFrame shape: {df.shape}")

        column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
        df = df.iloc[:, :26]
        df.columns = column_names
        
        print("Coercing sensor columns to numeric...")
        for col in [f'sensor_{i}' for i in range(1, 22)]:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Explicitly convert sensor_15 and sensor_16 to int
        if 'sensor_15' in df.columns:
            df['sensor_15'] = df['sensor_15'].fillna(0).astype(int)
        if 'sensor_16' in df.columns:
            df['sensor_16'] = df['sensor_16'].fillna(0).astype(int)
        
        print(f"Attempting to load {df.shape[0]} rows into train_fd001 table.")
        df.to_sql("train_fd001", db_conn, if_exists="replace", index=False)
        print(f"  - Loaded table: train_fd001")
        print(f"Table 'train_fd001' created successfully in {output_db_path}.")

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
    finally:
        if 'db_conn' in locals() and db_conn:
            db_conn.close()
            print("Database connection closed.")
    print("Data ingestion complete.")

if __name__ == "__main__":
    ingest_data()
