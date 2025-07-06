
import os
import zipfile
import sqlite3
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def ingest_data(output_db_path: str = "turbofan.sqlite"):
    """
    Downloads the NASA Turbofan Engine Degradation dataset from Kaggle,
    unzips it, and loads the data into a SQLite database.

    Args:
        output_db_path (str): The path to the output SQLite database file.
    """
    # --- 1. Authenticate and Download ---
    api = KaggleApi()
    api.authenticate()

    dataset = "behrad3d/nasa-cmaps"
    download_path = "data/raw"
    
    print(f"Downloading dataset: {dataset} to {download_path}...")
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print("Download complete.")

    # --- 2. Load Data into SQLite ---
    db_conn = sqlite3.connect(output_db_path)
    
    print(f"Loading data into SQLite database: {output_db_path}...")
    
    # The dataset contains multiple text files (e.g., train_FD001.txt, test_FD001.txt, RUL_FD001.txt)
    # We will load the training data for now.
    train_file = "CMaps/train_FD001.txt"
    column_names = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + [f'sensor_{i}' for i in range(1, 22)]
    
    df = pd.read_csv(os.path.join(download_path, train_file), sep=' ', header=None, names=column_names)
    
    # The last two columns are empty, so we drop them
    df = df.drop(columns=[f'sensor_{i}' for i in range(22, 24) if f'sensor_{i}' in df.columns])
    
    df.to_sql("train_fd001", db_conn, if_exists="replace", index=False)
    print(f"  - Loaded table: train_fd001")

    db_conn.close()
    print("Data ingestion complete.")


