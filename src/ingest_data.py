
import os
import zipfile
import sqlite3
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

def ingest_data(output_db_path: str = "olist.sqlite"):
    """
    Downloads the Olist e-commerce dataset from Kaggle, unzips it,
    and loads the CSV files into a SQLite database.

    Args:
        output_db_path (str): The path to the output SQLite database file.
    """
    # --- 1. Authenticate and Download ---
    api = KaggleApi()
    api.authenticate()

    dataset = "olistbr/brazilian-ecommerce"
    download_path = "data/raw"
    
    print(f"Downloading dataset: {dataset} to {download_path}...")
    api.dataset_download_files(dataset, path=download_path, unzip=True)
    print("Download complete.")

    # --- 2. Load CSVs into SQLite ---
    db_conn = sqlite3.connect(output_db_path)
    cursor = db_conn.cursor()

    print(f"Loading data into SQLite database: {output_db_path}...")
    for file in os.listdir(download_path):
        if file.endswith(".csv"):
            table_name = os.path.splitext(file)[0].replace("olist_", "").replace("_dataset", "")
            df = pd.read_csv(os.path.join(download_path, file))
            df.to_sql(table_name, db_conn, if_exists="replace", index=False)
            print(f"  - Loaded table: {table_name}")

    db_conn.close()
    print("Data ingestion complete.")

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    ingest_data()
