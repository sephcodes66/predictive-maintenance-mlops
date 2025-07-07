
import yaml
import pandas as pd
import os
from src.ingest_data import ingest_data
from src.data_splitter import split_data
from src.build_features import build_features
from src.train import train_model
from src.tune import tune_model

class Pipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def run(self):
        print("--- Running Full MLOps Pipeline ---")
        
        # 1. Data Ingestion
        ingest_data()

        # 2. Data Splitting
        split_data(self.config)

        # 3. Feature Engineering (for train, val, test)
        processed_data_dir = self.config["data"]["processed_data_dir"]
        
        train_df = pd.read_csv(os.path.join(processed_data_dir, "train.csv"))
        val_df = pd.read_csv(os.path.join(processed_data_dir, "val.csv"))
        test_df = pd.read_csv(os.path.join(processed_data_dir, "test.csv"))

        build_features(train_df.copy(), is_training_data=True)
        build_features(val_df.copy(), is_training_data=False)
        build_features(test_df.copy(), is_training_data=False)

        # 4. Model Training
        train_model(self.config)

        # 5. Hyperparameter Tuning
        tune_model(self.config)

        print("--- Full MLOps Pipeline Complete ---")
