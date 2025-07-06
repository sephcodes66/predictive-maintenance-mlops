
import yaml
from src.ingest_data import ingest_data
from src.build_features import build_features
from src.train import train_model
from src.tune import tune_model

class Pipeline:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def run(self):
        ingest_data(self.config['data']['database_path'])
        build_features(self.config['data']['database_path'])
        train_model(self.config)
        tune_model(self.config)

if __name__ == '__main__':
    pipeline = Pipeline('config/main_config.yaml')
    pipeline.run()
