
from src.build_features import build_features

def create_baseline(config_path: str):
    """
    Creates a baseline dataset for monitoring, including features and true values.

    Args:
        config_path (str): Path to the main configuration file.
    """
    print("--- Creating Baseline Dataset for Monitoring ---")

    # --- 1. Load Config ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 2. Load Data and create baseline test set ---
    features_df = build_features(db_path=config["data"]["database_path"])
    X = features_df.drop(columns=[config["data"]["target_column"]])
    y = features_df[config["data"]["target_column"]]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
    )
    
    baseline_df = X_test.copy()
    baseline_df['RUL'] = y_test

    # --- 3. Save Baseline ---
    output_dir = "data/monitoring"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "baseline_dataset.csv")
    baseline_df.to_csv(output_path, index=False)

    print(f"Baseline dataset saved to: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/main_config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()
    create_baseline(args.config)
