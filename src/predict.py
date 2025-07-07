import mlflow
import pandas as pd
import yaml
import os
import joblib
from src.build_features import build_features

def predict(config_path: str, input_csv_path: str, output_csv_path: str):
    """
    Loads the latest tuned model and makes predictions on new data.

    Args:
        config_path (str): Path to the main configuration file.
        input_csv_path (str): Path to the CSV file with new data.
        output_csv_path (str): Path to save the predictions.
    """
    # --- 1. Load Config ---
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 2. Load Model from MLflow Registry ---
    model_name = f"{config['model']['name']}_tuned"
    model_uri = f"models:/{model_name}/latest"
    
    print(f"Loading latest model: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # --- 3. Load and Prepare New Data ---
    new_data = pd.read_csv(input_csv_path)
    
    # --- 4. Apply Feature Engineering ---
    # The build_features function will apply feature engineering and RUL transformation
    # We pass is_training_data=False to ensure it loads the pre-fitted transformer
    new_data_processed = build_features(new_data.copy(), is_training_data=False)

    # --- 5. Make Predictions ---
    # Drop the RUL column from the processed data before predicting, as it's the target
    X_new = new_data_processed.drop(columns=[config["data"]["target_column"]])
    predictions = model.predict(X_new)

    # --- 6. Inverse Transform Predictions ---
    transformer_path = os.path.join("data/processed", "rul_transformer.joblib")
    transformer = joblib.load(transformer_path)
    predictions = transformer.inverse_transform(predictions.reshape(-1, 1))

    # --- 7. Save Predictions ---
    prediction_df = pd.DataFrame(predictions, columns=['RUL_prediction'])
    # Concatenate original new_data with predictions
    output_df = pd.concat([new_data.reset_index(drop=True), prediction_df], axis=1)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"Predictions saved to: {output_csv_path}")
    print("\n--- Sample Predictions ---")
    print(output_df.head())

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="config/main_config.yaml")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    predict(args.config_path, args.input_csv, args.output_csv)
