

import mlflow
import pandas as pd
import yaml
import os
import argparse

def predict(config_path: str, input_csv_path: str, output_csv_path: str):
    """
    Loads the latest tuned model and makes predictions on new data.

    Args:
        config_path (str): Path to the main configuration file.
        input_csv_path (str): Path to the CSV file with new customer data.
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
    
    # --- 4. Make Predictions ---
    predictions = model.predict(new_data)
    
    # --- 5. Save Predictions ---
    prediction_df = pd.DataFrame(predictions, columns=['churn_prediction'])
    output_df = pd.concat([new_data, prediction_df], axis=1)
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    
    print(f"Predictions saved to: {output_csv_path}")
    print("\n--- Sample Predictions ---")
    print(output_df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch prediction script for churn model.")
    parser.add_argument("--config", type=str, default="config/main_config.yaml", help="Path to the main configuration file.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file with customer data.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the output CSV with predictions.")
    args = parser.parse_args()

    predict(
        config_path=args.config,
        input_csv_path=args.input_csv,
        output_csv_path=args.output_csv
    )

