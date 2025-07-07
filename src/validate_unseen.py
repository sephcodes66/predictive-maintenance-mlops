import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import argparse
import joblib
from .visualize import plot_actual_vs_predicted

def validate_unseen_predictions(predictions_path: str, ground_truth_path: str, output_dir: str):
    output_dir = os.path.abspath(output_dir)
    """
    Validates the predictions on the unseen data against the ground truth.

    Args:
        predictions_path (str): Path to the CSV file with the predictions.
        ground_truth_path (str): Path to the CSV file with the ground truth.
        output_dir (str): Directory to save validation results.
    """
    # --- 1. Load Predictions and Ground Truth ---
    predictions_df = pd.read_csv(predictions_path)
    ground_truth_df = pd.read_csv(ground_truth_path)

    # --- 2. Extract Predictions and Ground Truth ---
    y_pred = predictions_df["RUL_prediction"]
    y_true_transformed = ground_truth_df["RUL"]

    # --- 3. Inverse Transform Ground Truth ---
    transformer_path = os.path.join("data/processed", "rul_transformer.joblib")
    transformer = joblib.load(transformer_path)
    y_true = transformer.inverse_transform(y_true_transformed.values.reshape(-1, 1)).flatten()

    # --- 4. Validate Predictions and Print Metrics ---
    print("\n--- Unseen Data Validation Results ---")
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # --- 5. Generate and Save Scatter Plot ---
    print("\nGenerating scatter plot for unseen data...")
    print(f"[validate_unseen.py] Plot output directory: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "simulated_data_scatter_plot.png")
    
    plot_actual_vs_predicted(
        y_true,
        y_pred,
        plot_path,
        dataset_name="Unseen Data (Original Scale)"
    )
    
    print(f"Scatter plot saved to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--ground_truth_path", required=True)
    parser.add_argument("--output_dir", default="validation_results")
    args = parser.parse_args()

    validate_unseen_predictions(
        predictions_path=args.predictions_path,
        ground_truth_path=args.ground_truth_path,
        output_dir=args.output_dir
    )