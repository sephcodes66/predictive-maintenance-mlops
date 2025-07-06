

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

def validate_unseen_predictions(predictions_path: str, ground_truth_path: str, output_dir: str):
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
    y_true = ground_truth_df["RUL"]

    # --- 3. Validate Predictions and Print Metrics ---
    print("\n--- Unseen Data Validation Results ---")
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    # --- 4. Generate and Save Scatter Plot ---
    print("\nGenerating scatter plot for unseen data...")
    
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "simulated_data_scatter_plot.png")
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("True vs. Predicted RUL on Unseen Data")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Scatter plot saved to: {plot_path}")

if __name__ == "__main__":
    validate_unseen_predictions(
        predictions_path="data/predictions/unseen_predictions.csv",
        ground_truth_path="path/to/your/ground_truth.csv", # Update this path
        output_dir="validation_results"
    )

