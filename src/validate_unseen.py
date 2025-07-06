

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
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

    # --- 2. Merge Predictions and Ground Truth ---
    merged_df = pd.concat([predictions_df, ground_truth_df], axis=1)

    y_true = merged_df["churn"]
    y_pred = merged_df["churn_prediction"]

    # --- 3. Validate Predictions and Print Metrics ---
    print("\n--- Unseen Data Validation Results ---")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # --- 4. Generate and Save Confusion Matrix ---
    print("\nGenerating confusion matrix for unseen data...")
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "simulated_data_confusion_matrix.png")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title('Unseen Data Confusion Matrix')
    plt.savefig(cm_path)
    plt.close(fig)
    
    print(f"Confusion matrix saved to: {cm_path}")

if __name__ == "__main__":
    validate_unseen_predictions(
        predictions_path="data/predictions/unseen_predictions.csv",
        ground_truth_path="data/simulated/simulated_diverse_ground_truth.csv",
        output_dir="validation_results"
    )

