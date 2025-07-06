
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import pandas as pd
import numpy as np
import optuna
import mlflow
from mlflow.tracking import MlflowClient
import os
import yaml
from sklearn.model_selection import train_test_split
import argparse

def plot_feature_distributions(X: pd.DataFrame, output_path: str):
    """
    Creates and saves a grid of histograms for each feature in the dataset.
    """
    features = X.columns
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
    
    for i, feature in enumerate(features):
        if i < len(axes):
            ax = axes[i]
            sns.histplot(X[feature], ax=ax, kde=True, color='skyblue')
            ax.set_title(f'Distribution of {feature}')
    
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    fig.suptitle('Feature Distributions', y=1.02, fontsize=16)
    plt.savefig(output_path)
    plt.close(fig)

def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series, output_path: str):
    """
    Creates and saves a correlation heatmap of all features and the target variable.
    """
    data = X.copy()
    data['target'] = y
    
    corr_matrix = data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap=cmap,
                center=0, square=True, linewidths=0.5, ax=ax)
    
    ax.set_title('Feature Correlation Heatmap', fontsize=16)
    plt.savefig(output_path)
    plt.close(fig)

def plot_optuna_trials(study: optuna.study.Study, output_path: str):
    """
    Creates and saves a plot showing the objective value of each Optuna trial.
    """
    trial_values = [t.value for t in study.trials]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(trial_values)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.set_title("Optuna Optimization History")
    plt.savefig(output_path)
    plt.close(fig)

def plot_feature_importance(model: xgb.XGBRegressor, output_path: str):
    """
    Creates and saves a plot showing the feature importance of the best model.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    xgb.plot_importance(model, ax=ax, importance_type='gain')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def plot_actual_vs_predicted(y_true: pd.Series, y_pred: np.ndarray, output_path: str):
    """
    Creates and saves a scatter plot of actual vs. predicted values.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax.set_xlabel('Actual RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title('Actual vs. Predicted RUL')
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save visualizations.")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save the visualizations.")
    args = parser.parse_args()

    # Load config
    with open("config/main_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = pd.read_parquet("data/processed/turbofan_features.parquet")
    X = data.drop(columns=[config["data"]["target_column"]])
    y = data[config["data"]["target_column"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
    )

    # Load best model from MLflow
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config["mlflow"]["experiment_name"])
    best_run = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.best_rmse ASC"],
        max_results=1,
    )[0]
    model_uri = f"runs:/{best_run.info.run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    # Generate and save plots
    plot_feature_distributions(X, os.path.join(output_dir, "feature_distributions.png"))
    plot_correlation_heatmap(X, y, os.path.join(output_dir, "correlation_heatmap.png"))
    
    y_pred = model.predict(X_test)

    plot_actual_vs_predicted(y_test, y_pred, os.path.join(output_dir, "actual_vs_predicted.png"))

    print(f"Visualizations saved to: {output_dir}")
