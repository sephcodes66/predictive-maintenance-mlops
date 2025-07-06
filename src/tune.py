
import mlflow
import yaml
import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from mlflow.models import infer_signature
import os
import pandas as pd

from .visualize import plot_optuna_trials, plot_feature_importance, plot_confusion_matrix, plot_roc_curve

def tune_model(config_path: str):
    """
    Performs hyperparameter tuning for an XGBoost model using Optuna.

    Args:
        config_path (str): Path to the main configuration file.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 1. Load Data ---
    features_df = pd.read_parquet("data/processed/customer_features_realistic.parquet")
    X = features_df.drop(columns=[config["data"]["target_column"]])
    y = features_df[config["data"]["target_column"]]

    if len(y.unique()) < 2:
        print("Warning: The dataset has only one class. Skipping tuning.")
        return

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"], stratify=y
    )

    if len(y_train.unique()) < 2:
        print("Warning: The training set has only one class. Skipping tuning.")
        return

    # --- 2. Define Objective Function ---
    def objective(trial):
        param = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params(param)
            regressor = xgb.XGBClassifier(**param)
            if len(y_test.unique()) > 1:
                regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            else:
                regressor.fit(X_train, y_train, verbose=False)

            if len(y_test.unique()) > 1:
                preds = regressor.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, preds)
                mlflow.log_metric("auc", auc)
            else:
                auc = 0.5
                mlflow.log_metric("auc", auc)
        
        trial.set_user_attr("model", regressor)
        return auc

    # --- 3. Run Tuning ---
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="Tuning Run") as run:
        study = optuna.create_study(direction=config["tuning"]["direction"])
        study.optimize(objective, n_trials=config["tuning"]["n_trials"])

        # --- 4. Log Best Trial ---
        best_trial = study.best_trial
        best_model = best_trial.user_attrs["model"]

        mlflow.log_metric("best_auc", best_trial.value)
        mlflow.log_params(best_trial.params)

        signature = infer_signature(X_train, best_model.predict_proba(X_test)[:, 1])

        # --- 5. Log Artifacts ---
        output_dir = config["visualization"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        plot_optuna_trials(study, f"{output_dir}/optuna_trials.png")
        plot_feature_importance(best_model, f"{output_dir}/feature_importance.png")
        
        if len(y_test.unique()) > 1:
            y_pred = best_model.predict(X_test)
            y_pred_proba = best_model.predict_proba(X_test)[:, 1]
            plot_confusion_matrix(y_test, y_pred, f"{output_dir}/confusion_matrix.png")
            plot_roc_curve(y_test, y_pred_proba, f"{output_dir}/roc_curve.png")

        mlflow.log_artifacts(output_dir)

        # --- 6. Log Model ---
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[[0]],
            registered_model_name=f"{config['model']['name']}_tuned",
        )

        # --- 7. Log Metrics ---
        if len(y_test.unique()) > 1:
            y_pred = best_model.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            print(f"F1 Score: {f1}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            parent_run_id = os.environ.get("MLFLOW_RUN_ID")
            if parent_run_id:
                mlflow.log_metric("f1_score", f1, run_id=parent_run_id)
                mlflow.log_metric("precision", precision, run_id=parent_run_id)
                mlflow.log_metric("recall", recall, run_id=parent_run_id)
            else:
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)

        print(f"Tuning complete. Best trial logged to experiment: {config['mlflow']['experiment_name']}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    from src.visualize import plot_optuna_trials, plot_feature_importance
    tune_model("config/main_config.yaml")

