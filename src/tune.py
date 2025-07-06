
import mlflow
import yaml
import optuna
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from mlflow.models import infer_signature
import os
import pandas as pd

from .visualize import plot_optuna_trials, plot_feature_importance, plot_actual_vs_predicted

def tune_model(config: dict):
    """
    Performs hyperparameter tuning for an XGBoost regression model using Optuna.

    Args:
        config (dict): A dictionary containing the configuration parameters.
    """

    # --- 1. Load Data ---
    features_df = pd.read_parquet("data/processed/turbofan_features.parquet")
    X = features_df.drop(columns=[config["data"]["target_column"]])
    y = features_df[config["data"]["target_column"]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["data"]["random_state"]
    )

    # --- 2. Define Objective Function ---
    def objective(trial):
        param = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "eta": trial.suggest_float("eta", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params(param)
            regressor = xgb.XGBRegressor(**param)
            regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

            preds = regressor.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            mlflow.log_metric("rmse", rmse)
        
        trial.set_user_attr("model", regressor)
        return rmse

    # --- 3. Run Tuning ---
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="Tuning Run") as run:
        study = optuna.create_study(direction=config["tuning"]["direction"])
        study.optimize(objective, n_trials=config["tuning"]["n_trials"])

        # --- 4. Log Best Trial ---
        best_trial = study.best_trial
        best_model = best_trial.user_attrs["model"]

        mlflow.log_metric("best_rmse", best_trial.value)
        mlflow.log_params(best_trial.params)

        signature = infer_signature(X_train, best_model.predict(X_test))

        # --- 5. Log Artifacts ---
        output_dir = config["visualization"]["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        plot_optuna_trials(study, f"{output_dir}/optuna_trials.png")
        plot_feature_importance(best_model, f"{output_dir}/feature_importance.png")
        
        y_pred = best_model.predict(X_test)
        plot_actual_vs_predicted(y_test, y_pred, f"{output_dir}/actual_vs_predicted.png")

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
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"MAE: {mae}")
        print(f"MSE: {mse}")
        print(f"R2: {r2}")

        parent_run_id = os.environ.get("MLFLOW_RUN_ID")
        if parent_run_id:
            mlflow.log_metric("mae", mae, run_id=parent_run_id)
            mlflow.log_metric("mse", mse, run_id=parent_run_id)
            mlflow.log_metric("r2", r2, run_id=parent_run_id)
        else:
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

        print(f"Tuning complete. Best trial logged to experiment: {config['mlflow']['experiment_name']}")
        print(f"Run ID: {run.info.run_id}")

if __name__ == "__main__":
    import sys
    sys.path.append(os.getcwd())
    from src.visualize import plot_optuna_trials, plot_feature_importance
    tune_model("config/main_config.yaml")

