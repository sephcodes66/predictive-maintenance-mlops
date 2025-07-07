
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

from src.build_features import build_features

def tune_model(config: dict):
    """
    Performs hyperparameter tuning for an XGBoost regression model using Optuna.

    Args:
        config (dict): A dictionary containing the configuration parameters.
    """    
    # --- 1. Load Data ---
    processed_data_dir = config["data"]["processed_data_dir"]
    train_df = pd.read_csv(os.path.join(processed_data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(processed_data_dir, "val.csv"))

    # --- 2. Apply Feature Engineering ---
    train_df_processed = build_features(train_df.copy(), is_training_data=True)
    val_df_processed = build_features(val_df.copy(), is_training_data=False)

    X_train = train_df_processed.drop(columns=[config["data"]["target_column"]])
    y_train = train_df_processed[config["data"]["target_column"]]

    X_val = val_df_processed.drop(columns=[config["data"]["target_column"]])
    y_val = val_df_processed[config["data"]["target_column"]]

    # --- 3. Define Objective Function ---
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
            regressor.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            preds = regressor.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, preds))
            mlflow.log_metric("rmse", rmse)
        
        trial.set_user_attr("model", regressor)
        return rmse

    # --- 4. Run Tuning ---
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name="Tuning Run") as run:
        study = optuna.create_study(direction=config["tuning"]["direction"])
        study.optimize(objective, n_trials=config["tuning"]["n_trials"])

        # --- 5. Log Best Trial ---
        best_trial = study.best_trial
        best_model = best_trial.user_attrs["model"]

        mlflow.log_metric("best_rmse", best_trial.value)
        mlflow.log_params(best_trial.params)

        signature = infer_signature(X_train, best_model.predict(X_val))

        # --- 6. Log Artifacts ---
        output_dir = os.path.abspath(config["visualization"]["output_dir"])
        print(f"[tune.py] Plot output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        plot_optuna_trials(study, os.path.join(output_dir, "optuna_trials.png"))
        plot_feature_importance(best_model, os.path.join(output_dir, "feature_importance_tuned.png"), dataset_name="Validation Set")
        
        y_pred = best_model.predict(X_val)
        plot_actual_vs_predicted(
            y_val,
            y_pred,
            os.path.join(output_dir, "actual_vs_predicted_tuned.png"),
            dataset_name="Validation Set"
        )

        mlflow.log_artifacts(output_dir)

        # --- 7. Log Model ---
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.iloc[[0]],
            registered_model_name=f"{config['model']['name']}_tuned",
        )

        # --- 8. Log Metrics ---
        y_pred = best_model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
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

        # --- 9. Save the run ID ---
        with open("mlruns/0/latest_run_id.txt", "w") as f:
            f.write(run.info.run_id)

