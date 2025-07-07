# End-to-End MLOps: Predictive Maintenance

This project demonstrates a complete, robust, and automated MLOps workflow for predicting equipment failure in an industrial setting. It is designed to be a template for building reproducible machine learning pipelines for predictive maintenance.

## Table of Contents

- [The Problem Statement](#the-problem-statement)
- [MLOps Lifecycle](#mlops-lifecycle)
- [Model Performance on Test Data](#model-performance-on-test-data)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
- [Handling Skewed Target Variable](#handling-skewed-target-variable)
- [CI/CD Pipeline](#ci/cd-pipeline)
- [Docker](#docker)
- [Inference](#inference)
- [Configuration](#configuration)

## The Problem Statement

The goal of this project is to predict the Remaining Useful Life (RUL) of turbofan engines based on sensor data. By predicting when an engine is likely to fail, maintenance can be scheduled proactively, reducing downtime and preventing catastrophic failures.

We define the problem as a regression task to predict the RUL, which is the number of operational cycles remaining before an engine is expected to fail.

## MLOps Lifecycle

This project follows a structured MLOps lifecycle, separating concerns into distinct, automated stages:

1.  **Data Ingestion:** Downloads the raw sensor data (e.g., from a public dataset like the NASA Turbofan Engine Degradation dataset) and loads it into a local SQLite database.
2.  **Data Splitting:** Splits the raw data into dedicated training, validation, and test sets to ensure robust model evaluation on truly unseen data.
3.  **Feature Engineering:** Transforms the raw time-series sensor data into a feature table suitable for modeling. This includes creating rolling averages, standard deviations, and other relevant features. Crucially, data transformers (like the RUL `PowerTransformer`) are fitted *only* on the training data and then applied consistently to validation and test sets.
4.  **Model Training:** Trains a baseline regression model (e.g., XGBoost) on the feature-engineered training data, with evaluation on the validation set.
5.  **Hyperparameter Tuning:** Uses Optuna to systematically search for the best model hyperparameters on the training data, evaluating performance on the validation set.
6.  **Experiment Tracking:** Uses MLflow to log all experiments, including parameters, metrics, and artifacts (such as plots and models).
7.  **Model Validation on Test Data:** Evaluates the final tuned model's performance on a completely unseen test set, providing a realistic assessment of its generalization capabilities.
8.  **Testing:** Includes a full suite of unit and integration tests using `pytest` to ensure code quality and reliability.
9.  **Containerization:** Encapsulates the entire workflow in a Docker container for portability and reproducibility.
10. **Deployment:** Provides a simple FastAPI script to serve the model as a REST API.
11. **Monitoring:** Includes a script to simulate monitoring for data drift and model performance degradation.

## Model Performance on Test Data

After hyperparameter tuning and rigorous validation on a dedicated test set, the model achieved the following performance metrics:

- **Mean Squared Error (MSE):** 2374.1666
- **Mean Absolute Error (MAE):** 36.4853
- **R2 Score:** 0.4905

These metrics indicate that the model is able to predict the Remaining Useful Life (RUL) with reasonable accuracy on unseen data.

## Project Structure

```
/
├── config/
│   ├── main_config.yaml
│   └── test_config.yaml
├── data/
│   ├── processed/             # Processed data (train.csv, val.csv, test.csv, rul_transformer.joblib)
│   ├── raw/                   # Raw downloaded data
│   └── simulated/             # Simulated data for monitoring/unseen validation
├── mlruns/
│   └── (MLflow experiment tracking data)
├── notebooks/
│   └── (Jupyter notebooks for exploration)
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── build_features.py      # Feature engineering and RUL transformation
│   ├── create_baseline.py
│   ├── data_quality_checks.py
│   ├── data_splitter.py       # New: Splits raw data into train/val/test
│   ├── evaluate_models.py
│   ├── feature_engineering.py
│   ├── ingest_data.py
│   ├── monitor_drift.py
│   ├── monitor_performance.py
│   ├── monitor.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── train.py
│   ├── tune.py
│   ├── validate_unseen.py     # Validates predictions on unseen data
│   ├── validate.py
│   ├── visualize.py
│   └── __pycache__/
├── tests/
│   ├── test_data_processing.py
│   ├── test_pipelines.py
│   └── __pycache__/
├── validation_results/
│   └── (Validation plots and metrics)
├── visualizations/
│   └── (Visualization plots)
├── venv/
│   └── (Virtual environment)
├── .dockerignore
├── .gitignore
├── Dockerfile
├── entrypoint.sh
├── pytest.ini
├── README.md
└── requirements.txt
```

## Getting Started

### Prerequisites

- Python 3.11+
- Docker

### Setup

1.  **Create and activate the virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Pipeline

To run the full MLOps pipeline, execute the following commands in order:

1.  **Data Ingestion:** Downloads raw data and loads it into SQLite.
    ```bash
    python -m src.ingest_data
    ```

2.  **Data Quality Checks:** Validates the schema and integrity of the ingested data.
    ```bash
    python -m src.data_quality_checks
    ```

3.  **Data Splitting:** Splits the raw data into train, validation, and test sets.
    ```bash
    python -m src.data_splitter
    ```

4.  **Model Training:** Trains the baseline model on the training data.
    ```bash
    python -m src.train
    ```

5.  **Hyperparameter Tuning:** Tunes the model using Optuna on the training and validation sets.
    ```bash
    python -m src.tune
    ```

6.  **Make Predictions on Test Data:** Uses the best tuned model to predict RUL on the unseen test set.
    ```bash
    python -m src.predict --input_csv data/processed/test.csv --output_csv data/predictions/test_predictions.csv
    ```

7.  **Generate Ground Truth for Test Data:** Creates the true RUL values for the test set for validation.
    ```bash
    python -c "import pandas as pd; from src.build_features import build_features; df = pd.read_csv('data/processed/test.csv'); df_processed = build_features(df.copy(), is_training_data=False); df_processed[['RUL']].to_csv('data/processed/test_ground_truth.csv', index=False)"
    ```

8.  **Validate Model on Test Data:** Evaluates the model's performance on the test set and generates plots.
    ```bash
    python -m src.validate_unseen --predictions_path data/predictions/test_predictions.csv --ground_truth_path data/processed/test_ground_truth.csv --output_dir validation_results
    ```

9.  **Running Tests:** Executes the project's unit and integration tests.
    ```bash
    pytest
    ```

## Handling Skewed Target Variable

In many predictive maintenance datasets, the distribution of the target variable (RUL) can be highly skewed. This is because there is often a large amount of data from healthy machines and very little data from machines that are close to failure.

To address this, we apply a **Yeo-Johnson transformation** to the target variable. This is a power transformation that makes the data more Gaussian-like (normal), which can help improve the performance of the model. The fitted `PowerTransformer` object is saved during the training data feature engineering step and is then used to transform the validation, test, and new inference data, and to inverse-transform the predictions back to the original RUL scale.

## CI/CD Pipeline

This project uses GitHub Actions to automate the testing and execution of the MLOps pipeline. The pipeline is defined in `.github/workflows/main.yml` and consists of two main jobs:

1.  **`test`:** This job runs on every push and pull request to the `main` branch. It installs all dependencies and runs the full `pytest` suite to ensure that no new code breaks existing functionality.
2.  **`run_pipeline`:** This job runs only if the `test` job succeeds. It builds the Docker image and runs the container to execute the full end-to-end MLOps pipeline.
 
## Docker

To build the Docker image and run the full pipeline in a containerized environment:

1.  **Build the image:**
    ```bash
    docker build -t predictive-maintenance-app .
    ```

2.  **Run the container:**
    ```bash
    docker run predictive-maintenance-app
    ```

## Inference

There are two ways to get predictions from the trained model:

### 1. Batch Inference

To make predictions on a batch of new data, you can use the `src/predict.py` script. Ensure the input CSV has the same raw sensor columns as the training data. The script will automatically apply the necessary feature engineering and RUL inverse transformation.

```bash
python -m src.predict --input_csv path/to/your/input.csv --output_csv path/to/your/output.csv
```

### 2. Real-time Inference (API Endpoint)

You can serve the model as a REST API using MLflow's built-in server.

1.  **Start the model server:**

    ```bash
    mlflow models serve -m "models:/predictive_maintenance_model_tuned/latest" --port 5001
    ```

2.  **Send a prediction request:**

    Once the server is running, you can send prediction requests to it using a tool like `curl`.

    ```bash
    curl -X POST -H "Content-Type:application/json" --data '{
      "dataframe_split": {
        "columns": [
          "sensor_1", "sensor_2", "sensor_3", "sensor_4", "sensor_5", 
          "sensor_6", "sensor_7", "sensor_8", "sensor_9", "sensor_10"
        ],
        "data": [
          [23.4, 45.6, 12.3, 56.7, 89.0, 34.5, 67.8, 90.1, 23.4, 56.7],
          [12.3, 34.5, 67.8, 90.1, 23.4, 56.7, 89.0, 12.3, 45.6, 78.9]
        ]
      }
    }' http://127.0.0.1:5001/invocations
    ```

## Configuration

All parameters for the project are defined in `config/main_config.yaml`. This includes new parameters for data splitting (`val_size`, `processed_data_dir`). You can modify this file to change the behavior of the training and tuning processes without changing the source code.