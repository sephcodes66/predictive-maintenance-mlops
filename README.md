# End-to-End MLOps: Predictive Maintenance

This project demonstrates a complete, MLOps workflow for predicting equipment failure in an industrial setting. It is designed to be a template for building robust, reproducible, and automated machine learning pipelines for predictive maintenance.

## Table of Contents

- [The Business Problem](#the-business-problem)
- [MLOps Lifecycle](#mlops-lifecycle)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
- [Running the Pipeline](#running-the-pipeline)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Feature Engineering](#2-feature-engineering)
  - [3. Model Training](#3-model-training)
  - [4. Hyperparameter Tuning](#4-hyperparameter-tuning)
  - [5. Running Tests](#5-running-tests)
- [Using Docker](#using-docker)
- [Configuration](#configuration)

## The Business Problem

The goal of this project is to predict the Remaining Useful Life (RUL) of turbofan engines based on sensor data. By predicting when an engine is likely to fail, maintenance can be scheduled proactively, reducing downtime and preventing catastrophic failures.

We define the problem as a regression task to predict the RUL, which is the number of operational cycles remaining before an engine is expected to fail.

## MLOps Lifecycle

This project follows a structured MLOps lifecycle, separating concerns into distinct, automated stages:

1.  **Data Ingestion:** Downloads the raw sensor data (e.g., from a public dataset like the NASA Turbofan Engine Degradation dataset) and loads it into a local SQLite database.
2.  **Feature Engineering:** Transforms the raw time-series sensor data into a feature table suitable for modeling. This includes creating rolling averages, standard deviations, and other relevant features.
3.  **Model Training:** Trains a baseline regression model (e.g., XGBoost) on the feature table to predict RUL.
4.  **Hyperparameter Tuning:** Uses Optuna to systematically search for the best model hyperparameters.
5.  **Experiment Tracking:** Uses MLflow to log all experiments, including parameters, metrics, and artifacts.
6.  **Testing:** Includes a full suite of unit and integration tests using `pytest` to ensure code quality and reliability.
7.  **Containerization:** Encapsulates the entire workflow in a Docker container for portability and reproducibility.
8.  **Deployment:** Provides a simple FastAPI script to serve the model as a REST API.
9.  **Monitoring:** Includes a script to simulate monitoring for data drift.

## Project Structure

```
/
├── config/
│   ├── main_config.yaml
│   └── test_config.yaml
├── data/
│   ├── processed/
│   ├── raw/
│   └── simulated/
├── mlruns/
│   └── (MLflow experiment tracking data)
├── notebooks/
│   └── (Jupyter notebooks for exploration)
├── src/
│   ├── __init__.py
│   ├── ingest_data.py
│   ├── build_features.py
│   ├── train.py
│   ├── tune.py
│   ├── predict.py
│   ├── validate.py
│   ├── validate_unseen.py
│   ├── generate_simulated_data.py
│   └── visualize.py
├── tests/
│   └── (Test suite)
├── validation_results/
│   └── (Validation confusion matrices)
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

You can run each stage of the pipeline independently:

### 1. Data Ingestion

```bash
python -m src.ingest_data
```

### 2. Data Quality Checks

```bash
python -m src.data_quality_checks
```

### 3. Feature Engineering

```bash
python -m src.build_features
```

### 4. Model Training

```bash
python -m src.train
```

### 5. Hyperparameter Tuning

```bash
python -m src.tune
```

### 6. Model Validation

```bash
python -m src.validate
```

### 7. Unseen Data Validation

First, generate the simulated data:
```bash
python -m src.generate_simulated_data
```

Then, make predictions on the simulated data:
```bash
python -m src.predict --input_csv data/simulated/simulated_diverse_dataset.csv --output_csv data/predictions/unseen_predictions.csv
```

Finally, validate the predictions:
```bash
python -m src.validate_unseen
```

### 8. Monitoring

This project includes a monitoring pipeline to detect data drift and model performance degradation.

**a. Create Baseline:**
```bash
python -m src.create_baseline --config config/main_config.yaml
```

**b. Monitor Data Drift:**
```bash
python -m src.monitor_drift --config config/main_config.yaml --new_data data/simulated/simulated_diverse_dataset.csv
```

**c. Monitor Performance:**
```bash
python -m src.monitor_performance --config config/main_config.yaml --predictions data/predictions/unseen_predictions.csv --ground_truth data/simulated/simulated_diverse_ground_truth.csv
```

### 9. Running Tests

```bash
pytest
```

## Handling Skewed Target Variable

In many predictive maintenance datasets, the distribution of the target variable (RUL) can be highly skewed. This is because there is often a large amount of data from healthy machines and very little data from machines that are close to failure.

To address this, we apply a **Yeo-Johnson transformation** to the target variable. This is a power transformation that makes the data more Gaussian-like (normal), which can help improve the performance of the model. The fitted `PowerTransformer` object is saved and used to inverse-transform the predictions back to the original RUL scale.

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

To make predictions on a batch of new data, you can use the `src/predict.py` script.

1.  Create a CSV file with the sensor data you want to get predictions for. The file should have the same columns as the training data.
2.  Run the prediction script from your terminal:

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

All parameters for the project are defined in `config/main_config.yaml`. You can modify this file to change the behavior of the training and tuning processes without changing the source code.# predictive-maintenance-mlops
