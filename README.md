# End-to-End MLOps: Olist Churn Prediction

This project demonstrates a complete, professional MLOps workflow for predicting customer churn on the Olist e-commerce dataset. It is designed to be a template for building robust, reproducible, and automated machine learning pipelines.

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

The goal of this project is to predict customer churn for the Olist e-commerce platform. By identifying customers who are likely to stop making purchases, the business can proactively engage them with targeted marketing campaigns to improve customer retention.

We define **churn** as a customer who has not made a purchase in the last 180 days.

## MLOps Lifecycle

This project follows a structured MLOps lifecycle, separating concerns into distinct, automated stages:

1.  **Data Ingestion:** Downloads the raw data from Kaggle and loads it into a local SQLite database.
2.  **Feature Engineering:** Transforms the raw data into a feature table suitable for modeling, using SQL to calculate features like Recency, Frequency, and Monetary (RFM) values.
3.  **Model Training:** Trains a baseline XGBoost classification model on the feature table.
4.  **Hyperparameter Tuning:** Uses Optuna to systematically search for the best model hyperparameters.
5.  **Experiment Tracking:** Uses MLflow to log all experiments, including parameters, metrics, and artifacts, to a Databricks workspace.
6.  **Testing:** Includes a full suite of unit and integration tests using `pytest` to ensure code quality and reliability.
7.  **Containerization:** Encapsulates the entire workflow in a Docker container for portability and reproducibility.

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
- A Kaggle account and API token (`kaggle.json`)
- A Databricks workspace and personal access token

### Setup

1.  **Kaggle API Token:**
    - Go to your Kaggle account settings page (`https://www.kaggle.com/me/account`).
    - Click on "Create New API Token". This will download a `kaggle.json` file.
    - Place this file in the `~/.kaggle/` directory.

2.  **Databricks CLI:**
    - Run `databricks configure` and provide your Databricks host and personal access token.

3.  **Create and activate the virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Running the Pipeline

You can run each stage of the pipeline independently:

### 1. Data Quality Checks

```bash
python -m src.data_quality_checks
```

### 2. Data Ingestion

```bash
python -m src.ingest_data
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

### 8. Running Tests

```bash
pytest
```

## CI/CD Pipeline

This project uses GitHub Actions to automate the testing and execution of the MLOps pipeline. The pipeline is defined in `.github/workflows/main.yml` and consists of two main jobs:

1.  **`test`:** This job runs on every push and pull request to the `main` branch. It installs all dependencies and runs the full `pytest` suite to ensure that no new code breaks existing functionality.
2.  **`run_pipeline`:** This job runs only if the `test` job succeeds. It builds the Docker image and runs the container to execute the full end-to-end MLOps pipeline.

You can add a status badge to the top of this `README.md` to show the current status of the pipeline:

```markdown
[![MLOps Pipeline](https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>/actions/workflows/main.yml/badge.svg)](https://github.com/<YOUR_USERNAME>/<YOUR_REPOSITORY>/actions/workflows/main.yml)
```

## Docker


To build the Docker image and run the full pipeline in a containerized environment:

1.  **Build the image:**
    ```bash
    docker build -t olist-churn-predictor .
    ```

2.  **Run the container:**
    ```bash
    docker run olist-churn-predictor
    ```

## Inference

There are two ways to get predictions from the trained model:

### 1. Batch Inference

To make predictions on a batch of new customers, you can use the `src/predict.py` script.

1.  Create a CSV file with the customer data you want to get predictions for. The file should have the same columns as the training data.
2.  Run the prediction script from your terminal:

    ```bash
    python -m src.predict --input_csv path/to/your/input.csv --output_csv path/to/your/output.csv
    ```

### 2. Real-time Inference (API Endpoint)

You can serve the model as a REST API using MLflow's built-in server.

1.  **Start the model server:**

    ```bash
    mlflow models serve -m "models:/olist_churn_model_tuned/latest" --port 5001
    ```

2.  **Send a prediction request:**

    Once the server is running, you can send prediction requests to it using a tool like `curl`.

    ```bash
    curl -X POST -H "Content-Type:application/json" --data '{
      "dataframe_split": {
        "columns": [
          "frequency", "monetary", "avg_review_score", "total_items_purchased", 
          "avg_order_value", "num_unique_sellers", "days_between_orders", 
          "ltv", "avg_products_per_order", "avg_price_per_product", 
          "time_since_last_purchase", "recency_0_30_days", "recency_31_90_days", 
          "recency_91_180_days", "recency_181_plus_days"
        ],
        "data": [
          [5, 500.50, 4.5, 10, 100.10, 2, 30.5, 100.10, 2.0, 50.05, 200, 0, 0, 0, 1],
          [1, 25.00, 3.0, 1, 25.00, 1, 0, 25.00, 1.0, 25.00, 10, 1, 0, 0, 0]
        ]
      }
    }' http://127.0.0.1:5001/invocations
    ```

## Configuration

All parameters for the project are defined in `config/main_config.yaml`. You can modify this file to change the behavior of the training and tuning processes without changing the source code.