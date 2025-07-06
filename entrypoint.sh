#!/bin/bash

# Exit on error
set -e

# Run the data ingestion script
echo "--- Running data ingestion ---"
python -m src.ingest_data

# Run the feature engineering script
echo "--- Running feature engineering ---"
python -m src.build_features

# Run the training script
echo "--- Running training ---"
python -m src.train

# Run the tuning script
echo "--- Running tuning ---"
python -m src.tune

# Run the tests
echo "--- Running tests ---"
pytest

echo "--- MLOps pipeline complete ---"