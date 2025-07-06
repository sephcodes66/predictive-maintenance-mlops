
import pandas as pd
from src.data_processing import create_regression_data

def test_create_regression_data():
    """
    Tests the create_regression_data function to ensure it returns the correct data shapes and types.
    """
    # --- 1. Setup ---
    n_samples = 100
    n_features = 5

    # --- 2. Execution ---
    X, y = create_regression_data(n_samples=n_samples, n_features=n_features)

    # --- 3. Assertion ---
    assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame"
    assert isinstance(y, pd.Series), "y should be a pandas Series"
    assert X.shape[0] == n_samples, f"X should have {n_samples} rows"
    assert y.shape == (n_samples,), f"y should have {n_samples} rows"
