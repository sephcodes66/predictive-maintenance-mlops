
import numpy as np
import pandas as pd
from typing import Tuple

def create_regression_data(
    n_samples: int, 
    n_features: int,
    seed: int = 1994,
    noise_level: float = 0.3,
    nonlinear: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Generates synthetic regression data with interesting correlations for MLflow and XGBoost demonstrations.

    This function creates a DataFrame of continuous features and computes a target variable with nonlinear
    relationships and interactions between features. The data is designed to be complex enough to demonstrate
    the capabilities of XGBoost, but not so complex that a reasonable model can't be learned.

    Args:
        n_samples (int): Number of samples (rows) to generate.
        n_features (int): Number of feature columns.
        seed (int, optional): Random seed for reproducibility. Defaults to 1994.
        noise_level (float, optional): Level of Gaussian noise to add to the target. Defaults to 0.3.
        nonlinear (bool, optional): Whether to add nonlinear feature transformations. Defaults to True.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            - pd.DataFrame: DataFrame containing the synthetic features.
            - pd.Series: Series containing the target labels.
    """
    rng = np.random.RandomState(seed)
    
    # Generate random continuous features
    X = rng.uniform(-5, 5, size=(n_samples, n_features))
    
    # Create feature DataFrame with meaningful names
    columns = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    
    # Generate base target variable with linear relationship to a subset of features
    # Use only the first n_features//2 features to create some irrelevant features
    weights = rng.uniform(-2, 2, size=n_features//2)
    target = np.dot(X[:, :n_features//2], weights)
    
    # Add some nonlinear transformations if requested
    if nonlinear:
        # Add square term for first feature
        target += 0.5 * X[:, 0]**2
        
        # Add interaction between the second and third features
        if n_features >= 3:
            target += 1.5 * X[:, 1] * X[:, 2]
        
        # Add sine transformation of fourth feature
        if n_features >= 4:
            target += 2 * np.sin(X[:, 3])
        
        # Add exponential of fifth feature, scaled down
        if n_features >= 5:
            target += 0.1 * np.exp(X[:, 4] / 2)
            
        # Add threshold effect for sixth feature
        if n_features >= 6:
            target += 3 * (X[:, 5] > 1.5).astype(float)
    
    # Add Gaussian noise
    noise = rng.normal(0, noise_level * target.std(), size=n_samples)
    target += noise
    
    # Add a few more interesting features to the DataFrame
    
    # Add a correlated feature (but not used in target calculation)
    if n_features >= 7:
        df['feature_correlated'] = df['feature_0'] * 0.8 + rng.normal(0, 0.2, size=n_samples)
    
    # Add a cyclical feature
    df['feature_cyclical'] = np.sin(np.linspace(0, 4*np.pi, n_samples))
    
    # Add a feature with outliers
    df['feature_with_outliers'] = rng.normal(0, 1, size=n_samples)
    # Add outliers to ~1% of samples
    outlier_idx = rng.choice(n_samples, size=n_samples//100, replace=False)
    df.loc[outlier_idx, 'feature_with_outliers'] = rng.uniform(10, 15, size=len(outlier_idx))
    
    return df, pd.Series(target, name='target')
