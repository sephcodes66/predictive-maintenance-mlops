
import pandas as pd
import numpy as np
import os

def generate_simulated_data(output_dir: str, n_samples: int):
    """
    Generates simulated sensor data for the predictive maintenance task.
    This data can be used to test the monitoring script for data drift.

    Args:
        output_dir (str): The directory to save the output file.
        n_samples (int): The total number of data points to generate.
    """
    np.random.seed(42)
    
    data = {}
    for i in range(1, 22):
        # Introduce drift in some sensors
        if i % 5 == 0:
            data[f'sensor_{i}'] = np.random.normal(loc=0.5, scale=0.2, size=n_samples)
        else:
            data[f'sensor_{i}'] = np.random.normal(loc=0, scale=0.1, size=n_samples)
            
    simulated_df = pd.DataFrame(data)
    
    # --- Save the dataset ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "simulated_data.csv")
    simulated_df.to_csv(output_path, index=False)
    
    print(f"Generated {n_samples} simulated data points.")
    print(f"Simulated data saved to: {output_path}")

if __name__ == "__main__":
    generate_simulated_data(
        output_dir="data/simulated",
        n_samples=1000
    )
