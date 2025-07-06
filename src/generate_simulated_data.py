
import pandas as pd
import numpy as np
import os

def generate_simulated_data(output_dir: str, n_samples: int, churn_ratio: float):
    """
    Generates a diverse, simulated dataset for model testing.

    Args:
        output_dir (str): The directory to save the output files.
        n_samples (int): The total number of customers to generate.
        churn_ratio (float): The proportion of customers who should be labeled as churned.
    """
    np.random.seed(42)
    
    n_churned = int(n_samples * churn_ratio)
    n_not_churned = n_samples - n_churned

    archetypes = []

    # --- Generate Churned Customers (70%) ---
    # 1. High-Value, At-Risk
    for _ in range(int(n_churned * 0.4)):
        time_since_last_purchase = np.random.uniform(181, 365)
        archetypes.append({
            'frequency': np.random.randint(10, 20),
            'monetary': np.random.uniform(1000, 5000),
            'avg_review_score': np.random.uniform(3.5, 5.0),
            'total_items_purchased': np.random.randint(15, 30),
            'avg_order_value': np.random.uniform(100, 300),
            'num_unique_sellers': np.random.randint(5, 15),
            'days_between_orders': np.random.uniform(10, 40),
            'ltv': np.random.uniform(100, 300),
            'avg_products_per_order': np.random.uniform(1.5, 3.0),
            'avg_price_per_product': np.random.uniform(50, 150),
            'time_since_last_purchase': time_since_last_purchase,
            'recency_0_30_days': 1 if time_since_last_purchase <= 30 else 0,
            'recency_31_90_days': 1 if 31 <= time_since_last_purchase <= 90 else 0,
            'recency_91_180_days': 1 if 91 <= time_since_last_purchase <= 180 else 0,
            'recency_181_plus_days': 1 if time_since_last_purchase > 180 else 0,
            'churn': 1
        })

    # 2. Marketplace Explorers (Not Recent)
    for _ in range(int(n_churned * 0.3)):
        time_since_last_purchase = np.random.uniform(181, 365)
        archetypes.append({
            'frequency': np.random.randint(5, 15),
            'monetary': np.random.uniform(500, 2000),
            'avg_review_score': np.random.uniform(3.0, 4.5),
            'total_items_purchased': np.random.randint(10, 25),
            'avg_order_value': np.random.uniform(50, 150),
            'num_unique_sellers': np.random.randint(20, 50), # High unique sellers
            'days_between_orders': np.random.uniform(20, 60),
            'ltv': np.random.uniform(50, 150),
            'avg_products_per_order': np.random.uniform(1.0, 2.5),
            'avg_price_per_product': np.random.uniform(20, 100),
            'time_since_last_purchase': time_since_last_purchase,
            'recency_0_30_days': 1 if time_since_last_purchase <= 30 else 0,
            'recency_31_90_days': 1 if 31 <= time_since_last_purchase <= 90 else 0,
            'recency_91_180_days': 1 if 91 <= time_since_last_purchase <= 180 else 0,
            'recency_181_plus_days': 1 if time_since_last_purchase > 180 else 0,
            'churn': 1
        })
        
    # 3. Edge Case: Low Review Score, Not Recent
    for _ in range(int(n_churned * 0.3)):
        time_since_last_purchase = np.random.uniform(181, 365)
        archetypes.append({
            'frequency': np.random.randint(2, 5),
            'monetary': np.random.uniform(100, 500),
            'avg_review_score': np.random.uniform(1.0, 2.5), # Very low score
            'total_items_purchased': np.random.randint(2, 5),
            'avg_order_value': np.random.uniform(20, 100),
            'num_unique_sellers': np.random.randint(1, 5),
            'days_between_orders': np.random.uniform(50, 150),
            'ltv': np.random.uniform(20, 100),
            'avg_products_per_order': np.random.uniform(1.0, 1.5),
            'avg_price_per_product': np.random.uniform(10, 50),
            'time_since_last_purchase': time_since_last_purchase,
            'recency_0_30_days': 1 if time_since_last_purchase <= 30 else 0,
            'recency_31_90_days': 1 if 31 <= time_since_last_purchase <= 90 else 0,
            'recency_91_180_days': 1 if 91 <= time_since_last_purchase <= 180 else 0,
            'recency_181_plus_days': 1 if time_since_last_purchase > 180 else 0,
            'churn': 1
        })

    # --- Generate Not Churned Customers (30%) ---
    # 1. New, but Low-Engagement
    for _ in range(int(n_not_churned * 0.4)):
        time_since_last_purchase = np.random.uniform(1, 30)
        archetypes.append({
            'frequency': 1,
            'monetary': np.random.uniform(20, 100),
            'avg_review_score': np.random.uniform(2.5, 4.0),
            'total_items_purchased': 1,
            'avg_order_value': np.random.uniform(20, 100),
            'num_unique_sellers': 1,
            'days_between_orders': 0,
            'ltv': np.random.uniform(20, 100),
            'avg_products_per_order': 1.0,
            'avg_price_per_product': np.random.uniform(20, 100),
            'time_since_last_purchase': time_since_last_purchase,
            'recency_0_30_days': 1 if time_since_last_purchase <= 30 else 0,
            'recency_31_90_days': 1 if 31 <= time_since_last_purchase <= 90 else 0,
            'recency_91_180_days': 1 if 91 <= time_since_last_purchase <= 180 else 0,
            'recency_181_plus_days': 1 if time_since_last_purchase > 180 else 0,
            'churn': 0
        })

    # 2. Loyal, but Low-Value
    for _ in range(int(n_not_churned * 0.4)):
        time_since_last_purchase = np.random.uniform(1, 90)
        archetypes.append({
            'frequency': np.random.randint(15, 30),
            'monetary': np.random.uniform(300, 800),
            'avg_review_score': np.random.uniform(4.0, 5.0),
            'total_items_purchased': np.random.randint(20, 40),
            'avg_order_value': np.random.uniform(10, 30), # Low value
            'num_unique_sellers': np.random.randint(2, 5),
            'days_between_orders': np.random.uniform(5, 20),
            'ltv': np.random.uniform(10, 30),
            'avg_products_per_order': np.random.uniform(1.0, 1.5),
            'avg_price_per_product': np.random.uniform(5, 15),
            'time_since_last_purchase': time_since_last_purchase,
            'recency_0_30_days': 1 if time_since_last_purchase <= 30 else 0,
            'recency_31_90_days': 1 if 31 <= time_since_last_purchase <= 90 else 0,
            'recency_91_180_days': 1 if 91 <= time_since_last_purchase <= 180 else 0,
            'recency_181_plus_days': 1 if time_since_last_purchase > 180 else 0,
            'churn': 0
        })
        
    # 3. Edge Case: Perfect Score, Frequent
    for _ in range(n_samples - len(archetypes)):
        time_since_last_purchase = np.random.uniform(1, 180)
        archetypes.append({
            'frequency': np.random.randint(20, 40),
            'monetary': np.random.uniform(1000, 3000),
            'avg_review_score': 5.0, # Perfect score
            'total_items_purchased': np.random.randint(25, 50),
            'avg_order_value': np.random.uniform(50, 100),
            'num_unique_sellers': np.random.randint(5, 10),
            'days_between_orders': np.random.uniform(10, 30),
            'ltv': np.random.uniform(50, 100),
            'avg_products_per_order': np.random.uniform(1.2, 2.0),
            'avg_price_per_product': np.random.uniform(20, 80),
            'time_since_last_purchase': time_since_last_purchase,
            'recency_0_30_days': 1 if time_since_last_purchase <= 30 else 0,
            'recency_31_90_days': 1 if 31 <= time_since_last_purchase <= 90 else 0,
            'recency_91_180_days': 1 if 91 <= time_since_last_purchase <= 180 else 0,
            'recency_181_plus_days': 1 if time_since_last_purchase > 180 else 0,
            'churn': 0
        })

    simulated_df = pd.DataFrame(archetypes)
    
    # --- Save the datasets ---
    os.makedirs(output_dir, exist_ok=True)
    
    inference_data_path = os.path.join(output_dir, "simulated_diverse_dataset.csv")
    ground_truth_path = os.path.join(output_dir, "simulated_diverse_ground_truth.csv")
    
    # Save the inference data without the churn label
    simulated_df.drop(columns=['churn']).to_csv(inference_data_path, index=False)
    
    # Save the ground truth separately
    simulated_df[['churn']].to_csv(ground_truth_path, index=False)
    
    print(f"Generated {n_samples} simulated customers.")
    print(f"Inference data saved to: {inference_data_path}")
    print(f"Ground truth saved to: {ground_truth_path}")

if __name__ == "__main__":
    generate_simulated_data(
        output_dir="data/simulated",
        n_samples=500,
        churn_ratio=0.7
    )
