
import sqlite3
import pandas as pd
import os
from imblearn.over_sampling import SMOTE

def build_features(db_path: str = "olist.sqlite", output_dir: str = "data/processed"):
    """
    Builds a more realistic feature table for churn prediction.
    Churn is defined as: "Will a customer who was active in the last 6 months
    make another purchase in the next 3 months?"
    """
    # --- 1. Connect to the database ---
    conn = sqlite3.connect(db_path)

    # --- 2. Define the feature engineering query ---
    query = """
    WITH customer_cohort AS (
        -- Define a cohort of customers who were active in a 6-month period
        SELECT
            c.customer_unique_id
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_purchase_timestamp BETWEEN '2017-01-01' AND '2017-06-30'
        GROUP BY 1
    ),
    features AS (
        -- Create features for this cohort based on their behavior BEFORE the period
        SELECT
            c.customer_unique_id,
            COUNT(o.order_id) AS frequency,
            SUM(p.payment_value) AS monetary,
            AVG(rev.review_score) AS avg_review_score,
            SUM(oi.order_item_id) AS total_items_purchased,
            AVG(p.payment_value) AS avg_order_value,
            COUNT(DISTINCT oi.seller_id) AS num_unique_sellers,
            CAST(JULIANDAY(MAX(o.order_purchase_timestamp)) - JULIANDAY(MIN(o.order_purchase_timestamp)) AS REAL) / COUNT(o.order_id) AS days_between_orders,
            SUM(p.payment_value) AS ltv,
            AVG(oi.order_item_id) AS avg_products_per_order,
            AVG(p.payment_value / oi.order_item_id) AS avg_price_per_product,
            CAST(JULIANDAY('2017-06-30') - JULIANDAY(MAX(o.order_purchase_timestamp)) AS REAL) AS time_since_last_purchase,
            CASE WHEN CAST(JULIANDAY('2017-06-30') - JULIANDAY(MAX(o.order_purchase_timestamp)) AS REAL) <= 30 THEN 1 ELSE 0 END AS recency_0_30_days,
            CASE WHEN CAST(JULIANDAY('2017-06-30') - JULIANDAY(MAX(o.order_purchase_timestamp)) AS REAL) BETWEEN 31 AND 90 THEN 1 ELSE 0 END AS recency_31_90_days,
            CASE WHEN CAST(JULIANDAY('2017-06-30') - JULIANDAY(MAX(o.order_purchase_timestamp)) AS REAL) BETWEEN 91 AND 180 THEN 1 ELSE 0 END AS recency_91_180_days,
            CASE WHEN CAST(JULIANDAY('2017-06-30') - JULIANDAY(MAX(o.order_purchase_timestamp)) AS REAL) > 180 THEN 1 ELSE 0 END AS recency_181_plus_days
        FROM customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_payments p ON o.order_id = p.order_id
        JOIN order_reviews rev ON o.order_id = rev.order_id
        JOIN order_items oi ON o.order_id = oi.order_id
        WHERE o.order_purchase_timestamp < '2017-07-01'
          AND c.customer_unique_id IN (SELECT customer_unique_id FROM customer_cohort)
        GROUP BY 1
    ),
    churn_label AS (
        -- Define the churn label based on their behavior in the NEXT 3 months
        SELECT
            c.customer_unique_id,
            0 AS churn -- Assume they did not churn
        FROM orders o
        JOIN customers c ON o.customer_id = c.customer_id
        WHERE o.order_purchase_timestamp BETWEEN '2017-07-01' AND '2017-09-30'
        GROUP BY 1
    )
    SELECT
        f.*,
        COALESCE(c.churn, 1) AS churn -- If they didn't purchase, they churned (1)
    FROM features f
    LEFT JOIN churn_label c ON f.customer_unique_id = c.customer_unique_id;
    """

    # --- 3. Execute the query and create the feature table ---
    print("Building feature table with realistic churn definition...")
    features_df = pd.read_sql(query, conn)
    print("Feature table built successfully.")

    # --- 4. Handle class imbalance with SMOTE ---
    X = features_df.drop(columns=["churn", "customer_unique_id"])
    y = features_df["churn"]

    # --- 5. Cast columns to correct data types ---
    for col in X.columns:
        if X[col].dtype == 'float64':
            X[col] = X[col].astype(float)

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    resampled_df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name="churn")], axis=1)

    # --- 6. Save the feature table ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "customer_features_realistic.parquet")
    resampled_df.to_parquet(output_path, index=False)
    print(f"Realistic feature table saved to: {output_path}")

    conn.close()


if __name__ == "__main__":
    build_features()
