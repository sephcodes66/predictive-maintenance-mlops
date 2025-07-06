

import pandas as pd
import pandera as pa
import sqlite3
import sys

# --- Define Data Quality Schemas ---

# Schema for the 'orders' table
orders_schema = pa.DataFrameSchema({
    "order_id": pa.Column(str, required=True, unique=True, description="Unique identifier for the order."),
    "customer_id": pa.Column(str, required=True, description="Identifier for the customer."),
    "order_status": pa.Column(str, checks=pa.Check.isin(['delivered', 'shipped', 'canceled', 'invoiced', 'processing', 'unavailable', 'approved', 'created']), description="Status of the order."),
    "order_purchase_timestamp": pa.Column(pa.DateTime, required=True, description="Timestamp of the order purchase."),
    "order_approved_at": pa.Column(pa.DateTime, required=False, nullable=True, description="Timestamp when the order was approved."),
    "order_delivered_carrier_date": pa.Column(pa.DateTime, required=False, nullable=True, description="Timestamp when the order was delivered to the carrier."),
    "order_delivered_customer_date": pa.Column(pa.DateTime, required=False, nullable=True, description="Timestamp when the order was delivered to the customer."),
    "order_estimated_delivery_date": pa.Column(pa.DateTime, required=True, description="Estimated delivery date of the order."),
})

# Schema for the 'order_payments' table
order_payments_schema = pa.DataFrameSchema({
    "order_id": pa.Column(str, required=True, description="Identifier for the order."),
    "payment_sequential": pa.Column(int, required=True, description="Sequence number for the payment method."),
    "payment_type": pa.Column(str, checks=pa.Check.isin(['credit_card', 'boleto', 'voucher', 'debit_card', 'not_defined']), description="Type of payment."),
    "payment_installments": pa.Column(int, required=True, checks=pa.Check.ge(0), description="Number of payment installments."),
    "payment_value": pa.Column(float, required=True, checks=pa.Check.ge(0), description="Value of the payment."),
})


def run_data_quality_checks(db_path: str = "olist.sqlite"):
    """
    Runs data quality checks on the raw data tables.

    Args:
        db_path (str): Path to the SQLite database.
    """
    print("--- Running Data Quality Checks ---")
    
    # --- 1. Connect to the database ---
    conn = sqlite3.connect(db_path)

    # --- 2. Load Data ---
    orders_df = pd.read_sql("SELECT * FROM orders", conn, parse_dates=[
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    ])
    order_payments_df = pd.read_sql("SELECT * FROM order_payments", conn)

    # --- 3. Run Validations ---
    try:
        print("\nValidating 'orders' table...")
        orders_schema.validate(orders_df, lazy=True)
        print("'orders' table is valid.")

        print("\nValidating 'order_payments' table...")
        order_payments_schema.validate(order_payments_df, lazy=True)
        print("'order_payments' table is valid.")

    except pa.errors.SchemaErrors as err:
        print("\n--- Data Quality Checks Failed ---")
        print(err.failure_cases)
        print("\n--- Data Quality Checks Failed ---")
        sys.exit(1)

    finally:
        conn.close()

    print("\n--- All Data Quality Checks Passed ---")


if __name__ == "__main__":
    run_data_quality_checks()

