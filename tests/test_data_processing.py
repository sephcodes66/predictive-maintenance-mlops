
import pandas as pd
import duckdb

def test_sensor_data_outliers():
    """
    Tests the sensor data for outliers using a SQL query with duckdb.
    """
    # --- 1. Setup ---
    data = {
        'sensor_1': [10, 12, 11, 13, 10, 1000],
        'sensor_2': [20, 22, 21, 23, 20, 2000]
    }
    df = pd.DataFrame(data)

    # --- 2. Execution ---
    # Use duckdb to run a SQL query to find outliers
    # An outlier is defined as a value that is 50x greater than the median.
    result = duckdb.query("""
    WITH stats AS (
        SELECT
            MEDIAN(sensor_1) * 50 as threshold_1,
            MEDIAN(sensor_2) * 50 as threshold_2
        FROM df
    )
    SELECT
        SUM(CASE WHEN sensor_1 > stats.threshold_1 THEN 1 ELSE 0 END) as outlier_count_1,
        SUM(CASE WHEN sensor_2 > stats.threshold_2 THEN 1 ELSE 0 END) as outlier_count_2
    FROM df, stats
    """).df()

    # --- 3. Assertion ---
    assert result['outlier_count_1'][0] == 1, "Expected 1 outlier in sensor_1"
    assert result['outlier_count_2'][0] == 1, "Expected 1 outlier in sensor_2"
