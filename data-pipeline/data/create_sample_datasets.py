"""
Create sample datasets for testing the data analysis pipeline.

This script generates various synthetic datasets that demonstrate
different data analysis scenarios and challenges.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
import sqlite3


def create_employee_dataset(n_samples=1000, output_dir="samples"):
    """Create a synthetic employee dataset for HR analytics."""
    
    np.random.seed(42)
    
    # Generate employee data
    data = {
        'employee_id': range(1, n_samples + 1),
        'age': np.random.randint(22, 65, n_samples),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR', 'Finance'], n_samples),
        'years_experience': np.random.randint(0, 30, n_samples),
        'education_level': np.random.choice(['Bachelor', 'Master', 'PhD', 'High School'], n_samples, 
                                          p=[0.4, 0.35, 0.15, 0.1]),
        'salary': np.random.normal(75000, 25000, n_samples),
        'satisfaction_score': np.random.uniform(1, 10, n_samples),
        'performance_rating': np.random.choice(['Poor', 'Average', 'Good', 'Excellent'], n_samples,
                                             p=[0.1, 0.3, 0.4, 0.2]),
        'training_hours': np.random.randint(0, 100, n_samples),
        'promotion_eligible': np.random.choice([True, False], n_samples, p=[0.3, 0.7])
    }
    
    # Add correlations
    for i in range(n_samples):
        # Salary should correlate with experience and education
        if data['education_level'][i] == 'PhD':
            data['salary'][i] = max(data['salary'][i], np.random.normal(90000, 15000))
        elif data['education_level'][i] == 'Master':
            data['salary'][i] = max(data['salary'][i], np.random.normal(80000, 20000))
        
        # Add experience bonus
        data['salary'][i] += data['years_experience'][i] * 1000
        
        # Performance should correlate with satisfaction
        if data['satisfaction_score'][i] > 8:
            data['performance_rating'][i] = np.random.choice(['Good', 'Excellent'], p=[0.3, 0.7])
        elif data['satisfaction_score'][i] < 4:
            data['performance_rating'][i] = np.random.choice(['Poor', 'Average'], p=[0.6, 0.4])
    
    # Introduce missing values (5%)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['satisfaction_score', 'training_hours'])
        data[col][idx] = np.nan
    
    # Add outliers (2%)
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    for idx in outlier_indices:
        data['salary'][idx] = np.random.uniform(200000, 500000)
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "employee_data.csv", index=False)
    
    print(f"Created employee dataset: {len(df)} rows, saved to {output_path / 'employee_data.csv'}")
    return df


def create_sales_dataset(n_samples=2000, output_dir="samples"):
    """Create a synthetic sales dataset for business analytics."""
    
    np.random.seed(123)
    
    # Generate date range (2 years)
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    data = []
    transaction_id = 1
    
    for date in date_range:
        # Seasonal effect
        month = date.month
        seasonal_multiplier = 1.0
        if month in [11, 12]:  # Holiday season
            seasonal_multiplier = 1.5
        elif month in [6, 7, 8]:  # Summer
            seasonal_multiplier = 1.2
        
        # Daily transactions (5-50 per day)
        daily_transactions = int(np.random.poisson(20) * seasonal_multiplier)
        
        for _ in range(daily_transactions):
            product_category = np.random.choice(['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books'])
            customer_type = np.random.choice(['New', 'Returning'], p=[0.3, 0.7])
            sales_channel = np.random.choice(['Online', 'Store', 'Phone'], p=[0.6, 0.3, 0.1])
            
            # Price varies by category
            if product_category == 'Electronics':
                price = np.random.uniform(50, 2000)
            elif product_category == 'Clothing':
                price = np.random.uniform(20, 300)
            elif product_category == 'Home & Garden':
                price = np.random.uniform(30, 800)
            elif product_category == 'Sports':
                price = np.random.uniform(25, 500)
            else:  # Books
                price = np.random.uniform(10, 100)
            
            quantity = np.random.randint(1, 5)
            total_amount = price * quantity
            
            # Discount varies by channel and customer type
            discount = 0
            if customer_type == 'Returning':
                discount = np.random.uniform(0, 0.15)
            if sales_channel == 'Online':
                discount += np.random.uniform(0, 0.05)
            
            final_amount = total_amount * (1 - discount)
            
            data.append({
                'transaction_id': transaction_id,
                'date': date.date(),
                'product_category': product_category,
                'price': round(price, 2),
                'quantity': quantity,
                'total_amount': round(total_amount, 2),
                'discount': round(discount, 3),
                'final_amount': round(final_amount, 2),
                'customer_type': customer_type,
                'sales_channel': sales_channel,
                'sales_rep_id': np.random.randint(1, 50)
            })
            
            transaction_id += 1
    
    df = pd.DataFrame(data)
    
    # Introduce some data quality issues
    # Missing values (3%)
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.03), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['discount', 'sales_rep_id'])
        df.at[idx, col] = np.nan
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "sales_data.csv", index=False)
    
    print(f"Created sales dataset: {len(df)} rows, saved to {output_path / 'sales_data.csv'}")
    return df


def create_customer_dataset(n_samples=5000, output_dir="samples"):
    """Create a synthetic customer dataset for marketing analytics."""
    
    np.random.seed(456)
    
    # Customer demographics
    ages = np.random.normal(40, 15, n_samples).astype(int)
    ages = np.clip(ages, 18, 80)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': ages,
        'gender': np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.49, 0.03]),
        'income': np.random.lognormal(10.5, 0.8, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples,
                                    p=[0.3, 0.4, 0.25, 0.05]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples,
                                         p=[0.4, 0.5, 0.1]),
        'children': np.random.poisson(1.2, n_samples),
        'city_type': np.random.choice(['Urban', 'Suburban', 'Rural'], n_samples, p=[0.5, 0.35, 0.15]),
        'years_as_customer': np.random.exponential(2, n_samples),
        'total_purchases': np.random.gamma(2, 500, n_samples),
        'avg_order_value': np.random.gamma(2, 50, n_samples),
        'last_purchase_days': np.random.exponential(30, n_samples),
        'email_subscribed': np.random.choice([True, False], n_samples, p=[0.6, 0.4]),
        'preferred_channel': np.random.choice(['Email', 'SMS', 'Phone', 'Mail'], n_samples,
                                            p=[0.5, 0.3, 0.1, 0.1])
    }
    
    # Add customer lifetime value calculation
    clv = []
    for i in range(n_samples):
        base_clv = data['total_purchases'][i] * (data['years_as_customer'][i] + 1)
        if data['email_subscribed'][i]:
            base_clv *= 1.2
        if data['income'][i] > 75000:
            base_clv *= 1.3
        clv.append(base_clv)
    
    data['customer_lifetime_value'] = clv
    
    # Create customer segments
    segments = []
    for i in range(n_samples):
        if data['customer_lifetime_value'][i] > np.percentile(clv, 80):
            segments.append('Premium')
        elif data['customer_lifetime_value'][i] > np.percentile(clv, 50):
            segments.append('Standard')
        else:
            segments.append('Basic')
    
    data['customer_segment'] = segments
    
    df = pd.DataFrame(data)
    
    # Clean up data types and ranges
    df['children'] = df['children'].clip(0, 8)
    df['years_as_customer'] = df['years_as_customer'].clip(0, 20)
    df['income'] = df['income'].clip(20000, 500000)
    df['last_purchase_days'] = df['last_purchase_days'].clip(0, 365)
    
    # Introduce missing values (4%)
    missing_indices = np.random.choice(len(df), size=int(len(df) * 0.04), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['income', 'last_purchase_days', 'preferred_channel'])
        df.at[idx, col] = np.nan
    
    # Save to CSV and JSON
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path / "customer_data.csv", index=False)
    
    # Save a subset as JSON for API simulation
    df_sample = df.head(100)
    df_sample.to_json(output_path / "customer_sample.json", orient='records', indent=2)
    
    print(f"Created customer dataset: {len(df)} rows, saved to {output_path / 'customer_data.csv'}")
    return df


def create_iot_sensor_dataset(n_samples=10000, output_dir="samples"):
    """Create a synthetic IoT sensor dataset for time series analysis."""
    
    np.random.seed(789)
    
    # Generate timestamps (1 week of data, every 5 minutes)
    start_time = datetime(2024, 1, 1)
    timestamps = [start_time + timedelta(minutes=5*i) for i in range(n_samples)]
    
    data = {
        'timestamp': timestamps,
        'sensor_id': np.random.choice(['TEMP_01', 'TEMP_02', 'TEMP_03', 'HUM_01', 'HUM_02'], n_samples),
        'device_location': np.random.choice(['Building_A', 'Building_B', 'Building_C'], n_samples),
        'floor': np.random.randint(1, 11, n_samples)
    }
    
    # Generate sensor readings based on type
    temperature_readings = []
    humidity_readings = []
    pressure_readings = []
    battery_levels = []
    
    for i in range(n_samples):
        # Base patterns
        hour = timestamps[i].hour
        day_of_week = timestamps[i].weekday()
        
        # Temperature (varies by time of day and sensor)
        base_temp = 20 + 5 * np.sin((hour - 6) * np.pi / 12)  # Daily cycle
        if data['sensor_id'][i].startswith('TEMP'):
            temp = base_temp + np.random.normal(0, 2)
        else:
            temp = np.nan
        temperature_readings.append(temp)
        
        # Humidity (inversely related to temperature)
        if data['sensor_id'][i].startswith('HUM'):
            humidity = 60 - (temp - 20) * 2 + np.random.normal(0, 5)
            humidity = max(10, min(90, humidity))
        else:
            humidity = np.nan
        humidity_readings.append(humidity)
        
        # Pressure (varies by floor)
        pressure = 1013 - data['floor'][i] * 0.1 + np.random.normal(0, 1)
        pressure_readings.append(pressure)
        
        # Battery level (decreases over time with some noise)
        days_elapsed = (timestamps[i] - start_time).days
        battery = 100 - days_elapsed * 2 + np.random.normal(0, 5)
        battery = max(0, min(100, battery))
        battery_levels.append(battery)
    
    data.update({
        'temperature_c': temperature_readings,
        'humidity_percent': humidity_readings,
        'pressure_hpa': pressure_readings,
        'battery_level': battery_levels,
        'signal_strength': np.random.uniform(-90, -30, n_samples)
    })
    
    df = pd.DataFrame(data)
    
    # Introduce some sensor failures (missing readings)
    failure_indices = np.random.choice(len(df), size=int(len(df) * 0.02), replace=False)
    for idx in failure_indices:
        sensor_type = df.at[idx, 'sensor_id'][:4]
        if sensor_type == 'TEMP':
            df.at[idx, 'temperature_c'] = np.nan
        elif sensor_type == 'HUM_':
            df.at[idx, 'humidity_percent'] = np.nan
    
    # Save to CSV
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / "iot_sensor_data.csv", index=False)
    
    print(f"Created IoT sensor dataset: {len(df)} rows, saved to {output_path / 'iot_sensor_data.csv'}")
    return df


def create_database_samples(output_dir="samples"):
    """Create SQLite database with sample tables."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    db_path = output_path / "sample_database.db"
    
    # Remove existing database
    if db_path.exists():
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    
    # Create and populate products table
    products_data = {
        'product_id': range(1, 101),
        'product_name': [f"Product_{i}" for i in range(1, 101)],
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Sports'], 100),
        'price': np.random.uniform(10, 1000, 100),
        'cost': np.random.uniform(5, 500, 100),
        'supplier_id': np.random.randint(1, 21, 100),
        'in_stock': np.random.choice([True, False], 100, p=[0.8, 0.2])
    }
    
    products_df = pd.DataFrame(products_data)
    products_df.to_sql('products', conn, index=False, if_exists='replace')
    
    # Create and populate orders table
    np.random.seed(101)
    orders_data = {
        'order_id': range(1, 1001),
        'customer_id': np.random.randint(1, 201, 1000),
        'product_id': np.random.randint(1, 101, 1000),
        'quantity': np.random.randint(1, 10, 1000),
        'order_date': pd.date_range('2023-01-01', periods=1000, freq='H')[:1000],
        'status': np.random.choice(['Pending', 'Shipped', 'Delivered', 'Cancelled'], 1000,
                                 p=[0.1, 0.3, 0.5, 0.1])
    }
    
    orders_df = pd.DataFrame(orders_data)
    orders_df.to_sql('orders', conn, index=False, if_exists='replace')
    
    # Create customers table
    customers_data = {
        'customer_id': range(1, 201),
        'first_name': [f"Customer_{i}" for i in range(1, 201)],
        'email': [f"customer{i}@example.com" for i in range(1, 201)],
        'registration_date': pd.date_range('2020-01-01', periods=200, freq='2D')[:200],
        'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'France'], 200),
        'total_orders': np.random.randint(1, 50, 200)
    }
    
    customers_df = pd.DataFrame(customers_data)
    customers_df.to_sql('customers', conn, index=False, if_exists='replace')
    
    conn.close()
    
    print(f"Created SQLite database: {db_path}")
    print("Tables: products (100 rows), orders (1000 rows), customers (200 rows)")


def create_data_dictionary(output_dir="samples"):
    """Create data dictionary for all sample datasets."""
    
    data_dictionary = {
        "employee_data.csv": {
            "description": "Synthetic employee dataset for HR analytics",
            "rows": 1000,
            "columns": {
                "employee_id": "Unique employee identifier (integer)",
                "age": "Employee age in years (22-65)",
                "department": "Department name (Engineering, Sales, Marketing, HR, Finance)",
                "years_experience": "Years of work experience (0-30)",
                "education_level": "Highest education level (High School, Bachelor, Master, PhD)",
                "salary": "Annual salary in USD (normal distribution around 75k)",
                "satisfaction_score": "Job satisfaction score (1-10 scale)",
                "performance_rating": "Performance rating (Poor, Average, Good, Excellent)",
                "training_hours": "Annual training hours (0-100)",
                "promotion_eligible": "Eligible for promotion (boolean)"
            },
            "data_quality_notes": [
                "5% missing values in satisfaction_score and training_hours",
                "2% salary outliers (very high salaries)",
                "Correlations: salary~experience, performance~satisfaction"
            ],
            "analysis_suggestions": [
                "Predict performance rating from other features",
                "Analyze factors affecting employee satisfaction",
                "Study salary equity across departments"
            ]
        },
        
        "sales_data.csv": {
            "description": "Synthetic sales transaction dataset for business analytics",
            "rows": "~30000 (varies by seasonal patterns)",
            "columns": {
                "transaction_id": "Unique transaction identifier",
                "date": "Transaction date (2022-2023)",
                "product_category": "Product category (Electronics, Clothing, etc.)",
                "price": "Unit price in USD",
                "quantity": "Quantity purchased (1-5)",
                "total_amount": "Total amount before discount",
                "discount": "Discount percentage applied (0-20%)",
                "final_amount": "Final amount after discount",
                "customer_type": "Customer type (New, Returning)",
                "sales_channel": "Sales channel (Online, Store, Phone)",
                "sales_rep_id": "Sales representative ID (1-50)"
            },
            "data_quality_notes": [
                "3% missing values in discount and sales_rep_id",
                "Seasonal patterns in transaction volume",
                "Channel-specific discount patterns"
            ],
            "analysis_suggestions": [
                "Time series analysis of sales trends",
                "Customer segmentation analysis",
                "Sales channel performance comparison"
            ]
        },
        
        "customer_data.csv": {
            "description": "Synthetic customer dataset for marketing analytics",
            "rows": 5000,
            "columns": {
                "customer_id": "Unique customer identifier",
                "age": "Customer age (18-80, normal distribution)",
                "gender": "Gender (Male, Female, Other)",
                "income": "Annual income in USD (log-normal distribution)",
                "education": "Education level (High School, Bachelor, Master, PhD)",
                "marital_status": "Marital status (Single, Married, Divorced)",
                "children": "Number of children (Poisson distribution)",
                "city_type": "City type (Urban, Suburban, Rural)",
                "years_as_customer": "Years as customer (exponential distribution)",
                "total_purchases": "Total purchase amount in USD",
                "avg_order_value": "Average order value in USD",
                "last_purchase_days": "Days since last purchase",
                "email_subscribed": "Subscribed to email marketing (boolean)",
                "preferred_channel": "Preferred communication channel",
                "customer_lifetime_value": "Calculated customer lifetime value",
                "customer_segment": "Customer segment (Basic, Standard, Premium)"
            },
            "data_quality_notes": [
                "4% missing values in income, last_purchase_days, preferred_channel",
                "CLV calculated from purchase history and demographics",
                "Customer segments based on CLV percentiles"
            ],
            "analysis_suggestions": [
                "Customer lifetime value prediction",
                "Customer segmentation and targeting",
                "Churn prediction modeling"
            ]
        },
        
        "iot_sensor_data.csv": {
            "description": "Synthetic IoT sensor dataset for time series analysis",
            "rows": 10000,
            "columns": {
                "timestamp": "Measurement timestamp (5-minute intervals)",
                "sensor_id": "Sensor identifier (TEMP_01-03, HUM_01-02)",
                "device_location": "Building location (Building_A, B, C)",
                "floor": "Floor number (1-10)",
                "temperature_c": "Temperature in Celsius (for temp sensors only)",
                "humidity_percent": "Humidity percentage (for humidity sensors only)",
                "pressure_hpa": "Atmospheric pressure in hPa",
                "battery_level": "Device battery level (0-100%)",
                "signal_strength": "WiFi signal strength in dBm"
            },
            "data_quality_notes": [
                "2% missing readings due to sensor failures",
                "Daily temperature cycles with random noise",
                "Battery level decreases over time",
                "Some sensors measure only specific metrics"
            ],
            "analysis_suggestions": [
                "Time series forecasting of sensor readings",
                "Anomaly detection in sensor data",
                "Predictive maintenance based on battery levels"
            ]
        },
        
        "sample_database.db": {
            "description": "SQLite database with relational sample data",
            "tables": {
                "products": {
                    "rows": 100,
                    "description": "Product catalog with pricing and inventory",
                    "columns": {
                        "product_id": "Unique product identifier",
                        "product_name": "Product name",
                        "category": "Product category",
                        "price": "Selling price in USD",
                        "cost": "Cost price in USD",
                        "supplier_id": "Supplier identifier",
                        "in_stock": "Inventory availability (boolean)"
                    }
                },
                "orders": {
                    "rows": 1000,
                    "description": "Order transactions with status tracking",
                    "columns": {
                        "order_id": "Unique order identifier",
                        "customer_id": "Customer identifier",
                        "product_id": "Product identifier",
                        "quantity": "Quantity ordered",
                        "order_date": "Order timestamp",
                        "status": "Order status (Pending, Shipped, Delivered, Cancelled)"
                    }
                },
                "customers": {
                    "rows": 200,
                    "description": "Customer information and activity",
                    "columns": {
                        "customer_id": "Unique customer identifier",
                        "first_name": "Customer first name",
                        "email": "Customer email address",
                        "registration_date": "Account registration date",
                        "country": "Customer country",
                        "total_orders": "Total number of orders placed"
                    }
                }
            },
            "analysis_suggestions": [
                "Customer order behavior analysis",
                "Product performance and profitability analysis",
                "Supply chain and inventory optimization"
            ]
        }
    }
    
    # Save data dictionary
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "data_dictionary.json", 'w') as f:
        json.dump(data_dictionary, f, indent=2)
    
    print(f"Created data dictionary: {output_path / 'data_dictionary.json'}")


def main():
    """Create all sample datasets."""
    
    print("Creating sample datasets for AutoGen Data Analysis Pipeline...")
    print("=" * 70)
    
    output_dir = "samples"
    
    # Create all datasets
    create_employee_dataset(1000, output_dir)
    create_sales_dataset(2000, output_dir)
    create_customer_dataset(5000, output_dir)
    create_iot_sensor_dataset(10000, output_dir)
    create_database_samples(output_dir)
    create_data_dictionary(output_dir)
    
    print("\n" + "=" * 70)
    print("Sample dataset creation completed!")
    print(f"All files saved to: {Path(output_dir).absolute()}")
    print("\nDatasets created:")
    print("- employee_data.csv (1,000 rows) - HR analytics")
    print("- sales_data.csv (~30,000 rows) - Sales analytics") 
    print("- customer_data.csv (5,000 rows) - Marketing analytics")
    print("- iot_sensor_data.csv (10,000 rows) - Time series analytics")
    print("- sample_database.db - Relational data")
    print("- data_dictionary.json - Documentation")
    
    print("\nExample usage:")
    print("python run_pipeline.py --data data/samples/employee_data.csv --request 'Predict employee performance'")
    print("python run_pipeline.py --data data/samples/sales_data.csv --request 'Analyze sales trends and seasonality'")


if __name__ == "__main__":
    main()