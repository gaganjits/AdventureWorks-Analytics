"""
Phase 1: Data Preparation Script
Complete data loading, merging, feature engineering, and quality checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("PHASE 1: DATA PREPARATION")
print("=" * 70)

# Set paths
raw_path = Path('data/raw')
processed_path = Path('data/processed')
processed_path.mkdir(exist_ok=True)

# ============================================================================
# TASK 1: LOAD AND MERGE DATASETS
# ============================================================================

print("\n" + "=" * 70)
print("TASK 1: LOAD AND MERGE DATASETS")
print("=" * 70)

print("\n1.1 Loading all datasets...")

# Load sales data
sales_2015 = pd.read_csv(raw_path / 'converted_AdventureWorks_Sales_2015.csv')
sales_2016 = pd.read_csv(raw_path / 'converted_AdventureWorks_Sales_2016.csv')
sales_2017 = pd.read_csv(raw_path / 'converted_AdventureWorks_Sales_2017.csv')

# Load reference data
customers = pd.read_csv(raw_path / 'converted_AdventureWorks_Customers.csv')
products = pd.read_csv(raw_path / 'AdventureWorks_Products.csv')
returns = pd.read_csv(raw_path / 'converted_AdventureWorks_Returns.csv')
territories = pd.read_csv(raw_path / 'AdventureWorks_Territories.csv')
categories = pd.read_csv(raw_path / 'AdventureWorks_Product_Categories.csv')
subcategories = pd.read_csv(raw_path / 'AdventureWorks_Product_Subcategories.csv')

print(f"✓ Loaded 9 datasets")

print("\n1.2 Merging sales files (2015-2017)...")
sales = pd.concat([sales_2015, sales_2016, sales_2017], ignore_index=True)
sales['OrderDate'] = pd.to_datetime(sales['OrderDate'])
sales['StockDate'] = pd.to_datetime(sales['StockDate'])
sales = sales.sort_values('OrderDate').reset_index(drop=True)
print(f"✓ Merged sales: {sales.shape[0]:,} rows")

print("\n1.3 Building product hierarchy...")
products_full = products.merge(subcategories, on='ProductSubcategoryKey', how='left')
products_full = products_full.merge(categories, on='ProductCategoryKey', how='left')
print(f"✓ Product hierarchy: {products_full.shape[0]} products with categories")

print("\n1.4 Creating enriched sales dataset...")
sales_enriched = sales.merge(products_full, on='ProductKey', how='left')
sales_enriched = sales_enriched.merge(territories, left_on='TerritoryKey', right_on='SalesTerritoryKey', how='left')

# Calculate revenue and profit
sales_enriched['Revenue'] = sales_enriched['OrderQuantity'] * sales_enriched['ProductPrice']
sales_enriched['Profit'] = sales_enriched['Revenue'] - (sales_enriched['OrderQuantity'] * sales_enriched['ProductCost'])
sales_enriched['ProfitMargin'] = (sales_enriched['Profit'] / sales_enriched['Revenue'] * 100).round(2)

print(f"✓ Enriched sales: {sales_enriched.shape}")
print(f"  Columns added: Revenue, Profit, ProfitMargin")

# ============================================================================
# TASK 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 70)
print("TASK 2: FEATURE ENGINEERING")
print("=" * 70)

print("\n2.1 Creating date features...")
# Extract date components
sales_enriched['Year'] = sales_enriched['OrderDate'].dt.year
sales_enriched['Month'] = sales_enriched['OrderDate'].dt.month
sales_enriched['Quarter'] = sales_enriched['OrderDate'].dt.quarter
sales_enriched['DayOfWeek'] = sales_enriched['OrderDate'].dt.dayofweek
sales_enriched['DayOfWeekName'] = sales_enriched['OrderDate'].dt.day_name()
sales_enriched['MonthName'] = sales_enriched['OrderDate'].dt.month_name()
sales_enriched['IsWeekend'] = (sales_enriched['DayOfWeek'] >= 5).astype(int)
sales_enriched['IsMonthStart'] = sales_enriched['OrderDate'].dt.is_month_start.astype(int)
sales_enriched['IsMonthEnd'] = sales_enriched['OrderDate'].dt.is_month_end.astype(int)
sales_enriched['DayOfMonth'] = sales_enriched['OrderDate'].dt.day
sales_enriched['WeekOfYear'] = sales_enriched['OrderDate'].dt.isocalendar().week

print(f"✓ Date features created: Year, Month, Quarter, DayOfWeek, IsWeekend, etc.")

print("\n2.2 Calculating customer RFM metrics...")
reference_date = sales_enriched['OrderDate'].max()

# Recency
recency = sales_enriched.groupby('CustomerKey')['OrderDate'].max().reset_index()
recency['Recency_Days'] = (reference_date - recency['OrderDate']).dt.days
recency = recency[['CustomerKey', 'Recency_Days']]

# Frequency
frequency = sales_enriched.groupby('CustomerKey')['OrderNumber'].nunique().reset_index()
frequency.columns = ['CustomerKey', 'Frequency']

# Monetary
monetary = sales_enriched.groupby('CustomerKey')['Revenue'].sum().reset_index()
monetary.columns = ['CustomerKey', 'Monetary']

# Combine RFM
customer_rfm = recency.merge(frequency, on='CustomerKey')
customer_rfm = customer_rfm.merge(monetary, on='CustomerKey')

# Add more customer metrics
customer_stats = sales_enriched.groupby('CustomerKey').agg({
    'OrderDate': ['min', 'max'],
    'OrderNumber': 'nunique',
    'OrderQuantity': 'sum',
    'Revenue': ['sum', 'mean', 'std'],
    'Profit': 'sum'
}).reset_index()

customer_stats.columns = ['CustomerKey', 'FirstPurchaseDate', 'LastPurchaseDate',
                          'TotalOrders', 'TotalQuantity', 'TotalRevenue',
                          'AvgOrderValue', 'StdOrderValue', 'TotalProfit']

customer_rfm = customer_rfm.merge(customer_stats, on='CustomerKey')

# Customer lifetime (days)
customer_rfm['CustomerLifetime_Days'] = (customer_rfm['LastPurchaseDate'] - customer_rfm['FirstPurchaseDate']).dt.days
customer_rfm['AvgDaysBetweenOrders'] = (customer_rfm['CustomerLifetime_Days'] / customer_rfm['TotalOrders']).fillna(0)

# Merge with customer demographics
customer_rfm = customer_rfm.merge(customers, on='CustomerKey', how='left')

print(f"✓ RFM metrics calculated for {customer_rfm.shape[0]:,} customers")
print(f"  Metrics: Recency, Frequency, Monetary, Lifetime, AvgOrderValue, etc.")

print("\n2.3 Aggregating sales by time periods...")

# Daily aggregates
daily_sales = sales_enriched.groupby('OrderDate').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'OrderNumber': 'nunique',
    'OrderQuantity': 'sum',
    'CustomerKey': 'nunique'
}).reset_index()
daily_sales.columns = ['Date', 'Revenue', 'Profit', 'Orders', 'Quantity', 'Customers']

# Monthly aggregates
sales_enriched['YearMonth'] = sales_enriched['OrderDate'].dt.to_period('M')
monthly_sales = sales_enriched.groupby('YearMonth').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'OrderNumber': 'nunique',
    'OrderQuantity': 'sum',
    'CustomerKey': 'nunique'
}).reset_index()
monthly_sales.columns = ['YearMonth', 'Revenue', 'Profit', 'Orders', 'Quantity', 'Customers']
monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)

# Quarterly aggregates
sales_enriched['YearQuarter'] = sales_enriched['OrderDate'].dt.to_period('Q')
quarterly_sales = sales_enriched.groupby('YearQuarter').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'OrderNumber': 'nunique',
    'OrderQuantity': 'sum',
    'CustomerKey': 'nunique'
}).reset_index()
quarterly_sales.columns = ['YearQuarter', 'Revenue', 'Profit', 'Orders', 'Quantity', 'Customers']
quarterly_sales['YearQuarter'] = quarterly_sales['YearQuarter'].astype(str)

print(f"✓ Time aggregations created:")
print(f"  - Daily: {daily_sales.shape[0]} days")
print(f"  - Monthly: {monthly_sales.shape[0]} months")
print(f"  - Quarterly: {quarterly_sales.shape[0]} quarters")

print("\n2.4 Calculating product return rates...")

# Prepare returns data
returns['ReturnDate'] = pd.to_datetime(returns['ReturnDate'])
returns_enriched = returns.merge(products_full, on='ProductKey', how='left')

# Product-level return rates
product_sales = sales_enriched.groupby('ProductKey').agg({
    'OrderQuantity': 'sum',
    'Revenue': 'sum',
    'OrderNumber': 'nunique'
}).reset_index()
product_sales.columns = ['ProductKey', 'TotalSold', 'TotalRevenue', 'OrderCount']

product_returns = returns.groupby('ProductKey').agg({
    'ReturnQuantity': 'sum',
    'ReturnDate': 'count'
}).reset_index()
product_returns.columns = ['ProductKey', 'TotalReturned', 'ReturnCount']

# Merge and calculate rates
product_performance = product_sales.merge(product_returns, on='ProductKey', how='left')
product_performance = product_performance.merge(products_full[['ProductKey', 'ProductName', 'CategoryName', 'SubcategoryName']], on='ProductKey')

product_performance['TotalReturned'] = product_performance['TotalReturned'].fillna(0)
product_performance['ReturnCount'] = product_performance['ReturnCount'].fillna(0)
product_performance['ReturnRate_%'] = (product_performance['TotalReturned'] / product_performance['TotalSold'] * 100).round(2)
product_performance['AvgRevenue'] = (product_performance['TotalRevenue'] / product_performance['TotalSold']).round(2)

print(f"✓ Product return rates calculated for {product_performance.shape[0]} products")
print(f"  Overall return rate: {(product_performance['TotalReturned'].sum() / product_performance['TotalSold'].sum() * 100):.2f}%")

# ============================================================================
# TASK 3: DATA QUALITY CHECKS
# ============================================================================

print("\n" + "=" * 70)
print("TASK 3: DATA QUALITY CHECKS")
print("=" * 70)

print("\n3.1 Missing value analysis...")

datasets = {
    'Sales': sales,
    'Sales_Enriched': sales_enriched,
    'Customers': customers,
    'Products': products_full,
    'Returns': returns,
    'Customer_RFM': customer_rfm
}

missing_summary = {}
for name, df in datasets.items():
    missing_count = df.isnull().sum().sum()
    missing_pct = (missing_count / (df.shape[0] * df.shape[1]) * 100)
    missing_summary[name] = {
        'total_cells': df.shape[0] * df.shape[1],
        'missing_cells': missing_count,
        'missing_pct': round(missing_pct, 2)
    }

    if missing_count > 0:
        print(f"\n{name}:")
        cols_with_missing = df.isnull().sum()
        cols_with_missing = cols_with_missing[cols_with_missing > 0]
        for col, count in cols_with_missing.items():
            pct = (count / df.shape[0] * 100)
            print(f"  - {col}: {count:,} ({pct:.2f}%)")
    else:
        print(f"✓ {name}: No missing values")

print("\n3.2 Outlier detection...")

# Detect outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

print("\nSales Revenue outliers:")
outliers, lower, upper = detect_outliers_iqr(sales_enriched, 'Revenue')
print(f"  Total outliers: {len(outliers):,} ({len(outliers)/len(sales_enriched)*100:.2f}%)")
print(f"  Range: ${lower:.2f} - ${upper:.2f}")
print(f"  Max revenue: ${sales_enriched['Revenue'].max():.2f}")

print("\nCustomer Monetary outliers:")
outliers, lower, upper = detect_outliers_iqr(customer_rfm, 'Monetary')
print(f"  Total outliers: {len(outliers):,} ({len(outliers)/len(customer_rfm)*100:.2f}%)")
print(f"  Range: ${lower:.2f} - ${upper:.2f}")
print(f"  Max customer spend: ${customer_rfm['Monetary'].max():.2f}")

print("\n3.3 Data consistency validation...")

# Check for negative values
print("\nNegative value checks:")
print(f"  Negative OrderQuantity: {(sales_enriched['OrderQuantity'] < 0).sum()}")
print(f"  Negative Revenue: {(sales_enriched['Revenue'] < 0).sum()}")
print(f"  Negative Profit: {(sales_enriched['Profit'] < 0).sum()} (valid for losses)")

# Check for duplicates
print("\nDuplicate checks:")
print(f"  Duplicate sales rows: {sales_enriched.duplicated().sum()}")
print(f"  Duplicate customer IDs: {customer_rfm['CustomerKey'].duplicated().sum()}")

# Date consistency
print("\nDate consistency:")
invalid_dates = sales_enriched[sales_enriched['OrderDate'] > sales_enriched['StockDate']]
print(f"  Orders after stock date: {len(invalid_dates):,}")

# Key integrity
print("\nKey integrity:")
missing_products = sales_enriched['ProductKey'].isnull().sum()
missing_customers = sales_enriched['CustomerKey'].isnull().sum()
missing_territories = sales_enriched['TerritoryKey'].isnull().sum()
print(f"  Missing ProductKey: {missing_products}")
print(f"  Missing CustomerKey: {missing_customers}")
print(f"  Missing TerritoryKey: {missing_territories}")

# ============================================================================
# SAVE ALL PROCESSED DATASETS
# ============================================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED DATASETS")
print("=" * 70)

# Save all datasets
datasets_to_save = {
    'AdventureWorks_Sales_Enriched.csv': sales_enriched,
    'AdventureWorks_Customer_RFM.csv': customer_rfm,
    'AdventureWorks_Product_Performance.csv': product_performance,
    'AdventureWorks_Daily_Sales.csv': daily_sales,
    'AdventureWorks_Monthly_Sales.csv': monthly_sales,
    'AdventureWorks_Quarterly_Sales.csv': quarterly_sales,
}

for filename, df in datasets_to_save.items():
    filepath = processed_path / filename
    df.to_csv(filepath, index=False)
    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"✓ {filename} ({df.shape[0]:,} rows, {df.shape[1]} cols, {size_mb:.2f} MB)")

# ============================================================================
# GENERATE SUMMARY REPORT
# ============================================================================

print("\n" + "=" * 70)
print("PHASE 1 COMPLETION SUMMARY")
print("=" * 70)

print("\n✓ TASK 1: LOAD AND MERGE DATASETS - COMPLETE")
print(f"  - Sales merged: {sales.shape[0]:,} rows")
print(f"  - Joined with products, customers, territories")
print(f"  - Revenue and profit calculated")

print("\n✓ TASK 2: FEATURE ENGINEERING - COMPLETE")
print(f"  - Date features: 11 features created")
print(f"  - RFM metrics: {customer_rfm.shape[0]:,} customers analyzed")
print(f"  - Time aggregations: Daily, Monthly, Quarterly")
print(f"  - Product return rates: {product_performance.shape[0]} products")

print("\n✓ TASK 3: DATA QUALITY CHECKS - COMPLETE")
print(f"  - Missing value analysis: Done")
print(f"  - Outlier detection: Done")
print(f"  - Data consistency validation: Done")

print("\n✓ DATASETS SAVED: 6 processed files")

print("\n" + "=" * 70)
print("PHASE 1: DATA PREPARATION - COMPLETE ✓")
print("=" * 70)

print("\nNext: Phase 2 - Model Development")
