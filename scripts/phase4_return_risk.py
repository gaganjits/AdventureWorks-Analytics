"""
Phase 4: Return Risk Products Analysis
Calculate return rates by product and predict high-return risk products.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up paths
project_root = Path(__file__).parent.parent
data_path = project_root / 'data'
raw_path = data_path / 'raw'
processed_path = data_path / 'processed'

print("="*80)
print("PHASE 4: RETURN RISK PRODUCTS ANALYSIS")
print("="*80)
print()

# ============================================================================
# 1. Load Data
# ============================================================================
print("Step 1: Loading data...")
print("-" * 80)

# Load sales data
sales = pd.read_csv(processed_path / 'AdventureWorks_Sales_Enriched.csv')
print(f"✓ Loaded sales data: {sales.shape[0]:,} transactions")

# Load returns data
returns = pd.read_csv(raw_path / 'converted_AdventureWorks_Returns.csv')
print(f"✓ Loaded returns data: {returns.shape[0]:,} returns")

# Load products data
products = pd.read_csv(raw_path / 'AdventureWorks_Products.csv')
print(f"✓ Loaded products data: {products.shape[0]:,} products")

# Load categories
categories = pd.read_csv(raw_path / 'AdventureWorks_Product_Categories.csv')
subcategories = pd.read_csv(raw_path / 'AdventureWorks_Product_Subcategories.csv')
print(f"✓ Loaded categories: {categories.shape[0]:,} categories, {subcategories.shape[0]:,} subcategories")
print()

# ============================================================================
# 2. Calculate Return Rates by Product
# ============================================================================
print("Step 2: Calculating return rates by product...")
print("-" * 80)

# Calculate total sales quantity by product
product_sales = sales.groupby('ProductKey').agg({
    'OrderQuantity': 'sum',
    'Revenue': 'sum',
    'Profit': 'sum',
    'OrderNumber': 'count',  # Number of orders
    'ProductName': 'first',
    'ProductCost': 'first',
    'ProductPrice': 'first',
    'ProductColor': 'first',
    'ProductSize': 'first',
    'ProductStyle': 'first',
    'SubcategoryName': 'first',
    'CategoryName': 'first',
    'ProductSubcategoryKey': 'first',
    'ProductCategoryKey': 'first'
}).reset_index()

# Rename columns for clarity
product_sales.columns = ['ProductKey', 'TotalSalesQuantity', 'TotalRevenue', 'TotalProfit',
                         'TotalOrders', 'ProductName', 'ProductCost', 'ProductPrice',
                         'ProductColor', 'ProductSize', 'ProductStyle', 'SubcategoryName',
                         'CategoryName', 'ProductSubcategoryKey', 'ProductCategoryKey']

# Calculate total returns quantity by product
product_returns = returns.groupby('ProductKey').agg({
    'ReturnQuantity': 'sum',
    'ReturnDate': 'count'  # Number of return events
}).reset_index()

product_returns.columns = ['ProductKey', 'TotalReturnsQuantity', 'TotalReturnEvents']

# Merge sales and returns
product_analysis = product_sales.merge(product_returns, on='ProductKey', how='left')

# Fill NaN returns with 0 (products with no returns)
product_analysis['TotalReturnsQuantity'] = product_analysis['TotalReturnsQuantity'].fillna(0)
product_analysis['TotalReturnEvents'] = product_analysis['TotalReturnEvents'].fillna(0)

# Calculate return rate
product_analysis['ReturnRate'] = (product_analysis['TotalReturnsQuantity'] /
                                   product_analysis['TotalSalesQuantity']) * 100

# Calculate profit margin
product_analysis['ProfitMargin'] = (product_analysis['TotalProfit'] /
                                     product_analysis['TotalRevenue']) * 100

print(f"✓ Analyzed {product_analysis.shape[0]:,} products")
print(f"✓ Products with returns: {(product_analysis['TotalReturnsQuantity'] > 0).sum():,}")
print(f"✓ Products with no returns: {(product_analysis['TotalReturnsQuantity'] == 0).sum():,}")
print(f"✓ Overall return rate: {product_analysis['ReturnRate'].mean():.2f}%")
print(f"✓ Max return rate: {product_analysis['ReturnRate'].max():.2f}%")
print()

# ============================================================================
# 3. Define High Return Risk Threshold
# ============================================================================
print("Step 3: Defining high return risk threshold...")
print("-" * 80)

# Calculate percentile thresholds
percentile_75 = product_analysis['ReturnRate'].quantile(0.75)
percentile_90 = product_analysis['ReturnRate'].quantile(0.90)
mean_return = product_analysis['ReturnRate'].mean()
std_return = product_analysis['ReturnRate'].std()
mean_plus_std = mean_return + std_return

print(f"Return Rate Statistics:")
print(f"  - Mean: {mean_return:.2f}%")
print(f"  - Std: {std_return:.2f}%")
print(f"  - 75th percentile: {percentile_75:.2f}%")
print(f"  - 90th percentile: {percentile_90:.2f}%")
print(f"  - Mean + 1 Std: {mean_plus_std:.2f}%")
print()

# Use 75th percentile as threshold (more balanced classes)
high_return_threshold = percentile_75
print(f"✓ High Return Threshold: {high_return_threshold:.2f}%")
print()

# Create binary target variable
product_analysis['HighReturnRisk'] = (product_analysis['ReturnRate'] > high_return_threshold).astype(int)

print(f"Class Distribution:")
print(f"  - Normal Return (0): {(product_analysis['HighReturnRisk'] == 0).sum():,} products "
      f"({(product_analysis['HighReturnRisk'] == 0).sum() / len(product_analysis) * 100:.1f}%)")
print(f"  - High Return Risk (1): {(product_analysis['HighReturnRisk'] == 1).sum():,} products "
      f"({(product_analysis['HighReturnRisk'] == 1).sum() / len(product_analysis) * 100:.1f}%)")
print()

# ============================================================================
# 4. Feature Engineering
# ============================================================================
print("Step 4: Engineering product features...")
print("-" * 80)

# Price-based features
product_analysis['PriceRange'] = pd.cut(product_analysis['ProductPrice'],
                                        bins=[0, 50, 200, 1000, 5000],
                                        labels=['Budget', 'Mid-Range', 'Premium', 'Luxury'])

product_analysis['CostPriceRatio'] = product_analysis['ProductCost'] / product_analysis['ProductPrice']

# Sales volume features
product_analysis['AvgQuantityPerOrder'] = (product_analysis['TotalSalesQuantity'] /
                                           product_analysis['TotalOrders'])

product_analysis['AvgRevenuePerOrder'] = (product_analysis['TotalRevenue'] /
                                          product_analysis['TotalOrders'])

# Product popularity
product_analysis['SalesVolumeCategory'] = pd.cut(product_analysis['TotalSalesQuantity'],
                                                  bins=[0, 10, 50, 200, float('inf')],
                                                  labels=['Low', 'Medium', 'High', 'Very High'])

# Color and Size presence (binary features)
product_analysis['HasColor'] = (product_analysis['ProductColor'].notna() &
                                (product_analysis['ProductColor'] != '0')).astype(int)

product_analysis['HasSize'] = (product_analysis['ProductSize'].notna() &
                               (product_analysis['ProductSize'] != '0') &
                               (product_analysis['ProductSize'].astype(str) != '0')).astype(int)

product_analysis['HasStyle'] = (product_analysis['ProductStyle'].notna() &
                                (product_analysis['ProductStyle'] != 'U')).astype(int)

# Return-related features
product_analysis['ReturnFrequency'] = (product_analysis['TotalReturnEvents'] /
                                       product_analysis['TotalOrders']) * 100

# Handle any infinities or NaN values
product_analysis = product_analysis.replace([np.inf, -np.inf], np.nan)
numeric_columns = product_analysis.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    if product_analysis[col].isna().any():
        product_analysis[col] = product_analysis[col].fillna(product_analysis[col].median())

print(f"✓ Engineered {len(product_analysis.columns)} total features")
print()

# ============================================================================
# 5. Category-Level Analysis
# ============================================================================
print("Step 5: Analyzing category-level return patterns...")
print("-" * 80)

# Category-level statistics
category_stats = product_analysis.groupby('CategoryName').agg({
    'ReturnRate': ['mean', 'median', 'std', 'min', 'max'],
    'HighReturnRisk': ['sum', 'mean'],
    'ProductKey': 'count',
    'TotalSalesQuantity': 'sum',
    'TotalReturnsQuantity': 'sum',
    'TotalRevenue': 'sum',
    'ProfitMargin': 'mean'
}).round(2)

category_stats.columns = ['_'.join(col) for col in category_stats.columns]
category_stats = category_stats.reset_index()

print("Category-Level Return Patterns:")
print(category_stats.to_string())
print()

# Subcategory-level statistics
subcategory_stats = product_analysis.groupby(['CategoryName', 'SubcategoryName']).agg({
    'ReturnRate': 'mean',
    'HighReturnRisk': ['sum', 'mean'],
    'ProductKey': 'count',
    'TotalSalesQuantity': 'sum',
    'TotalReturnsQuantity': 'sum'
}).round(2)

subcategory_stats.columns = ['_'.join(col) for col in subcategory_stats.columns]
subcategory_stats = subcategory_stats.reset_index()
subcategory_stats = subcategory_stats.sort_values('ReturnRate_mean', ascending=False)

print("Top 10 Subcategories by Return Rate:")
print(subcategory_stats.head(10).to_string())
print()

# ============================================================================
# 6. Save Processed Data
# ============================================================================
print("Step 6: Saving processed data...")
print("-" * 80)

# Save main product analysis
output_file = processed_path / 'Product_Return_Risk_Features.csv'
product_analysis.to_csv(output_file, index=False)
print(f"✓ Saved product features: {output_file}")
print(f"  Shape: {product_analysis.shape}")

# Save category statistics
category_stats_file = processed_path / 'Category_Return_Statistics.csv'
category_stats.to_csv(category_stats_file, index=False)
print(f"✓ Saved category statistics: {category_stats_file}")

# Save subcategory statistics
subcategory_stats_file = processed_path / 'Subcategory_Return_Statistics.csv'
subcategory_stats.to_csv(subcategory_stats_file, index=False)
print(f"✓ Saved subcategory statistics: {subcategory_stats_file}")
print()

# ============================================================================
# 7. Summary Statistics
# ============================================================================
print("="*80)
print("PHASE 4 FEATURE ENGINEERING COMPLETE")
print("="*80)
print()
print("Summary:")
print(f"  Total Products Analyzed: {product_analysis.shape[0]:,}")
print(f"  Total Features Created: {product_analysis.shape[1]}")
print(f"  High Return Risk Products: {(product_analysis['HighReturnRisk'] == 1).sum():,}")
print(f"  Normal Return Products: {(product_analysis['HighReturnRisk'] == 0).sum():,}")
print(f"  Overall Return Rate: {product_analysis['ReturnRate'].mean():.2f}%")
print(f"  High Return Threshold: {high_return_threshold:.2f}%")
print()
print("Next Step: Run phase4_models_training.py to train classification models")
print("="*80)
