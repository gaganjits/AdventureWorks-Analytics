"""
Phase 2: Revenue Forecasting
Complete implementation of revenue forecasting models with multiple approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Time series models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet

# ML models
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Save models
import joblib

print("=" * 80)
print("PHASE 2: REVENUE FORECASTING")
print("=" * 80)

# Set paths
processed_path = Path('data/processed')
models_path = Path('models/revenue_forecasting')
outputs_path = Path('outputs/visualizations')
models_path.mkdir(parents=True, exist_ok=True)
outputs_path.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 6)

# ============================================================================
# TASK 1: AGGREGATE REVENUE BY QUARTER
# ============================================================================

print("\n" + "=" * 80)
print("TASK 1: AGGREGATE REVENUE AND PREPARE TIME SERIES DATA")
print("=" * 80)

print("\n1.1 Loading sales data...")
sales = pd.read_csv(processed_path / 'AdventureWorks_Sales_Enriched.csv')
sales['OrderDate'] = pd.to_datetime(sales['OrderDate'])

print(f"✓ Loaded {len(sales):,} sales records")
print(f"  Date range: {sales['OrderDate'].min()} to {sales['OrderDate'].max()}")

print("\n1.2 Creating time-based aggregations...")

# Daily revenue
daily_revenue = sales.groupby('OrderDate').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'OrderNumber': 'nunique',
    'OrderQuantity': 'sum'
}).reset_index()
daily_revenue.columns = ['Date', 'Revenue', 'Profit', 'Orders', 'Quantity']

# Weekly revenue
sales['Week'] = sales['OrderDate'].dt.to_period('W')
weekly_revenue = sales.groupby('Week').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'OrderNumber': 'nunique',
    'OrderQuantity': 'sum'
}).reset_index()
weekly_revenue['Week'] = weekly_revenue['Week'].dt.to_timestamp()
weekly_revenue.columns = ['Date', 'Revenue', 'Profit', 'Orders', 'Quantity']

# Monthly revenue
sales['Month'] = sales['OrderDate'].dt.to_period('M')
monthly_revenue = sales.groupby('Month').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'OrderNumber': 'nunique',
    'OrderQuantity': 'sum'
}).reset_index()
monthly_revenue['Month'] = monthly_revenue['Month'].dt.to_timestamp()
monthly_revenue.columns = ['Date', 'Revenue', 'Profit', 'Orders', 'Quantity']

# Quarterly revenue
sales['Quarter'] = sales['OrderDate'].dt.to_period('Q')
quarterly_revenue = sales.groupby('Quarter').agg({
    'Revenue': 'sum',
    'Profit': 'sum',
    'OrderNumber': 'nunique',
    'OrderQuantity': 'sum'
}).reset_index()
quarterly_revenue['Quarter'] = quarterly_revenue['Quarter'].dt.to_timestamp()
quarterly_revenue.columns = ['Date', 'Revenue', 'Profit', 'Orders', 'Quantity']

print(f"✓ Aggregations created:")
print(f"  - Daily: {len(daily_revenue)} days")
print(f"  - Weekly: {len(weekly_revenue)} weeks")
print(f"  - Monthly: {len(monthly_revenue)} months")
print(f"  - Quarterly: {len(quarterly_revenue)} quarters")

print("\n1.3 Quarterly revenue summary:")
print(quarterly_revenue)

# Save aggregations
quarterly_revenue.to_csv(processed_path / 'Revenue_Quarterly.csv', index=False)
monthly_revenue.to_csv(processed_path / 'Revenue_Monthly.csv', index=False)
weekly_revenue.to_csv(processed_path / 'Revenue_Weekly.csv', index=False)
daily_revenue.to_csv(processed_path / 'Revenue_Daily.csv', index=False)

print("\n✓ Revenue aggregations saved")

# ============================================================================
# TASK 2: FEATURE ENGINEERING FOR TIME SERIES
# ============================================================================

print("\n" + "=" * 80)
print("TASK 2: CREATE TIME SERIES FEATURES")
print("=" * 80)

# We'll use monthly data for forecasting (good balance between granularity and data points)
df = monthly_revenue.copy()
df = df.sort_values('Date').reset_index(drop=True)

print(f"\n2.1 Working with monthly data: {len(df)} months")

print("\n2.2 Creating lag features...")
# Lag features (1, 3, 6, 12 months)
for lag in [1, 2, 3, 6, 12]:
    df[f'Revenue_Lag_{lag}'] = df['Revenue'].shift(lag)
    df[f'Orders_Lag_{lag}'] = df['Orders'].shift(lag)

print(f"✓ Created lag features for 1, 2, 3, 6, 12 months")

print("\n2.3 Creating rolling averages...")
# Rolling averages (3, 6, 12 months)
for window in [3, 6, 12]:
    df[f'Revenue_MA_{window}'] = df['Revenue'].rolling(window=window, min_periods=1).mean()
    df[f'Revenue_STD_{window}'] = df['Revenue'].rolling(window=window, min_periods=1).std()

print(f"✓ Created rolling averages and std dev for 3, 6, 12 month windows")

print("\n2.4 Creating seasonality features...")
# Extract time components
df['Year'] = pd.to_datetime(df['Date']).dt.year
df['Month_Num'] = pd.to_datetime(df['Date']).dt.month
df['Quarter_Num'] = pd.to_datetime(df['Date']).dt.quarter
df['MonthsSinceStart'] = (df['Date'] - df['Date'].min()).dt.days / 30.44

# Cyclical encoding for month (sine/cosine)
df['Month_Sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
df['Month_Cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)

# Cyclical encoding for quarter
df['Quarter_Sin'] = np.sin(2 * np.pi * df['Quarter_Num'] / 4)
df['Quarter_Cos'] = np.cos(2 * np.pi * df['Quarter_Num'] / 4)

print(f"✓ Created seasonality features: Year, Month, Quarter, Cyclical encodings")

print("\n2.5 Creating growth rate features...")
df['Revenue_Growth_1M'] = df['Revenue'].pct_change(1) * 100
df['Revenue_Growth_3M'] = df['Revenue'].pct_change(3) * 100
df['Revenue_Growth_YoY'] = df['Revenue'].pct_change(12) * 100

print(f"✓ Created growth rate features")

# Save feature-engineered dataset
df.to_csv(processed_path / 'Revenue_Monthly_Features.csv', index=False)
print(f"\n✓ Feature-engineered dataset saved: {df.shape[1]} columns")

# Show feature summary
print(f"\nFeature categories:")
print(f"  - Original: Revenue, Profit, Orders, Quantity")
print(f"  - Lag features: {len([c for c in df.columns if 'Lag' in c])}")
print(f"  - Rolling features: {len([c for c in df.columns if 'MA' in c or 'STD' in c])}")
print(f"  - Seasonality: Year, Month, Quarter + cyclical encodings")
print(f"  - Growth rates: 1M, 3M, YoY")

# ============================================================================
# TASK 3: EXPLORATORY TIME SERIES ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("TASK 3: EXPLORATORY TIME SERIES ANALYSIS")
print("=" * 80)

# Plot revenue trend
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Revenue trend
axes[0, 0].plot(monthly_revenue['Date'], monthly_revenue['Revenue'], marker='o', linewidth=2)
axes[0, 0].set_title('Monthly Revenue Trend', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Revenue ($)')
axes[0, 0].grid(True, alpha=0.3)

# Profit trend
axes[0, 1].plot(monthly_revenue['Date'], monthly_revenue['Profit'], marker='o', color='green', linewidth=2)
axes[0, 1].set_title('Monthly Profit Trend', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Profit ($)')
axes[0, 1].grid(True, alpha=0.3)

# Orders trend
axes[1, 0].plot(monthly_revenue['Date'], monthly_revenue['Orders'], marker='o', color='orange', linewidth=2)
axes[1, 0].set_title('Monthly Order Count', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Number of Orders')
axes[1, 0].grid(True, alpha=0.3)

# Revenue distribution
axes[1, 1].hist(monthly_revenue['Revenue'], bins=15, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('Revenue Distribution', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Revenue ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(outputs_path / 'revenue_trends.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Revenue trends visualization saved")

# Seasonal decomposition
print("\nPerforming seasonal decomposition...")
ts_data = monthly_revenue.set_index('Date')['Revenue']
decomposition = seasonal_decompose(ts_data, model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(14, 10))
decomposition.observed.plot(ax=axes[0], title='Observed', legend=False)
axes[0].set_ylabel('Revenue')
decomposition.trend.plot(ax=axes[1], title='Trend', legend=False)
axes[1].set_ylabel('Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal', legend=False)
axes[2].set_ylabel('Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual', legend=False)
axes[3].set_ylabel('Residual')

plt.tight_layout()
plt.savefig(outputs_path / 'seasonal_decomposition.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Seasonal decomposition saved")

# ACF and PACF plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(ts_data.dropna(), lags=12, ax=axes[0])
axes[0].set_title('Autocorrelation Function (ACF)')
plot_pacf(ts_data.dropna(), lags=12, ax=axes[1])
axes[1].set_title('Partial Autocorrelation Function (PACF)')

plt.tight_layout()
plt.savefig(outputs_path / 'acf_pacf_plots.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ ACF/PACF plots saved")

print("\nContinuing with model building...")
print("(Script continues in next part)")
