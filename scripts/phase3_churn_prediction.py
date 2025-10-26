"""
Phase 3: Customer Churn Prediction
Complete implementation of churn prediction with multiple models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML models and preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Evaluation metrics
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score
)

import joblib

print("=" * 80)
print("PHASE 3: CUSTOMER CHURN PREDICTION")
print("=" * 80)

# Set paths
processed_path = Path('data/processed')
models_path = Path('models/churn_prediction')
outputs_path = Path('outputs/visualizations')
predictions_path = Path('outputs/predictions')
models_path.mkdir(parents=True, exist_ok=True)
predictions_path.mkdir(parents=True, exist_ok=True)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# ============================================================================
# TASK 1: DEFINE CHURN AND LABEL CUSTOMERS
# ============================================================================

print("\n" + "=" * 80)
print("TASK 1: DEFINE CHURN CRITERIA AND LABEL CUSTOMERS")
print("=" * 80)

print("\n1.1 Loading customer RFM data...")
customer_rfm = pd.read_csv(processed_path / 'AdventureWorks_Customer_RFM.csv')
customer_rfm['FirstPurchaseDate'] = pd.to_datetime(customer_rfm['FirstPurchaseDate'])
customer_rfm['LastPurchaseDate'] = pd.to_datetime(customer_rfm['LastPurchaseDate'])

print(f"✓ Loaded {len(customer_rfm):,} customers")

# Reference date (last date in dataset)
reference_date = customer_rfm['LastPurchaseDate'].max()
print(f"  Reference date: {reference_date.date()}")

print("\n1.2 Defining churn labels...")
# Churn Definition 1: No purchase in last 90 days
churn_threshold_90 = 90
customer_rfm['Churn_90'] = (customer_rfm['Recency_Days'] > churn_threshold_90).astype(int)

# Churn Definition 2: No purchase in last 180 days
churn_threshold_180 = 180
customer_rfm['Churn_180'] = (customer_rfm['Recency_Days'] > churn_threshold_180).astype(int)

print(f"\n90-Day Churn:")
print(f"  Churned: {customer_rfm['Churn_90'].sum():,} ({customer_rfm['Churn_90'].mean()*100:.2f}%)")
print(f"  Active:  {(~customer_rfm['Churn_90'].astype(bool)).sum():,} ({(1-customer_rfm['Churn_90'].mean())*100:.2f}%)")

print(f"\n180-Day Churn:")
print(f"  Churned: {customer_rfm['Churn_180'].sum():,} ({customer_rfm['Churn_180'].mean()*100:.2f}%)")
print(f"  Active:  {(~customer_rfm['Churn_180'].astype(bool)).sum():,} ({(1-customer_rfm['Churn_180'].mean())*100:.2f}%)")

# We'll use 90-day churn as our primary target
print(f"\n✓ Using 90-day churn as primary target")
print(f"  Class imbalance ratio: {customer_rfm['Churn_90'].mean():.2%} churned")

# ============================================================================
# TASK 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("TASK 2: FEATURE ENGINEERING FOR CHURN PREDICTION")
print("=" * 80)

print("\n2.1 RFM Features (already calculated)...")
rfm_features = ['Recency_Days', 'Frequency', 'Monetary']
print(f"✓ RFM features: {rfm_features}")

print("\n2.2 Creating additional customer behavior features...")
# Purchase frequency metrics
customer_rfm['AvgDaysBetweenPurchases'] = customer_rfm['AvgDaysBetweenOrders'].fillna(0)
customer_rfm['PurchaseFrequency'] = 1 / (customer_rfm['AvgDaysBetweenPurchases'] + 1)

# Revenue metrics
customer_rfm['AvgTransactionValue'] = customer_rfm['AvgOrderValue'].fillna(0)
customer_rfm['TotalLifetimeValue'] = customer_rfm['TotalRevenue']
customer_rfm['RevenueTrend'] = customer_rfm['TotalRevenue'] / (customer_rfm['CustomerLifetime_Days'] + 1)

# Engagement metrics
customer_rfm['EngagementScore'] = (
    (customer_rfm['Frequency'] / customer_rfm['Frequency'].max()) * 0.4 +
    (1 - customer_rfm['Recency_Days'] / customer_rfm['Recency_Days'].max()) * 0.3 +
    (customer_rfm['Monetary'] / customer_rfm['Monetary'].max()) * 0.3
)

# Recency categories
customer_rfm['Recency_Category'] = pd.cut(
    customer_rfm['Recency_Days'],
    bins=[0, 30, 60, 90, 180, float('inf')],
    labels=['Very_Recent', 'Recent', 'Moderate', 'At_Risk', 'Churned']
)

# Frequency categories
customer_rfm['Frequency_Category'] = pd.cut(
    customer_rfm['Frequency'],
    bins=[0, 1, 3, 5, float('inf')],
    labels=['One_Time', 'Occasional', 'Regular', 'Frequent']
)

# Monetary categories
customer_rfm['Monetary_Category'] = pd.qcut(
    customer_rfm['Monetary'],
    q=4,
    labels=['Low', 'Medium', 'High', 'VIP'],
    duplicates='drop'
)

print(f"✓ Created behavioral features: Engagement, Purchase Frequency, Revenue Trend")

print("\n2.3 Processing customer demographics...")
# Encode categorical demographics
demo_features = []

# Gender
if 'Gender' in customer_rfm.columns:
    customer_rfm['Gender'] = customer_rfm['Gender'].fillna('Unknown')
    customer_rfm['Gender_M'] = (customer_rfm['Gender'] == 'M').astype(int)
    customer_rfm['Gender_F'] = (customer_rfm['Gender'] == 'F').astype(int)
    demo_features.extend(['Gender_M', 'Gender_F'])

# Marital Status
if 'MaritalStatus' in customer_rfm.columns:
    customer_rfm['MaritalStatus'] = customer_rfm['MaritalStatus'].fillna('Unknown')
    customer_rfm['Marital_Married'] = (customer_rfm['MaritalStatus'] == 'M').astype(int)
    customer_rfm['Marital_Single'] = (customer_rfm['MaritalStatus'] == 'S').astype(int)
    demo_features.extend(['Marital_Married', 'Marital_Single'])

# Home Owner
if 'HomeOwner' in customer_rfm.columns:
    customer_rfm['HomeOwner'] = customer_rfm['HomeOwner'].fillna('Unknown')
    customer_rfm['Is_HomeOwner'] = (customer_rfm['HomeOwner'] == 'Y').astype(int)
    demo_features.append('Is_HomeOwner')

# Annual Income
if 'AnnualIncome' in customer_rfm.columns:
    # Convert string to numeric (remove $ and commas)
    customer_rfm['AnnualIncome'] = customer_rfm['AnnualIncome'].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
    customer_rfm['AnnualIncome'] = pd.to_numeric(customer_rfm['AnnualIncome'], errors='coerce')
    customer_rfm['AnnualIncome'] = customer_rfm['AnnualIncome'].fillna(customer_rfm['AnnualIncome'].median())
    customer_rfm['Income_HighEarner'] = (customer_rfm['AnnualIncome'] > customer_rfm['AnnualIncome'].median()).astype(int)
    demo_features.extend(['AnnualIncome', 'Income_HighEarner'])

# Total Children
if 'TotalChildren' in customer_rfm.columns:
    customer_rfm['TotalChildren'] = customer_rfm['TotalChildren'].fillna(0)
    customer_rfm['Has_Children'] = (customer_rfm['TotalChildren'] > 0).astype(int)
    demo_features.extend(['TotalChildren', 'Has_Children'])

# Education Level
if 'EducationLevel' in customer_rfm.columns:
    customer_rfm['EducationLevel'] = customer_rfm['EducationLevel'].fillna('Unknown')
    education_dummies = pd.get_dummies(customer_rfm['EducationLevel'], prefix='Edu')
    customer_rfm = pd.concat([customer_rfm, education_dummies], axis=1)
    demo_features.extend(education_dummies.columns.tolist())

# Occupation
if 'Occupation' in customer_rfm.columns:
    customer_rfm['Occupation'] = customer_rfm['Occupation'].fillna('Unknown')
    occupation_dummies = pd.get_dummies(customer_rfm['Occupation'], prefix='Occ')
    customer_rfm = pd.concat([customer_rfm, occupation_dummies], axis=1)
    demo_features.extend(occupation_dummies.columns.tolist())

print(f"✓ Processed demographic features: {len(demo_features)} features")

print("\n2.4 Loading product preferences...")
# Load sales data to get product preferences
sales = pd.read_csv(processed_path / 'AdventureWorks_Sales_Enriched.csv')
sales['OrderDate'] = pd.to_datetime(sales['OrderDate'])

# Calculate product category preferences per customer
category_prefs = sales.groupby(['CustomerKey', 'CategoryName']).agg({
    'Revenue': 'sum',
    'OrderQuantity': 'sum'
}).reset_index()

# Pivot to get category preferences as features
category_pivot = category_prefs.pivot_table(
    index='CustomerKey',
    columns='CategoryName',
    values='Revenue',
    fill_value=0
)
category_pivot.columns = [f'CategorySpend_{col}' for col in category_pivot.columns]

# Merge with customer data
customer_rfm = customer_rfm.merge(category_pivot, left_on='CustomerKey', right_index=True, how='left')
product_pref_features = category_pivot.columns.tolist()

# Fill missing category spends with 0
for col in product_pref_features:
    customer_rfm[col] = customer_rfm[col].fillna(0)

print(f"✓ Created product preference features: {len(product_pref_features)} categories")

# ============================================================================
# COMPILE FEATURE SET
# ============================================================================

print("\n2.5 Compiling final feature set...")

# RFM features
rfm_feats = ['Recency_Days', 'Frequency', 'Monetary', 'AvgTransactionValue',
             'TotalLifetimeValue', 'CustomerLifetime_Days', 'AvgDaysBetweenPurchases',
             'PurchaseFrequency', 'RevenueTrend', 'EngagementScore']

# Combine all features
all_features = rfm_feats + demo_features + product_pref_features

# Remove features with missing data or non-numeric
available_features = [f for f in all_features if f in customer_rfm.columns and customer_rfm[f].dtype in ['int64', 'float64']]

print(f"\n✓ Final feature set: {len(available_features)} features")
print(f"  - RFM & Behavioral: {len([f for f in available_features if f in rfm_feats])}")
print(f"  - Demographics: {len([f for f in available_features if f in demo_features])}")
print(f"  - Product Preferences: {len([f for f in available_features if f in product_pref_features])}")

# Save feature-engineered dataset
customer_rfm.to_csv(processed_path / 'Customer_Churn_Features.csv', index=False)
print(f"\n✓ Feature-engineered dataset saved")

# ============================================================================
# TASK 3: PREPARE DATA FOR MODELING
# ============================================================================

print("\n" + "=" * 80)
print("TASK 3: PREPARE DATA FOR MODELING")
print("=" * 80)

# Remove rows with NaN in features
df_model = customer_rfm[available_features + ['Churn_90', 'CustomerKey']].copy()
df_model = df_model.dropna()

print(f"\nDataset after removing NaN: {len(df_model):,} customers")

# Prepare features and target
X = df_model[available_features]
y = df_model['Churn_90']

print(f"\nFeature matrix: {X.shape}")
print(f"Target distribution:")
print(f"  Churned (1): {y.sum():,} ({y.mean()*100:.2f}%)")
print(f"  Active (0):  {(~y.astype(bool)).sum():,} ({(1-y.mean())*100:.2f}%)")

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train):,} customers")
print(f"Test set:  {len(X_test):,} customers")
print(f"  Test churned: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✓ Features scaled using StandardScaler")

# Save scaler
joblib.dump(scaler, models_path / 'feature_scaler.pkl')

print("\nContinuing with model training...")
print("(Script continues in phase3_models_training.py)")
