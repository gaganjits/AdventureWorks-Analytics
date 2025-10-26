"""
Phase 5: Ensemble Models
Create ensemble models combining multiple algorithms for improved predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_absolute_percentage_error, mean_absolute_error,
                             accuracy_score, f1_score, roc_auc_score)
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib

# Set up paths
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'processed'
models_path = project_root / 'models'

print("="*80)
print("PHASE 5: ENSEMBLE MODELS")
print("="*80)
print()

# ============================================================================
# PART 1: REVENUE FORECASTING ENSEMBLE
# ============================================================================
print("PART 1: REVENUE FORECASTING ENSEMBLE")
print("="*80)
print()

# Load Phase 2 data
print("Loading Phase 2 revenue data...")
revenue_data = pd.read_csv(data_path / 'Revenue_Monthly_Features.csv')

feature_cols = [col for col in revenue_data.columns if col not in
                ['Date', 'Revenue', 'Year', 'Quarter']]
X_revenue = revenue_data[feature_cols].fillna(0)
y_revenue = revenue_data['Revenue']

train_size = int(len(X_revenue) * 0.8)
X_train_rev, X_test_rev = X_revenue[:train_size], X_revenue[train_size:]
y_train_rev, y_test_rev = y_revenue[:train_size], y_revenue[train_size:]

print(f"✓ Loaded: {revenue_data.shape}")
print()

# Create ensemble with multiple regressors
print("Creating Voting Regressor Ensemble...")
print("-" * 80)

# Load optimized model
optimized_xgb = joblib.load(models_path / 'phase5_xgboost_revenue_optimized.pkl')

# Define ensemble members
ensemble_rev = VotingRegressor(
    estimators=[
        ('xgb_optimized', optimized_xgb),
        ('lgb', lgb.LGBMRegressor(n_estimators=150, learning_rate=0.1, max_depth=7, random_state=42)),
        ('rf', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
    ],
    weights=[2, 1, 1]  # Give more weight to optimized XGBoost
)

# Train ensemble
ensemble_rev.fit(X_train_rev, y_train_rev)
y_pred_ensemble_rev = ensemble_rev.predict(X_test_rev)

# Metrics
ensemble_mape_rev = mean_absolute_percentage_error(y_test_rev, y_pred_ensemble_rev) * 100
ensemble_mae_rev = mean_absolute_error(y_test_rev, y_pred_ensemble_rev)

print(f"Voting Regressor Ensemble Results:")
print(f"  MAPE: {ensemble_mape_rev:.2f}%")
print(f"  MAE: ${ensemble_mae_rev:,.0f}")
print(f"  Ensemble members: XGBoost (optimized), LightGBM, Random Forest")
print(f"  Weights: [2, 1, 1]")
print()

# Save ensemble
joblib.dump(ensemble_rev, models_path / 'phase5_ensemble_revenue.pkl')
print("✓ Saved revenue ensemble model")
print()

# ============================================================================
# PART 2: CHURN PREDICTION ENSEMBLE
# ============================================================================
print("PART 2: CHURN PREDICTION ENSEMBLE")
print("="*80)
print()

# Load Phase 3 data
print("Loading Phase 3 churn data...")
churn_data = pd.read_csv(data_path / 'Customer_Churn_Features.csv')

X_churn = churn_data.drop(['CustomerKey', 'Churn_90', 'Churn_180'], axis=1, errors='ignore')
y_churn = churn_data['Churn_90']

# Handle missing values
X_churn = X_churn.fillna(X_churn.median(numeric_only=True))
X_churn = X_churn.fillna(0)

# Handle categorical variables
for col in X_churn.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_churn[col] = le.fit_transform(X_churn[col].astype(str))

# Split and scale
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

scaler_churn = StandardScaler()
X_train_churn_scaled = scaler_churn.fit_transform(X_train_churn)
X_test_churn_scaled = scaler_churn.transform(X_test_churn)

print(f"✓ Loaded: {churn_data.shape}")
print()

# Create ensemble with multiple classifiers
print("Creating Voting Classifier Ensemble...")
print("-" * 80)

# Load optimized model
optimized_xgb_churn = joblib.load(models_path / 'phase5_xgboost_churn_optimized.pkl')

# Calculate scale_pos_weight
scale_pos_weight_churn = (y_train_churn == 0).sum() / (y_train_churn == 1).sum()

# Define ensemble members
ensemble_churn = VotingClassifier(
    estimators=[
        ('xgb_optimized', optimized_xgb_churn),
        ('lgb', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=7,
                                    scale_pos_weight=scale_pos_weight_churn, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                       class_weight='balanced', random_state=42))
    ],
    voting='soft',
    weights=[2, 1, 1]  # Give more weight to optimized XGBoost
)

# Train ensemble
ensemble_churn.fit(X_train_churn_scaled, y_train_churn)
y_pred_ensemble_churn = ensemble_churn.predict(X_test_churn_scaled)
y_proba_ensemble_churn = ensemble_churn.predict_proba(X_test_churn_scaled)[:, 1]

# Metrics
ensemble_acc_churn = accuracy_score(y_test_churn, y_pred_ensemble_churn)
ensemble_f1_churn = f1_score(y_test_churn, y_pred_ensemble_churn)
ensemble_auc_churn = roc_auc_score(y_test_churn, y_proba_ensemble_churn)

print(f"Voting Classifier Ensemble Results:")
print(f"  Accuracy: {ensemble_acc_churn:.4f}")
print(f"  F1-Score: {ensemble_f1_churn:.4f}")
print(f"  ROC-AUC: {ensemble_auc_churn:.4f}")
print(f"  Ensemble members: XGBoost (optimized), LightGBM, Random Forest")
print(f"  Voting: soft, Weights: [2, 1, 1]")
print()

# Save ensemble
joblib.dump(ensemble_churn, models_path / 'phase5_ensemble_churn.pkl')
print("✓ Saved churn ensemble model")
print()

# ============================================================================
# PART 3: RETURN RISK ENSEMBLE
# ============================================================================
print("PART 3: RETURN RISK ENSEMBLE")
print("="*80)
print()

# Load Phase 4 data
print("Loading Phase 4 return risk data...")
return_data = pd.read_csv(data_path / 'Product_Return_Risk_Features.csv')

feature_cols_return = [
    'ProductCost', 'ProductPrice', 'TotalSalesQuantity', 'TotalRevenue',
    'TotalProfit', 'TotalOrders', 'ReturnRate', 'ProfitMargin',
    'CostPriceRatio', 'AvgQuantityPerOrder', 'AvgRevenuePerOrder',
    'HasColor', 'HasSize', 'HasStyle', 'ReturnFrequency'
]

categorical_features_return = ['CategoryName', 'SubcategoryName', 'PriceRange', 'SalesVolumeCategory']

X_return = return_data[feature_cols_return + categorical_features_return].copy()
y_return = return_data['HighReturnRisk']

# Encode categorical variables
for col in categorical_features_return:
    le = LabelEncoder()
    X_return[col] = le.fit_transform(X_return[col].astype(str))

# Split and scale
X_train_return, X_test_return, y_train_return, y_test_return = train_test_split(
    X_return, y_return, test_size=0.2, random_state=42, stratify=y_return
)

scaler_return = StandardScaler()
X_train_return_scaled = scaler_return.fit_transform(X_train_return)
X_test_return_scaled = scaler_return.transform(X_test_return)

print(f"✓ Loaded: {return_data.shape}")
print()

# Create ensemble
print("Creating Voting Classifier Ensemble...")
print("-" * 80)

# Load optimized model
optimized_xgb_return = joblib.load(models_path / 'phase5_xgboost_return_optimized.pkl')

# Calculate scale_pos_weight
scale_pos_weight_return = (y_train_return == 0).sum() / (y_train_return == 1).sum()

# Define ensemble members
ensemble_return = VotingClassifier(
    estimators=[
        ('xgb_optimized', optimized_xgb_return),
        ('lgb', lgb.LGBMClassifier(n_estimators=150, learning_rate=0.1, max_depth=7,
                                    scale_pos_weight=scale_pos_weight_return, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                       class_weight='balanced', random_state=42))
    ],
    voting='soft',
    weights=[2, 1, 1]
)

# Train ensemble
ensemble_return.fit(X_train_return_scaled, y_train_return)
y_pred_ensemble_return = ensemble_return.predict(X_test_return_scaled)
y_proba_ensemble_return = ensemble_return.predict_proba(X_test_return_scaled)[:, 1]

# Metrics
ensemble_acc_return = accuracy_score(y_test_return, y_pred_ensemble_return)
ensemble_f1_return = f1_score(y_test_return, y_pred_ensemble_return)
ensemble_auc_return = roc_auc_score(y_test_return, y_proba_ensemble_return)

print(f"Voting Classifier Ensemble Results:")
print(f"  Accuracy: {ensemble_acc_return:.4f}")
print(f"  F1-Score: {ensemble_f1_return:.4f}")
print(f"  ROC-AUC: {ensemble_auc_return:.4f}")
print(f"  Ensemble members: XGBoost (optimized), LightGBM, Random Forest")
print(f"  Voting: soft, Weights: [2, 1, 1]")
print()

# Save ensemble
joblib.dump(ensemble_return, models_path / 'phase5_ensemble_return.pkl')
print("✓ Saved return risk ensemble model")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("="*80)
print("PHASE 5 ENSEMBLE MODELS COMPLETE")
print("="*80)
print()

print("Summary:")
print("-" * 80)
print()

print("Revenue Forecasting Ensemble:")
print(f"  MAPE: {ensemble_mape_rev:.2f}%")
print(f"  MAE: ${ensemble_mae_rev:,.0f}")
print()

print("Churn Prediction Ensemble:")
print(f"  Accuracy: {ensemble_acc_churn:.4f}")
print(f"  F1-Score: {ensemble_f1_churn:.4f}")
print(f"  ROC-AUC: {ensemble_auc_churn:.4f}")
print()

print("Return Risk Ensemble:")
print(f"  Accuracy: {ensemble_acc_return:.4f}")
print(f"  F1-Score: {ensemble_f1_return:.4f}")
print(f"  ROC-AUC: {ensemble_auc_return:.4f}")
print()

print("Models Saved:")
print("  ✓ phase5_ensemble_revenue.pkl")
print("  ✓ phase5_ensemble_churn.pkl")
print("  ✓ phase5_ensemble_return.pkl")
print()

print("="*80)
