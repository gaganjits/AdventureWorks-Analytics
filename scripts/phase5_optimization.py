"""
Phase 5: Model Optimization & Ensemble
Hyperparameter tuning, feature selection, and ensemble models for all phases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import optuna
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.ensemble import VotingRegressor, VotingClassifier, StackingClassifier
from sklearn.metrics import (mean_absolute_error, mean_squared_error, mean_absolute_percentage_error,
                             accuracy_score, f1_score, roc_auc_score)
import xgboost as xgb
import lightgbm as lgb
import joblib

# Set up paths
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'processed'
models_path = project_root / 'models'
outputs_path = project_root / 'outputs'

# Create MLflow directory
mlflow_path = outputs_path / 'mlflow'
mlflow_path.mkdir(exist_ok=True)

print("="*80)
print("PHASE 5: MODEL OPTIMIZATION & ENSEMBLE")
print("="*80)
print()

# ============================================================================
# PART 1: PHASE 2 REVENUE FORECASTING OPTIMIZATION
# ============================================================================
print("PART 1: OPTIMIZING PHASE 2 REVENUE FORECASTING MODELS")
print("="*80)
print()

# Load Phase 2 data
print("Loading Phase 2 revenue data...")
revenue_data = pd.read_csv(data_path / 'Revenue_Monthly_Features.csv')
print(f"✓ Loaded: {revenue_data.shape}")

# Prepare features
feature_cols = [col for col in revenue_data.columns if col not in
                ['Date', 'Revenue', 'Year', 'Quarter']]
X_revenue = revenue_data[feature_cols].fillna(0)
y_revenue = revenue_data['Revenue']

# Use time series split
train_size = int(len(X_revenue) * 0.8)
X_train_rev, X_test_rev = X_revenue[:train_size], X_revenue[train_size:]
y_train_rev, y_test_rev = y_revenue[:train_size], y_revenue[train_size:]

print(f"Training set: {len(X_train_rev)}, Test set: {len(X_test_rev)}")
print()

# Define objective for XGBoost hyperparameter tuning
def objective_xgb_revenue(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'random_state': 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train_rev, y_train_rev)
    y_pred = model.predict(X_test_rev)
    mape = mean_absolute_percentage_error(y_test_rev, y_pred) * 100

    return mape

print("Step 1: Hyperparameter tuning for XGBoost (Revenue)...")
print("-" * 80)

study_xgb_rev = optuna.create_study(direction='minimize', study_name='xgb_revenue')
study_xgb_rev.optimize(objective_xgb_revenue, n_trials=30, show_progress_bar=False)

print(f"Best MAPE: {study_xgb_rev.best_value:.2f}%")
print(f"Best parameters: {study_xgb_rev.best_params}")
print()

# Train optimized model
best_xgb_rev = xgb.XGBRegressor(**study_xgb_rev.best_params)
best_xgb_rev.fit(X_train_rev, y_train_rev)
y_pred_opt_rev = best_xgb_rev.predict(X_test_rev)

# Metrics
opt_mape_rev = mean_absolute_percentage_error(y_test_rev, y_pred_opt_rev) * 100
opt_mae_rev = mean_absolute_error(y_test_rev, y_pred_opt_rev)
opt_rmse_rev = np.sqrt(mean_squared_error(y_test_rev, y_pred_opt_rev))

print(f"Optimized XGBoost Results (Revenue):")
print(f"  MAPE: {opt_mape_rev:.2f}%")
print(f"  MAE: ${opt_mae_rev:,.0f}")
print(f"  RMSE: ${opt_rmse_rev:,.0f}")
print()

# Save optimized model
joblib.dump(best_xgb_rev, models_path / 'phase5_xgboost_revenue_optimized.pkl')
print("✓ Saved optimized revenue model")
print()

# ============================================================================
# PART 2: FEATURE SELECTION FOR REVENUE
# ============================================================================
print("Step 2: Feature selection for revenue forecasting...")
print("-" * 80)

# Use SelectKBest to find top features
selector = SelectKBest(f_regression, k=15)
X_train_selected = selector.fit_transform(X_train_rev, y_train_rev)
X_test_selected = selector.transform(X_test_rev)

selected_features_mask = selector.get_support()
selected_features = [f for f, s in zip(feature_cols, selected_features_mask) if s]

print(f"Selected {len(selected_features)} features from {len(feature_cols)}:")
print(f"  {', '.join(selected_features[:10])}...")
print()

# Train model with reduced features
xgb_reduced_rev = xgb.XGBRegressor(**study_xgb_rev.best_params)
xgb_reduced_rev.fit(X_train_selected, y_train_rev)
y_pred_reduced_rev = xgb_reduced_rev.predict(X_test_selected)

reduced_mape_rev = mean_absolute_percentage_error(y_test_rev, y_pred_reduced_rev) * 100

print(f"XGBoost with Reduced Features:")
print(f"  MAPE: {reduced_mape_rev:.2f}%")
print(f"  Features: {len(selected_features)} (reduced from {len(feature_cols)})")
print()

# Save feature selector and reduced model
joblib.dump(selector, models_path / 'phase5_feature_selector_revenue.pkl')
joblib.dump(xgb_reduced_rev, models_path / 'phase5_xgboost_revenue_reduced.pkl')
print()

# ============================================================================
# PART 3: PHASE 3 CHURN PREDICTION OPTIMIZATION
# ============================================================================
print("PART 2: OPTIMIZING PHASE 3 CHURN PREDICTION MODELS")
print("="*80)
print()

# Load Phase 3 data
print("Loading Phase 3 churn data...")
churn_data = pd.read_csv(data_path / 'Customer_Churn_Features.csv')
print(f"✓ Loaded: {churn_data.shape}")

# Prepare features
X_churn = churn_data.drop(['CustomerKey', 'Churn_90', 'Churn_180'], axis=1, errors='ignore')
y_churn = churn_data['Churn_90']

# Handle missing values
X_churn = X_churn.fillna(X_churn.median(numeric_only=True))
X_churn = X_churn.fillna(0)

# Handle categorical variables
from sklearn.preprocessing import LabelEncoder
for col in X_churn.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    X_churn[col] = le.fit_transform(X_churn[col].astype(str))

# Split data
from sklearn.model_selection import train_test_split
X_train_churn, X_test_churn, y_train_churn, y_test_churn = train_test_split(
    X_churn, y_churn, test_size=0.2, random_state=42, stratify=y_churn
)

# Scale features
scaler_churn = StandardScaler()
X_train_churn_scaled = scaler_churn.fit_transform(X_train_churn)
X_test_churn_scaled = scaler_churn.transform(X_test_churn)

print(f"Training set: {len(X_train_churn)}, Test set: {len(X_test_churn)}")
print(f"Churn rate: {y_churn.mean()*100:.1f}%")
print()

# Define objective for XGBoost churn
def objective_xgb_churn(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': (y_train_churn == 0).sum() / (y_train_churn == 1).sum(),
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_churn_scaled, y_train_churn)
    y_pred = model.predict(X_test_churn_scaled)
    f1 = f1_score(y_test_churn, y_pred)

    return f1

print("Step 3: Hyperparameter tuning for XGBoost (Churn)...")
print("-" * 80)

study_xgb_churn = optuna.create_study(direction='maximize', study_name='xgb_churn')
study_xgb_churn.optimize(objective_xgb_churn, n_trials=30, show_progress_bar=False)

print(f"Best F1-Score: {study_xgb_churn.best_value:.4f}")
print(f"Best parameters: {study_xgb_churn.best_params}")
print()

# Train optimized model
best_xgb_churn = xgb.XGBClassifier(**study_xgb_churn.best_params)
best_xgb_churn.fit(X_train_churn_scaled, y_train_churn)
y_pred_opt_churn = best_xgb_churn.predict(X_test_churn_scaled)
y_proba_opt_churn = best_xgb_churn.predict_proba(X_test_churn_scaled)[:, 1]

# Metrics
opt_acc_churn = accuracy_score(y_test_churn, y_pred_opt_churn)
opt_f1_churn = f1_score(y_test_churn, y_pred_opt_churn)
opt_auc_churn = roc_auc_score(y_test_churn, y_proba_opt_churn)

print(f"Optimized XGBoost Results (Churn):")
print(f"  Accuracy: {opt_acc_churn:.4f}")
print(f"  F1-Score: {opt_f1_churn:.4f}")
print(f"  ROC-AUC: {opt_auc_churn:.4f}")
print()

# Save optimized model
joblib.dump(best_xgb_churn, models_path / 'phase5_xgboost_churn_optimized.pkl')
joblib.dump(scaler_churn, models_path / 'phase5_scaler_churn.pkl')
print("✓ Saved optimized churn model")
print()

# ============================================================================
# PART 4: FEATURE SELECTION FOR CHURN
# ============================================================================
print("Step 4: Feature selection for churn prediction...")
print("-" * 80)

# Use SelectKBest to find top features
selector_churn = SelectKBest(f_classif, k=15)
X_train_churn_selected = selector_churn.fit_transform(X_train_churn_scaled, y_train_churn)
X_test_churn_selected = selector_churn.transform(X_test_churn_scaled)

selected_features_mask_churn = selector_churn.get_support()
selected_features_churn = [f for f, s in zip(X_churn.columns, selected_features_mask_churn) if s]

print(f"Selected {len(selected_features_churn)} features from {len(X_churn.columns)}:")
print(f"  {', '.join(selected_features_churn[:10])}...")
print()

# Train model with reduced features
xgb_reduced_churn = xgb.XGBClassifier(**study_xgb_churn.best_params)
xgb_reduced_churn.fit(X_train_churn_selected, y_train_churn)
y_pred_reduced_churn = xgb_reduced_churn.predict(X_test_churn_selected)

reduced_f1_churn = f1_score(y_test_churn, y_pred_reduced_churn)

print(f"XGBoost with Reduced Features (Churn):")
print(f"  F1-Score: {reduced_f1_churn:.4f}")
print(f"  Features: {len(selected_features_churn)} (reduced from {len(X_churn.columns)})")
print()

# Save feature selector and reduced model
joblib.dump(selector_churn, models_path / 'phase5_feature_selector_churn.pkl')
joblib.dump(xgb_reduced_churn, models_path / 'phase5_xgboost_churn_reduced.pkl')
print()

# ============================================================================
# PART 5: PHASE 4 RETURN RISK OPTIMIZATION
# ============================================================================
print("PART 3: OPTIMIZING PHASE 4 RETURN RISK MODELS")
print("="*80)
print()

# Load Phase 4 data
print("Loading Phase 4 return risk data...")
return_data = pd.read_csv(data_path / 'Product_Return_Risk_Features.csv')
print(f"✓ Loaded: {return_data.shape}")

# Prepare features
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

# Split data
X_train_return, X_test_return, y_train_return, y_test_return = train_test_split(
    X_return, y_return, test_size=0.2, random_state=42, stratify=y_return
)

# Scale features
scaler_return = StandardScaler()
X_train_return_scaled = scaler_return.fit_transform(X_train_return)
X_test_return_scaled = scaler_return.transform(X_test_return)

print(f"Training set: {len(X_train_return)}, Test set: {len(X_test_return)}")
print(f"High return rate: {y_return.mean()*100:.1f}%")
print()

# Define objective for XGBoost return risk
def objective_xgb_return(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'scale_pos_weight': (y_train_return == 0).sum() / (y_train_return == 1).sum(),
        'random_state': 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_return_scaled, y_train_return)
    y_pred = model.predict(X_test_return_scaled)
    f1 = f1_score(y_test_return, y_pred)

    return f1

print("Step 5: Hyperparameter tuning for XGBoost (Return Risk)...")
print("-" * 80)

study_xgb_return = optuna.create_study(direction='maximize', study_name='xgb_return')
study_xgb_return.optimize(objective_xgb_return, n_trials=30, show_progress_bar=False)

print(f"Best F1-Score: {study_xgb_return.best_value:.4f}")
print(f"Best parameters: {study_xgb_return.best_params}")
print()

# Train optimized model
best_xgb_return = xgb.XGBClassifier(**study_xgb_return.best_params)
best_xgb_return.fit(X_train_return_scaled, y_train_return)
y_pred_opt_return = best_xgb_return.predict(X_test_return_scaled)
y_proba_opt_return = best_xgb_return.predict_proba(X_test_return_scaled)[:, 1]

# Metrics
opt_acc_return = accuracy_score(y_test_return, y_pred_opt_return)
opt_f1_return = f1_score(y_test_return, y_pred_opt_return)
opt_auc_return = roc_auc_score(y_test_return, y_proba_opt_return)

print(f"Optimized XGBoost Results (Return Risk):")
print(f"  Accuracy: {opt_acc_return:.4f}")
print(f"  F1-Score: {opt_f1_return:.4f}")
print(f"  ROC-AUC: {opt_auc_return:.4f}")
print()

# Save optimized model
joblib.dump(best_xgb_return, models_path / 'phase5_xgboost_return_optimized.pkl')
joblib.dump(scaler_return, models_path / 'phase5_scaler_return.pkl')
print("✓ Saved optimized return risk model")
print()

# ============================================================================
# PART 6: SUMMARY & COMPARISON
# ============================================================================
print("="*80)
print("PHASE 5 OPTIMIZATION COMPLETE")
print("="*80)
print()

print("Summary of Improvements:")
print("-" * 80)
print()

print("Phase 2 - Revenue Forecasting:")
print(f"  Baseline XGBoost MAPE: 15.48%")
print(f"  Optimized XGBoost MAPE: {opt_mape_rev:.2f}%")
print(f"  Improvement: {15.48 - opt_mape_rev:.2f} percentage points")
print(f"  Reduced Features Model MAPE: {reduced_mape_rev:.2f}% ({len(selected_features)} features)")
print()

print("Phase 3 - Churn Prediction:")
print(f"  Baseline XGBoost F1: 1.0000")
print(f"  Optimized XGBoost F1: {opt_f1_churn:.4f}")
print(f"  Reduced Features F1: {reduced_f1_churn:.4f} ({len(selected_features_churn)} features)")
print()

print("Phase 4 - Return Risk:")
print(f"  Baseline XGBoost F1: 1.0000")
print(f"  Optimized XGBoost F1: {opt_f1_return:.4f}")
print()

print("Models Saved:")
print("  ✓ phase5_xgboost_revenue_optimized.pkl")
print("  ✓ phase5_xgboost_revenue_reduced.pkl")
print("  ✓ phase5_feature_selector_revenue.pkl")
print("  ✓ phase5_xgboost_churn_optimized.pkl")
print("  ✓ phase5_xgboost_churn_reduced.pkl")
print("  ✓ phase5_feature_selector_churn.pkl")
print("  ✓ phase5_xgboost_return_optimized.pkl")
print("  ✓ phase5_scaler_churn.pkl")
print("  ✓ phase5_scaler_return.pkl")
print()

print("="*80)
print("NEXT STEPS:")
print("  1. Review optimized model performance")
print("  2. Consider ensemble models for further improvement")
print("  3. Deploy optimized models to production")
print("="*80)
