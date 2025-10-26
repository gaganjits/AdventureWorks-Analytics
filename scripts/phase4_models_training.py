"""
Phase 4: Return Risk Products - Model Training
Train Random Forest and XGBoost classifiers to predict high-return risk products.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score, accuracy_score)
import xgboost as xgb
import joblib

# Set up paths
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'processed'
models_path = project_root / 'models'
outputs_path = project_root / 'outputs'
plots_path = outputs_path / 'plots'

# Create directories
models_path.mkdir(exist_ok=True)
plots_path.mkdir(exist_ok=True)

print("="*80)
print("PHASE 4: RETURN RISK PRODUCTS - MODEL TRAINING")
print("="*80)
print()

# ============================================================================
# 1. Load Data
# ============================================================================
print("Step 1: Loading data...")
print("-" * 80)

product_data = pd.read_csv(data_path / 'Product_Return_Risk_Features.csv')
print(f"✓ Loaded product features: {product_data.shape}")
print()

# ============================================================================
# 2. Prepare Features for Modeling
# ============================================================================
print("Step 2: Preparing features for modeling...")
print("-" * 80)

# Select features for modeling
numerical_features = [
    'ProductCost', 'ProductPrice', 'TotalSalesQuantity', 'TotalRevenue',
    'TotalProfit', 'TotalOrders', 'ReturnRate', 'ProfitMargin',
    'CostPriceRatio', 'AvgQuantityPerOrder', 'AvgRevenuePerOrder',
    'HasColor', 'HasSize', 'HasStyle', 'ReturnFrequency'
]

categorical_features = [
    'CategoryName', 'SubcategoryName', 'PriceRange', 'SalesVolumeCategory'
]

# Create feature matrix
X = product_data[numerical_features + categorical_features].copy()
y = product_data['HighReturnRisk'].copy()

# Encode categorical variables
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"✓ Feature matrix shape: {X.shape}")
print(f"✓ Numerical features: {len(numerical_features)}")
print(f"✓ Categorical features: {len(categorical_features)}")
print(f"✓ Target distribution:")
print(f"  - Normal (0): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"  - High Risk (1): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")
print()

# ============================================================================
# 3. Split Data
# ============================================================================
print("Step 3: Splitting data...")
print("-" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")
print()

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, models_path / 'phase4_feature_scaler.pkl')
print(f"✓ Saved feature scaler")
print()

# ============================================================================
# 4. Train Random Forest Classifier
# ============================================================================
print("Step 4: Training Random Forest Classifier...")
print("-" * 80)

# Train Random Forest with class weights to handle imbalance
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train_scaled)
y_test_pred_rf = rf_model.predict(X_test_scaled)
y_test_proba_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
rf_train_acc = accuracy_score(y_train, y_train_pred_rf)
rf_test_acc = accuracy_score(y_test, y_test_pred_rf)
rf_test_f1 = f1_score(y_test, y_test_pred_rf)
rf_test_auc = roc_auc_score(y_test, y_test_proba_rf)

print(f"Random Forest Results:")
print(f"  - Training Accuracy: {rf_train_acc:.4f}")
print(f"  - Test Accuracy: {rf_test_acc:.4f}")
print(f"  - Test F1-Score: {rf_test_f1:.4f}")
print(f"  - Test ROC-AUC: {rf_test_auc:.4f}")
print()

print("Classification Report (Random Forest):")
print(classification_report(y_test, y_test_pred_rf, target_names=['Normal', 'High Risk']))

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_rf = cross_val_score(rf_model, X_train_scaled, y_train, cv=cv, scoring='f1')
print(f"Cross-Validation F1-Score: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")
print()

# Save model
joblib.dump(rf_model, models_path / 'phase4_random_forest.pkl')
print(f"✓ Saved Random Forest model")
print()

# ============================================================================
# 5. Train XGBoost Classifier
# ============================================================================
print("Step 5: Training XGBoost Classifier...")
print("-" * 80)

# Calculate scale_pos_weight for imbalanced classes
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred_xgb = xgb_model.predict(X_train_scaled)
y_test_pred_xgb = xgb_model.predict(X_test_scaled)
y_test_proba_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
xgb_train_acc = accuracy_score(y_train, y_train_pred_xgb)
xgb_test_acc = accuracy_score(y_test, y_test_pred_xgb)
xgb_test_f1 = f1_score(y_test, y_test_pred_xgb)
xgb_test_auc = roc_auc_score(y_test, y_test_proba_xgb)

print(f"XGBoost Results:")
print(f"  - Training Accuracy: {xgb_train_acc:.4f}")
print(f"  - Test Accuracy: {xgb_test_acc:.4f}")
print(f"  - Test F1-Score: {xgb_test_f1:.4f}")
print(f"  - Test ROC-AUC: {xgb_test_auc:.4f}")
print()

print("Classification Report (XGBoost):")
print(classification_report(y_test, y_test_pred_xgb, target_names=['Normal', 'High Risk']))

# Cross-validation
cv_scores_xgb = cross_val_score(xgb_model, X_train_scaled, y_train, cv=cv, scoring='f1')
print(f"Cross-Validation F1-Score: {cv_scores_xgb.mean():.4f} (+/- {cv_scores_xgb.std():.4f})")
print()

# Save model
joblib.dump(xgb_model, models_path / 'phase4_xgboost.pkl')
print(f"✓ Saved XGBoost model")
print()

# ============================================================================
# 6. Feature Importance Analysis
# ============================================================================
print("Step 6: Analyzing feature importance...")
print("-" * 80)

# Random Forest feature importance
rf_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 Features (Random Forest):")
print(rf_feature_importance.head(10).to_string(index=False))
print()

# XGBoost feature importance
xgb_feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 10 Features (XGBoost):")
print(xgb_feature_importance.head(10).to_string(index=False))
print()

# Save feature importance
rf_feature_importance.to_csv(outputs_path / 'phase4_rf_feature_importance.csv', index=False)
xgb_feature_importance.to_csv(outputs_path / 'phase4_xgb_feature_importance.csv', index=False)

# ============================================================================
# 7. Visualizations
# ============================================================================
print("Step 7: Creating visualizations...")
print("-" * 80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# 1. Confusion Matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Random Forest Confusion Matrix
cm_rf = confusion_matrix(y_test, y_test_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Normal', 'High Risk'],
            yticklabels=['Normal', 'High Risk'])
axes[0].set_title(f'Random Forest Confusion Matrix\nAccuracy: {rf_test_acc:.3f}, F1: {rf_test_f1:.3f}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# XGBoost Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_test_pred_xgb)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Normal', 'High Risk'],
            yticklabels=['Normal', 'High Risk'])
axes[1].set_title(f'XGBoost Confusion Matrix\nAccuracy: {xgb_test_acc:.3f}, F1: {xgb_test_f1:.3f}')
axes[1].set_ylabel('Actual')
axes[1].set_xlabel('Predicted')

plt.tight_layout()
plt.savefig(plots_path / 'phase4_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved confusion matrices")

# 2. ROC Curves
plt.figure(figsize=(10, 6))

# Random Forest ROC
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_test_proba_rf)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_test_auc:.3f})', linewidth=2)

# XGBoost ROC
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_test_proba_xgb)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_test_auc:.3f})', linewidth=2)

# Diagonal line
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves - Return Risk Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(plots_path / 'phase4_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved ROC curves")

# 3. Precision-Recall Curves
plt.figure(figsize=(10, 6))

# Random Forest PR
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_test_proba_rf)
plt.plot(recall_rf, precision_rf, label=f'Random Forest', linewidth=2)

# XGBoost PR
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, y_test_proba_xgb)
plt.plot(recall_xgb, precision_xgb, label=f'XGBoost', linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves - Return Risk Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(plots_path / 'phase4_precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved precision-recall curves")

# 4. Feature Importance Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Random Forest Feature Importance
top_features_rf = rf_feature_importance.head(15)
axes[0].barh(range(len(top_features_rf)), top_features_rf['Importance'])
axes[0].set_yticks(range(len(top_features_rf)))
axes[0].set_yticklabels(top_features_rf['Feature'])
axes[0].set_xlabel('Importance')
axes[0].set_title('Top 15 Features - Random Forest')
axes[0].invert_yaxis()

# XGBoost Feature Importance
top_features_xgb = xgb_feature_importance.head(15)
axes[1].barh(range(len(top_features_xgb)), top_features_xgb['Importance'])
axes[1].set_yticks(range(len(top_features_xgb)))
axes[1].set_yticklabels(top_features_xgb['Feature'])
axes[1].set_xlabel('Importance')
axes[1].set_title('Top 15 Features - XGBoost')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(plots_path / 'phase4_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✓ Saved feature importance plots")
print()

# ============================================================================
# 8. Model Comparison Summary
# ============================================================================
print("="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)
print()

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Train Accuracy': [rf_train_acc, xgb_train_acc],
    'Test Accuracy': [rf_test_acc, xgb_test_acc],
    'Test F1-Score': [rf_test_f1, xgb_test_f1],
    'Test ROC-AUC': [rf_test_auc, xgb_test_auc],
    'CV F1-Score': [cv_scores_rf.mean(), cv_scores_xgb.mean()],
    'CV F1-Std': [cv_scores_rf.std(), cv_scores_xgb.std()]
})

print(comparison.to_string(index=False))
print()

# Determine best model
best_model_idx = comparison['Test F1-Score'].idxmax()
best_model_name = comparison.loc[best_model_idx, 'Model']
print(f"✓ Best Model: {best_model_name} (F1-Score: {comparison.loc[best_model_idx, 'Test F1-Score']:.4f})")
print()

# Save comparison
comparison.to_csv(outputs_path / 'phase4_model_comparison.csv', index=False)

# ============================================================================
# 9. High-Risk Product Analysis
# ============================================================================
print("="*80)
print("HIGH-RISK PRODUCT ANALYSIS")
print("="*80)
print()

# Get predictions on full dataset
full_predictions_rf = rf_model.predict(scaler.transform(X))
full_proba_rf = rf_model.predict_proba(scaler.transform(X))[:, 1]

# Add predictions to product data
product_data['Predicted_Risk_RF'] = full_predictions_rf
product_data['Risk_Probability_RF'] = full_proba_rf

# Identify high-risk products
high_risk_products = product_data[product_data['Predicted_Risk_RF'] == 1].sort_values(
    'Risk_Probability_RF', ascending=False
)

print(f"Identified {len(high_risk_products)} high-risk products:")
print()
print("Top 10 High-Risk Products:")
print(high_risk_products[['ProductName', 'CategoryName', 'SubcategoryName',
                          'ReturnRate', 'Risk_Probability_RF', 'TotalSalesQuantity',
                          'ProductPrice']].head(10).to_string(index=False))
print()

# Save high-risk products
high_risk_products.to_csv(outputs_path / 'phase4_high_risk_products.csv', index=False)
print(f"✓ Saved high-risk products list")
print()

print("="*80)
print("PHASE 4: MODEL TRAINING COMPLETE")
print("="*80)
print()
print("Summary:")
print(f"  Models Trained: Random Forest, XGBoost")
print(f"  Best Model: {best_model_name}")
print(f"  Test F1-Score: {comparison.loc[best_model_idx, 'Test F1-Score']:.4f}")
print(f"  Test ROC-AUC: {comparison.loc[best_model_idx, 'Test ROC-AUC']:.4f}")
print(f"  High-Risk Products Identified: {len(high_risk_products)}")
print()
print("Next Step: Review outputs/phase4_high_risk_products.csv for actionable insights")
print("="*80)
