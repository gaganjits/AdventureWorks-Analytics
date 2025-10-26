"""
Phase 3: Churn Prediction - Model Training & Evaluation
Implements Logistic Regression, Random Forest, and XGBoost with SMOTE
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load the prepared data from phase3_churn_prediction.py
# Since we need to continue from where we left off, we'll reload and process

# ML models and preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# Evaluation metrics
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, accuracy_score, precision_score, recall_score,
    average_precision_score
)

import joblib

print("=" * 80)
print("PHASE 3: CHURN PREDICTION - MODEL TRAINING")
print("=" * 80)

# Set paths
processed_path = Path('data/processed')
models_path = Path('models/churn_prediction')
outputs_path = Path('outputs/visualizations')
predictions_path = Path('outputs/predictions')

# Load feature-engineered data
print("\nLoading feature-engineered churn data...")
customer_churn = pd.read_csv(processed_path / 'Customer_Churn_Features.csv')

# Get feature columns (exclude non-feature columns)
exclude_cols = ['CustomerKey', 'Churn_90', 'Churn_180', 'FirstPurchaseDate', 'LastPurchaseDate',
                'Prefix', 'FirstName', 'LastName', 'BirthDate', 'MaritalStatus', 'Gender',
                'EmailAddress', 'EducationLevel', 'Occupation', 'HomeOwner',
                'Recency_Category', 'Frequency_Category', 'Monetary_Category', 'StdOrderValue']

# Select numeric features
numeric_features = customer_churn.select_dtypes(include=['int64', 'float64']).columns.tolist()
feature_cols = [col for col in numeric_features if col not in exclude_cols and 'Churn' not in col]

# Prepare data
df_model = customer_churn[feature_cols + ['Churn_90', 'CustomerKey']].copy()
df_model = df_model.dropna()

X = df_model[feature_cols]
y = df_model['Churn_90']

print(f"✓ Loaded {len(df_model):,} customers")
print(f"✓ Features: {len(feature_cols)}")
print(f"✓ Churn rate: {y.mean()*100:.2f}%")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, models_path / 'feature_scaler.pkl')

print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================================
# HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================================

print("\n" + "=" * 80)
print("HANDLING CLASS IMBALANCE")
print("=" * 80)

print(f"\nOriginal class distribution:")
print(f"  Class 0 (Active): {(~y_train.astype(bool)).sum():,} ({(1-y_train.mean())*100:.2f}%)")
print(f"  Class 1 (Churned): {y_train.sum():,} ({y_train.mean()*100:.2f}%)")

# Apply SMOTE
print(f"\nApplying SMOTE (Synthetic Minority Over-sampling)...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

print(f"\nBalanced class distribution:")
print(f"  Class 0 (Active): {(~y_train_balanced.astype(bool)).sum():,} ({(1-y_train_balanced.mean())*100:.2f}%)")
print(f"  Class 1 (Churned): {y_train_balanced.sum():,} ({y_train_balanced.mean()*100:.2f}%)")
print(f"✓ SMOTE completed: {len(X_train_balanced):,} training samples")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 80)

print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',  # Additional balancing
    C=0.1
)

lr_model.fit(X_train_balanced, y_train_balanced)
print("✓ Logistic Regression trained")

# Predictions
lr_pred = lr_model.predict(X_test_scaled)
lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_precision = precision_score(y_test, lr_pred)
lr_recall = recall_score(y_test, lr_pred)
lr_f1 = f1_score(y_test, lr_pred)
lr_roc_auc = roc_auc_score(y_test, lr_proba)

print(f"\nLogistic Regression Performance:")
print(f"  Accuracy:  {lr_accuracy:.4f}")
print(f"  Precision: {lr_precision:.4f}")
print(f"  Recall:    {lr_recall:.4f}")
print(f"  F1-Score:  {lr_f1:.4f}")
print(f"  ROC-AUC:   {lr_roc_auc:.4f}")

# Save model
joblib.dump(lr_model, models_path / 'logistic_regression_model.pkl')
print("✓ Model saved")

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 2: RANDOM FOREST")
print("=" * 80)

print("\nTraining Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_balanced, y_train_balanced)
print("✓ Random Forest trained")

# Predictions
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)
rf_roc_auc = roc_auc_score(y_test, rf_proba)

print(f"\nRandom Forest Performance:")
print(f"  Accuracy:  {rf_accuracy:.4f}")
print(f"  Precision: {rf_precision:.4f}")
print(f"  Recall:    {rf_recall:.4f}")
print(f"  F1-Score:  {rf_f1:.4f}")
print(f"  ROC-AUC:   {rf_roc_auc:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nTop 10 Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Save model and importance
joblib.dump(rf_model, models_path / 'random_forest_model.pkl')
feature_importance.to_csv(predictions_path / 'rf_feature_importance.csv', index=False)
print("✓ Model and feature importance saved")

# ============================================================================
# MODEL 3: XGBOOST
# ============================================================================

print("\n" + "=" * 80)
print("MODEL 3: XGBOOST")
print("=" * 80)

print("\nTraining XGBoost...")
# Calculate scale_pos_weight for imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train_balanced, y_train_balanced)
print("✓ XGBoost trained")

# Predictions
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)
xgb_roc_auc = roc_auc_score(y_test, xgb_proba)

print(f"\nXGBoost Performance:")
print(f"  Accuracy:  {xgb_accuracy:.4f}")
print(f"  Precision: {xgb_precision:.4f}")
print(f"  Recall:    {xgb_recall:.4f}")
print(f"  F1-Score:  {xgb_f1:.4f}")
print(f"  ROC-AUC:   {xgb_roc_auc:.4f}")

# Save model
joblib.dump(xgb_model, models_path / 'xgboost_model.pkl')
print("✓ Model saved")

# ============================================================================
# MODEL COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Accuracy': [lr_accuracy, rf_accuracy, xgb_accuracy],
    'Precision': [lr_precision, rf_precision, xgb_precision],
    'Recall': [lr_recall, rf_recall, xgb_recall],
    'F1-Score': [lr_f1, rf_f1, xgb_f1],
    'ROC-AUC': [lr_roc_auc, rf_roc_auc, xgb_roc_auc]
})

print("\nModel Performance Summary:")
print(results.to_string(index=False))

# Determine best model (by F1-score for imbalanced data)
best_idx = results['F1-Score'].idxmax()
best_model_name = results.loc[best_idx, 'Model']
print(f"\n🏆 Best Model: {best_model_name} (F1-Score: {results.loc[best_idx, 'F1-Score']:.4f})")

# Save comparison
results.to_csv(predictions_path / 'churn_model_comparison.csv', index=False)

# ============================================================================
# CONFUSION MATRICES
# ============================================================================

print("\n" + "=" * 80)
print("CONFUSION MATRICES")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

models_info = [
    ('Logistic Regression', lr_pred, lr_f1),
    ('Random Forest', rf_pred, rf_f1),
    ('XGBoost', xgb_pred, xgb_f1)
]

for idx, (name, preds, f1) in enumerate(models_info):
    cm = confusion_matrix(y_test, preds)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['Active', 'Churned'],
                yticklabels=['Active', 'Churned'])
    axes[idx].set_title(f'{name}\n(F1: {f1:.4f})', fontweight='bold')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig(outputs_path / 'churn_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Confusion matrices saved")

# ============================================================================
# ROC CURVES
# ============================================================================

print("\n" + "=" * 80)
print("ROC CURVES")
print("=" * 80)

plt.figure(figsize=(10, 8))

models_proba = [
    ('Logistic Regression', lr_proba, lr_roc_auc),
    ('Random Forest', rf_proba, rf_roc_auc),
    ('XGBoost', xgb_proba, xgb_roc_auc)
]

for name, proba, roc_auc in models_proba:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {roc_auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Churn Prediction Models', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outputs_path / 'churn_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ ROC curves saved")

# ============================================================================
# PRECISION-RECALL CURVES
# ============================================================================

print("\n" + "=" * 80)
print("PRECISION-RECALL CURVES")
print("=" * 80)

plt.figure(figsize=(10, 8))

for name, proba, roc_auc in models_proba:
    precision, recall, _ = precision_recall_curve(y_test, proba)
    avg_precision = average_precision_score(y_test, proba)
    plt.plot(recall, precision, linewidth=2, label=f'{name} (AP = {avg_precision:.4f})')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Churn Prediction Models', fontsize=14, fontweight='bold')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(outputs_path / 'churn_precision_recall_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Precision-Recall curves saved")

# ============================================================================
# FEATURE IMPORTANCE (Random Forest)
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE VISUALIZATION")
print("=" * 80)

top_features = feature_importance.head(20)

plt.figure(figsize=(10, 10))
plt.barh(range(len(top_features)), top_features['Importance'])
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 20 Features - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(outputs_path / 'churn_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature importance plot saved")

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING PREDICTIONS")
print("=" * 80)

# Create predictions dataframe
predictions_df = pd.DataFrame({
    'CustomerKey': df_model.loc[X_test.index, 'CustomerKey'],
    'Actual_Churn': y_test,
    'LR_Prediction': lr_pred,
    'LR_Probability': lr_proba,
    'RF_Prediction': rf_pred,
    'RF_Probability': rf_proba,
    'XGB_Prediction': xgb_pred,
    'XGB_Probability': xgb_proba
})

predictions_df.to_csv(predictions_path / 'churn_predictions.csv', index=False)
print(f"✓ Predictions saved for {len(predictions_df):,} customers")

# ============================================================================
# CLASSIFICATION REPORTS
# ============================================================================

print("\n" + "=" * 80)
print("DETAILED CLASSIFICATION REPORTS")
print("=" * 80)

for name, preds in [('Logistic Regression', lr_pred), ('Random Forest', rf_pred), ('XGBoost', xgb_pred)]:
    print(f"\n{name}:")
    print(classification_report(y_test, preds, target_names=['Active', 'Churned']))

print("\n" + "=" * 80)
print("PHASE 3: CHURN PREDICTION - COMPLETE ✓")
print("=" * 80)

print("\n✓ 3 models trained (Logistic Regression, Random Forest, XGBoost)")
print("✓ Class imbalance handled with SMOTE")
print("✓ Confusion matrices generated")
print("✓ ROC and Precision-Recall curves created")
print("✓ All models and predictions saved")

print(f"\n🏆 Best model: {best_model_name}")
print(f"\nNext: Phase 4 - Return Risk Analysis (or deployment)")
