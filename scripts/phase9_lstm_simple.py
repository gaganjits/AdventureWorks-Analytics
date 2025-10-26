"""
Phase 9: Simplified LSTM Time Series Forecasting
Optimized version to avoid TensorFlow initialization issues

This script implements a streamlined LSTM for revenue forecasting.

Author: AdventureWorks Data Science Team
Date: October 25, 2025
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow with suppressed logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "deep_learning"
OUTPUTS_DIR = BASE_DIR / "outputs" / "plots"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PHASE 9: SIMPLIFIED LSTM TIME SERIES FORECASTING")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[1/5] Loading revenue data...")
monthly_revenue_path = DATA_DIR / "Revenue_Monthly.csv"
df = pd.read_csv(monthly_revenue_path)
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df = df.sort_values('OrderDate')

print(f"✓ Loaded {len(df)} months of revenue data")
print(f"  Date range: {df['OrderDate'].min()} to {df['OrderDate'].max()}")

# ============================================================================
# STEP 2: PREPARE SEQUENCES
# ============================================================================

print("\n[2/5] Preparing sequences...")

def create_sequences(data, lookback=6, forecast_horizon=1):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback:i + lookback + forecast_horizon])
    return np.array(X), np.array(y)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
revenue_scaled = scaler.fit_transform(df[['Revenue']].values)

# Parameters
LOOKBACK = 6
FORECAST_HORIZON = 1
TRAIN_SIZE = 0.8

# Create sequences
X, y = create_sequences(revenue_scaled, lookback=LOOKBACK, forecast_horizon=FORECAST_HORIZON)
print(f"✓ Created {len(X)} sequences")

# Split
split_idx = int(len(X) * TRAIN_SIZE)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape for LSTM
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ============================================================================
# STEP 3: BUILD AND TRAIN LSTM
# ============================================================================

print("\n[3/5] Building LSTM model...")

model = Sequential([
    LSTM(32, activation='relu', input_shape=(LOOKBACK, 1)),
    Dropout(0.2),
    Dense(FORECAST_HORIZON)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("✓ Model built (32 units, 1 layer)")

# Train with optimized parameters
print("\n[4/5] Training model...")
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=10,  # Reduced for faster training
    batch_size=8,  # Larger batch for speed
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=0  # Silent training
)

actual_epochs = len(history.history['loss'])
print(f"✓ Training complete ({actual_epochs} epochs)")

# ============================================================================
# STEP 4: EVALUATE
# ============================================================================

print("\n[5/5] Evaluating model...")

# Predict
y_pred_scaled = model.predict(X_test, verbose=0)

# Inverse transform
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_original = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))

# Calculate metrics
mae = mean_absolute_error(y_test_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
mape = mean_absolute_percentage_error(y_test_original, y_pred_original) * 100

print(f"\n  LSTM Performance:")
print(f"    MAE:  ${mae:,.2f}")
print(f"    RMSE: ${rmse:,.2f}")
print(f"    MAPE: {mape:.2f}%")

# Compare with Phase 2
print(f"\n  Comparison:")
print(f"    XGBoost (Phase 2): 11.58% MAPE")
print(f"    LSTM (Phase 9):    {mape:.2f}% MAPE")

if mape < 11.58:
    improvement = ((11.58 - mape) / 11.58) * 100
    print(f"    ✓ Improvement: {improvement:.1f}% better")
else:
    diff = mape - 11.58
    print(f"    Note: {diff:.2f}% higher than XGBoost (acceptable for neural network baseline)")

# ============================================================================
# STEP 5: SAVE RESULTS
# ============================================================================

print("\n[6/6] Saving results...")

# Save model comparison
results_df = pd.DataFrame([{
    'Model': 'Simple LSTM',
    'MAE': mae,
    'RMSE': rmse,
    'MAPE (%)': mape,
    'Epochs': actual_epochs,
    'Parameters': model.count_params()
}])
results_df.to_csv(DATA_DIR / "LSTM_Model_Comparison.csv", index=False)
print("  ✓ Saved model comparison")

# Save model
model.save(MODELS_DIR / "lstm_revenue_forecast.keras")
joblib.dump(scaler, MODELS_DIR / "lstm_scaler.pkl")
print("  ✓ Saved model and scaler")

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('LSTM Time Series Forecasting Results', fontsize=16, fontweight='bold')

# 1. Training history
ax = axes[0, 0]
ax.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax.set_title('Training History', fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (MSE)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Predictions vs Actual
ax = axes[0, 1]
test_dates = df['OrderDate'].iloc[-len(y_test_original):]
ax.plot(test_dates, y_test_original, 'o-', label='Actual', linewidth=2, markersize=8)
ax.plot(test_dates, y_pred_original, 's--', label='LSTM Predicted', linewidth=2, markersize=8)
ax.set_title('Predictions vs Actual', fontweight='bold')
ax.set_xlabel('Date')
ax.set_ylabel('Revenue ($)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

# 3. Prediction errors
ax = axes[1, 0]
errors = y_pred_original.flatten() - y_test_original.flatten()
ax.hist(errors, bins=10, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
ax.set_title('Prediction Error Distribution', fontweight='bold')
ax.set_xlabel('Prediction Error ($)')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# 4. Model performance metrics
ax = axes[1, 1]
ax.axis('off')
metrics_text = f"""
LSTM Model Performance

Metrics:
  • MAE:  ${mae:,.0f}
  • RMSE: ${rmse:,.0f}
  • MAPE: {mape:.2f}%

Training:
  • Epochs: {actual_epochs}
  • Parameters: {model.count_params():,}
  • Lookback: {LOOKBACK} months

Comparison:
  • XGBoost MAPE: 11.58%
  • LSTM MAPE: {mape:.2f}%
  • Difference: {abs(mape - 11.58):.2f}%
"""
ax.text(0.1, 0.5, metrics_text, fontsize=12, verticalalignment='center',
        fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "lstm_forecasting_results.png", dpi=300, bbox_inches='tight')
print("  ✓ Saved visualization")

print("\n" + "="*70)
print("LSTM FORECASTING COMPLETE!")
print("="*70)
print(f"\nResults:")
print(f"  • Model performance: {mape:.2f}% MAPE")
print(f"  • Trained in {actual_epochs} epochs")
print(f"  • Saved to: {MODELS_DIR}")
print(f"\nFiles created:")
print(f"  • {DATA_DIR}/LSTM_Model_Comparison.csv")
print(f"  • {MODELS_DIR}/lstm_revenue_forecast.keras")
print(f"  • {OUTPUTS_DIR}/lstm_forecasting_results.png")
