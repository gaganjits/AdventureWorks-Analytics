"""
Save feature names from trained models
This helps the API know which features to expect
"""

import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data" / "processed"

# Save churn features
print("Saving churn prediction feature names...")
churn_data_path = DATA_DIR / "Customer_Churn_Features.csv"
if churn_data_path.exists():
    df = pd.read_csv(churn_data_path)
    feature_cols = [col for col in df.columns if col not in ['CustomerKey', 'Churned', 'Churn']]

    feature_names_path = MODELS_DIR / "churn_prediction" / "feature_names.pkl"
    joblib.dump(feature_cols, feature_names_path)
    print(f"✓ Saved {len(feature_cols)} churn features to {feature_names_path}")
else:
    print(f"✗ Churn data not found: {churn_data_path}")

# Save return risk features
print("\nSaving return risk feature names...")
return_data_path = DATA_DIR / "Product_Return_Risk_Features.csv"
if return_data_path.exists():
    df = pd.read_csv(return_data_path)
    feature_cols = [col for col in df.columns if col not in ['ProductKey', 'HighReturnRisk', 'ProductName']]

    feature_names_path = MODELS_DIR / "return_risk" / "feature_names.pkl"
    joblib.dump(feature_cols, feature_names_path)
    print(f"✓ Saved {len(feature_cols)} return risk features to {feature_names_path}")
else:
    print(f"✗ Return risk data not found: {return_data_path}")

print("\n✅ Feature names saved successfully!")
