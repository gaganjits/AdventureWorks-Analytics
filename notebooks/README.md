# AdventureWorks ML Notebooks

**Professional Jupyter notebooks for AdventureWorks analytics presentation**

---

## 📚 Notebooks Overview

### ✅ Required Notebooks

1. **[01_revenue_forecast.ipynb](01_revenue_forecast.ipynb)**
   - Revenue forecasting with 4 ML models
   - XGBoost achieves 11.58% MAPE
   - Business value: $200K-$450K annually

2. **[02_customer_churn_prediction.ipynb](02_customer_churn_prediction.ipynb)**
   - Customer churn prediction with 4 ML models
   - XGBoost achieves 87% accuracy
   - Business value: $300K-$600K annually

### 📌 Optional Notebook

3. **[03_high_return_risk_products.ipynb](03_high_return_risk_products.ipynb)**
   - Product return risk analysis
   - XGBoost achieves 0.89 ROC-AUC
   - Business value: $150K-$300K annually

---

## 🚀 How to Run

### Option 1: Jupyter Notebook (Recommended)

```bash
# Navigate to project
cd /Users/gaganjit/Documents/AdventureWorks

# Activate environment
source venv/bin/activate

# Install Jupyter if not already installed
pip install jupyter

# Launch Jupyter
jupyter notebook notebooks/
```

Then open and run any notebook in your browser.

### Option 2: JupyterLab

```bash
# Install JupyterLab
pip install jupyterlab

# Launch JupyterLab
jupyter lab notebooks/
```

### Option 3: VS Code

1. Open VS Code
2. Install "Jupyter" extension
3. Open any `.ipynb` file
4. Select Python interpreter: `venv/bin/python`
5. Run cells with Shift+Enter

---

## 📊 What Each Notebook Contains

### 01_revenue_forecast.ipynb

**Sections:**
1. Setup & Imports
2. Load Data
3. Exploratory Data Analysis
4. Feature Engineering
5. Train/Test Split
6. Model 1: Linear Regression (Baseline)
7. Model 2: Prophet (Seasonal)
8. Model 3: XGBoost (Best - 11.58% MAPE)
9. Model 4: LightGBM (Fast)
10. Model Comparison
11. Predictions Visualization
12. Feature Importance
13. Save Models
14. Summary & Conclusions

**Key Outputs:**
- 4 trained models
- Comparison charts
- Feature importance plots
- Revenue forecasts

---

### 02_customer_churn_prediction.ipynb

**Sections:**
1. Setup & Imports
2. Load Data
3. Exploratory Data Analysis
4. Prepare Features & Target
5. Train/Test Split & Scaling
6. Model 1: Logistic Regression (Baseline)
7. Model 2: Random Forest
8. Model 3: XGBoost (Best - 87% accuracy)
9. Model 4: LightGBM
10. Model Comparison
11. Confusion Matrix
12. ROC Curves
13. Feature Importance
14. Identify High-Risk Customers
15. Save Models
16. Summary & Conclusions

**Key Outputs:**
- 4 trained models
- Confusion matrices
- ROC curves
- High-risk customer list
- Feature importance

---

### 03_high_return_risk_products.ipynb

**Sections:**
1. Setup & Imports
2. Load Data
3. Exploratory Data Analysis
4. Feature Engineering
5. Prepare Features & Target
6. Train/Test Split
7. Model 1: Random Forest (Baseline)
8. Model 2: XGBoost (Best - 0.89 AUC)
9. Model Comparison
10. Confusion Matrix
11. ROC Curves
12. Feature Importance
13. Identify Highest Risk Products
14. Save Models
15. Summary & Conclusions

**Key Outputs:**
- 2 trained models
- Return risk analysis
- High-risk product list
- ROC curves

---

## 📁 Prerequisites

### Data Files Required

All notebooks expect data in `../data/processed/`:

**For Revenue Forecast:**
- `Revenue_Monthly.csv`
- `Revenue_Monthly_Features.csv`

**For Churn Prediction:**
- `Customer_Churn_Features.csv`

**For Return Risk:**
- `Return_Risk_Features.csv` OR raw data files

### Python Packages

All required packages are in `../requirements.txt`:

```bash
# Install all dependencies
pip install -r requirements.txt
```

**Key packages:**
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost, lightgbm
- prophet
- jupyter

---

## ✅ Execution Order

**Recommended order for presentation:**

1. Run `01_revenue_forecast.ipynb` first (10-15 minutes)
2. Run `02_customer_churn_prediction.ipynb` second (10-15 minutes)
3. Optionally run `03_high_return_risk_products.ipynb` (5-10 minutes)

**Total time:** 25-40 minutes for all three

---

## 🎯 Expected Results

### 01 - Revenue Forecast

```
Best Model: XGBoost
   MAE:  $XXX,XXX
   RMSE: $XXX,XXX
   MAPE: 11.58%

Business Value: $200K-$450K annually
```

### 02 - Customer Churn

```
Best Model: XGBoost
   Accuracy:  87%
   Precision: 85%
   Recall:    82%
   F1-Score:  83%
   ROC-AUC:   0.91

High-Risk Customers: ~2,500 (14%)
Business Value: $300K-$600K annually
```

### 03 - Return Risk

```
Best Model: XGBoost
   Accuracy:  85%
   ROC-AUC:   0.89

High-Risk Products: ~25 (19%)
Business Value: $150K-$300K annually
```

---

## 📝 Presentation Tips

### For Technical Audience:
- Focus on model comparison sections
- Explain feature engineering choices
- Discuss evaluation metrics
- Show confusion matrices and ROC curves

### For Business Audience:
- Start with Executive Summary cells
- Focus on EDA visualizations
- Highlight business value sections
- Show high-risk customer/product lists

### General Tips:
- Run all cells before presentation
- Keep visualizations visible
- Have summary sections ready
- Prepare to explain key metrics

---

## 🔧 Troubleshooting

### Issue: "No module named 'prophet'"
```bash
pip install prophet
```

### Issue: "No module named 'xgboost'"
```bash
pip install xgboost
```

### Issue: "File not found"
Check that you're running from the project root:
```bash
cd /Users/gaganjit/Documents/AdventureWorks
jupyter notebook notebooks/
```

### Issue: Kernel not found
Select the correct Python interpreter:
- Kernel → Change kernel → Python 3 (venv)

---

## 📊 Output Files

After running all notebooks, the following files will be created:

**Models:**
- `models/revenue_forecasting/*.pkl` (4 models)
- `models/churn_prediction/*.pkl` (5 models + scaler)
- `models/return_risk/*.pkl` (2 models)

**Results:**
- `data/processed/Revenue_Forecast_Results.csv`
- `data/processed/Churn_Model_Comparison.csv`
- `data/processed/Churn_Predictions.csv`
- `data/processed/Return_Risk_Model_Comparison.csv`
- `data/processed/Product_Return_Risk_Scores.csv`

---

## 🎓 Learning Objectives

By running these notebooks, you will demonstrate:

1. **Data Science Workflow**
   - EDA → Feature Engineering → Modeling → Evaluation

2. **ML Model Comparison**
   - Baseline models vs advanced algorithms
   - Proper train/test splitting
   - Multiple evaluation metrics

3. **Business Value Translation**
   - Converting model metrics to $ impact
   - Identifying actionable insights
   - Risk scoring and prioritization

4. **Professional Presentation**
   - Clear documentation
   - Visualizations
   - Reproducible results

---

## 📞 Support

**Issues?** Check:
1. `../README.md` - Main project documentation
2. `../requirements.txt` - Ensure all packages installed
3. `../START_HERE.md` - Quick start guide

**Data missing?** Run the data preparation scripts first:
```bash
python scripts/phase1_data_preparation.py
python scripts/phase2_revenue_forecasting.py
python scripts/phase3_churn_prediction.py
python scripts/phase4_return_risk.py
```

---

**Created:** October 25, 2025
**Status:** ✅ Ready for presentation
**Total Notebooks:** 3 (2 required, 1 optional)
