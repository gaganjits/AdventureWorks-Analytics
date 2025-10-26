# AdventureWorks - Where to View Your Results

**Quick Start:** This guide shows you exactly where to find and view all outputs from your analytics platform.

---

## 📊 1. Interactive Dashboard (BEST WAY TO VIEW RESULTS)

### Streamlit Dashboard - 5 Interactive Pages

**How to Launch:**
```bash
cd /Users/gaganjit/Documents/AdventureWorks
source venv/bin/activate
streamlit run dashboards/adventureworks_dashboard.py
```

**What You'll See:**
- 🏠 **Overview Page** - Key metrics, revenue trends, customer summary
- 💰 **Revenue Forecasting** - Interactive charts, model comparisons, predictions
- 👥 **Customer Insights** - Churn analysis, RFM segmentation, customer health
- 📦 **Product Analysis** - Return risk, top products, product performance
- 🎯 **Recommendations** - Personalized suggestions, ML model results

**Access:** Opens automatically in your browser at `http://localhost:8501`

**Status:** ✅ Already running (Process 77544)

---

## 📈 2. Visualizations & Plots

### Location: `outputs/plots/`

**30+ Generated Visualizations:**

```bash
# View all plots
open outputs/plots/

# Key visualizations:
open outputs/plots/revenue_forecast_comparison.png
open outputs/plots/churn_feature_importance.png
open outputs/plots/customer_segmentation_clusters.png
open outputs/plots/product_recommendations_heatmap.png
open outputs/plots/lstm_forecasting_results.png  # (if LSTM completes)
```

**What's Inside:**
- Revenue forecasting charts (4 models compared)
- Churn prediction confusion matrices
- Feature importance plots
- Customer segmentation visualizations
- Product recommendation heatmaps
- RFM analysis charts
- Return risk distributions

**File Count:** 30+ PNG files

---

## 📄 3. Comprehensive Reports

### Location: `outputs/reports/`

**9 Detailed Markdown Reports:**

```bash
# View all reports
open outputs/reports/

# Or read specific reports:
cat outputs/reports/Phase9_Deep_Learning_NLP_Report.md
cat outputs/reports/Phase8_Advanced_Analytics_Report.md
cat outputs/reports/Phase7_API_Integration_Report.md
```

**Available Reports:**
1. `Phase1_Data_Preparation_Report.md` - Data pipeline overview
2. `Phase2_Revenue_Forecasting_Report.md` - 4 model comparison (11.58% MAPE)
3. `Phase3_Churn_Prediction_Report.md` - 5 models (87% accuracy)
4. `Phase4_Return_Risk_Report.md` - Risk analysis results
5. `Phase5_Optimization_Report.md` - Hyperparameter tuning results
6. `Phase6_BI_Dashboards_Report.md` - Dashboard documentation
7. `Phase7_API_Integration_Report.md` - API endpoints guide
8. `Phase8_Advanced_Analytics_Report.md` - Segmentation + recommendations
9. `Phase9_Deep_Learning_NLP_Report.md` - NLP interface + LSTM status

**Total Pages:** 50+ pages of documentation

---

## 💾 4. Data Outputs & Predictions

### Location: `data/processed/`

**22 Processed Datasets:**

```bash
# Navigate to data
cd data/processed/

# View key outputs:
open Customer_Segmentation_Results.csv         # 17,416 customers with segments
open Product_Recommendations.csv               # 5,000 personalized suggestions
open NLP_Query_Examples.csv                    # 6 validated NLP queries
open Churn_Predictions.csv                     # Customer churn probabilities
open Revenue_Forecast_Results.csv              # Revenue predictions
```

**Key Files:**

| File | Description | Records |
|------|-------------|---------|
| `Customer_Segmentation_Results.csv` | Customer segments (VIP At-Risk, New Engaged) | 17,416 |
| `Product_Recommendations.csv` | Personalized product suggestions | 5,000 |
| `Frequent_Product_Pairs.csv` | Market basket analysis | 32 pairs |
| `Customer_Segment_Profiles.csv` | Segment characteristics | 2 segments |
| `NLP_Query_Examples.csv` | NLP test queries | 6 queries |
| `LSTM_Model_Comparison.csv` | Deep learning results | (pending) |

**How to View:**
```bash
# Quick preview in terminal
head -20 data/processed/Customer_Segmentation_Results.csv

# Or open in Excel/Numbers
open data/processed/Product_Recommendations.csv
```

---

## 🚀 5. REST API (Live Predictions)

### API Server - 8 Endpoints

**How to Access:**

**Status:** ✅ Already running on port 8000 (Process 78044)

**View API Documentation:**
```bash
# Open in browser:
open http://localhost:8000/docs
```

**Available Endpoints:**

1. **POST** `/predict/revenue` - Get revenue forecast
2. **POST** `/predict/churn` - Predict customer churn
3. **POST** `/predict/return-risk` - Product return risk
4. **POST** `/batch/churn` - Batch churn predictions
5. **POST** `/batch/revenue` - Batch revenue forecasts
6. **GET** `/customer/{id}/segment` - Customer segment lookup
7. **GET** `/customer/{id}/recommendations` - Personalized recommendations
8. **GET** `/health` - System health check

**Test the API:**
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test customer segment
curl http://localhost:8000/customer/11000/segment

# Test recommendations
curl http://localhost:8000/customer/11000/recommendations
```

**API Documentation:** See `outputs/reports/Phase7_API_Integration_Report.md`

---

## 🤖 6. NLP Query Interface (Natural Language)

### Interactive Query System

**How to Run:**
```bash
cd /Users/gaganjit/Documents/AdventureWorks
source venv/bin/activate
python scripts/phase9_nlp_query_interface.py
```

**What It Does:**
- Processes natural language queries
- Returns data-driven insights
- Accesses all 19 ML models

**Example Queries:**
- "Show me customers at risk of churning"
- "Recommend products for customer 11000"
- "Which customer segments do we have?"
- "What was revenue last month?"

**Results Location:** Printed to console + saved to `data/processed/NLP_Query_Examples.csv`

---

## 📱 7. Quick Demo Script

### One Command to See Everything

**Run This:**
```bash
cd /Users/gaganjit/Documents/AdventureWorks
source venv/bin/activate
python demo_system.py
```

**What You'll See:**
- ✅ Data pipeline status (17,416 customers, 5,000 recommendations)
- ✅ All ML models loaded (Revenue, Churn, Segmentation)
- ✅ NLP interface test results (100% success rate)
- ✅ System integration verification
- ✅ Business value summary ($1.08M - $2.27M/year)

**Output:** Complete system demonstration in your terminal

---

## 📊 8. Automated Reports (Weekly HTML)

### Location: `outputs/automated_reports/`

**HTML Reports Generated:**
```bash
# View latest automated report
open outputs/automated_reports/weekly_report_latest.html

# Or navigate to folder
open outputs/automated_reports/
```

**What's Inside:**
- Executive summary
- Revenue trends
- Customer churn alerts
- Product performance
- Recommendations for action

**Schedule:** Configured for weekly generation (see `setup_automated_reports.sh`)

---

## 🗂️ 9. Model Files (For Developers)

### Location: `models/`

**19 Trained ML Models:**

```bash
# View all models
ls -R models/

# Directories:
models/
├── revenue_forecasting/     # 4 models (XGBoost best: 11.58% MAPE)
├── churn_prediction/        # 5 models (XGBoost best: 87% accuracy)
├── return_risk/             # 1 model (AUC: 0.89)
├── customer_segmentation/   # 2 files (K-means + scaler)
├── recommendations/         # 2 files (user-item matrix + similarities)
├── nlp/                     # 1 config file (7 intents, 5 entities)
└── deep_learning/           # LSTM models (when trained)
```

**Load Models Programmatically:**
```python
import joblib

# Load revenue model
revenue_model = joblib.load('models/revenue_forecasting/xgboost_model.pkl')

# Load churn model
churn_model = joblib.load('models/churn_prediction/xgboost_model.pkl')

# Load segmentation
segment_model = joblib.load('models/customer_segmentation/kmeans_model.pkl')
```

---

## 🔍 10. System Status & Logs

### Verify Everything is Working

**Quick Health Check:**
```bash
cd /Users/gaganjit/Documents/AdventureWorks
source venv/bin/activate
python demo_system.py
```

**Check Running Services:**
```bash
# Check API server
curl http://localhost:8000/health

# Check Streamlit dashboard
ps aux | grep streamlit

# Check all Python processes
ps aux | grep Python | grep AdventureWorks
```

**View System Coordination:**
```bash
cat SYSTEM_COORDINATION_STATUS.md
```

---

## 📋 Quick Reference Table

| What You Want | Where to Look | How to Access |
|---------------|---------------|---------------|
| **Interactive visualizations** | Streamlit dashboard | `streamlit run dashboards/adventureworks_dashboard.py` |
| **Static charts/plots** | `outputs/plots/` | `open outputs/plots/` |
| **Detailed reports** | `outputs/reports/` | `cat outputs/reports/Phase9_*.md` |
| **Customer segments** | `data/processed/Customer_Segmentation_Results.csv` | `open data/processed/Customer_Segmentation_Results.csv` |
| **Product recommendations** | `data/processed/Product_Recommendations.csv` | `open data/processed/Product_Recommendations.csv` |
| **NLP test results** | `data/processed/NLP_Query_Examples.csv` | `cat data/processed/NLP_Query_Examples.csv` |
| **Live predictions** | REST API on port 8000 | `open http://localhost:8000/docs` |
| **Natural language queries** | NLP interface | `python scripts/phase9_nlp_query_interface.py` |
| **Quick demo** | Demo script | `python demo_system.py` |
| **Weekly reports** | `outputs/automated_reports/` | `open outputs/automated_reports/weekly_report_latest.html` |
| **System status** | Terminal | `python demo_system.py` |

---

## 🎯 Recommended Viewing Order

### For Business Users:
1. **Start with Dashboard** → `streamlit run dashboards/adventureworks_dashboard.py`
2. **Try NLP queries** → `python scripts/phase9_nlp_query_interface.py`
3. **Read Phase 9 report** → `cat outputs/reports/Phase9_Deep_Learning_NLP_Report.md`
4. **View automated reports** → `open outputs/automated_reports/`

### For Technical Users:
1. **Run demo script** → `python demo_system.py`
2. **Explore API** → `open http://localhost:8000/docs`
3. **View data outputs** → `open data/processed/`
4. **Read technical reports** → `cat outputs/reports/*.md`
5. **Check model files** → `ls -R models/`

### For Executives:
1. **Dashboard overview** → `streamlit run dashboards/adventureworks_dashboard.py`
2. **Phase 9 report** → Executive summary section
3. **Business value** → See `SYSTEM_COORDINATION_STATUS.md`
4. **Weekly reports** → `outputs/automated_reports/`

---

## 🚀 Quick Start Commands

**Copy & paste these to get started:**

```bash
# 1. Navigate to project
cd /Users/gaganjit/Documents/AdventureWorks

# 2. Activate environment
source venv/bin/activate

# 3. Choose what to view:

# Option A: Interactive dashboard (RECOMMENDED)
streamlit run dashboards/adventureworks_dashboard.py

# Option B: Quick demo
python demo_system.py

# Option C: NLP interface
python scripts/phase9_nlp_query_interface.py

# Option D: View all plots
open outputs/plots/

# Option E: Read reports
cat outputs/reports/Phase9_Deep_Learning_NLP_Report.md

# Option F: Check API
open http://localhost:8000/docs

# Option G: View data
open data/processed/Customer_Segmentation_Results.csv
```

---

## 💡 Tips

1. **Best Overall Experience:** Use the Streamlit dashboard - it's interactive and shows everything
2. **Quick Insights:** Run `python demo_system.py` for a 30-second overview
3. **Detailed Analysis:** Read the reports in `outputs/reports/`
4. **Raw Data:** Excel users can open any CSV in `data/processed/`
5. **Live Testing:** Use the API endpoints for real-time predictions
6. **Natural Language:** Try the NLP interface for intuitive queries

---

**Need Help?**
- See `README.md` for setup instructions
- See `SYSTEM_COORDINATION_STATUS.md` for integration details
- See individual phase reports for deep dives
- Run `python demo_system.py` to verify everything is working

---

**Last Updated:** October 25, 2025
**Status:** ✅ All systems operational and ready to view!
