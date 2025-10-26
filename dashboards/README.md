# AdventureWorks Business Intelligence Dashboards

This directory contains interactive dashboards and automated reporting tools for the AdventureWorks analytics project.

## 📊 Available Dashboards

### 1. Main Streamlit Dashboard (`adventureworks_dashboard.py`)

A comprehensive multi-page dashboard providing:

- **🏠 Executive Summary:** High-level KPIs and business metrics
- **📈 Revenue Forecasting:** Interactive revenue predictions and accuracy metrics
- **👥 Customer Churn Analysis:** Churn risk scoring and RFM analysis
- **📦 Product Return Risk:** Return rate analysis and quality insights
- **🔍 Model Performance:** Detailed model metrics and deployment recommendations

## 🚀 Quick Start

### Running the Streamlit Dashboard

```bash
# Activate virtual environment
source ../venv/bin/activate

# Run dashboard
streamlit run adventureworks_dashboard.py
```

The dashboard will open in your default browser at `http://localhost:8501`

### Features

- **Interactive Visualizations:** Plotly charts with zoom, pan, and hover tooltips
- **Multi-Page Navigation:** Sidebar navigation between different analysis views
- **Real-Time Metrics:** Key performance indicators updated from latest data
- **Export Capabilities:** Download charts as PNG images
- **Responsive Design:** Works on desktop and tablet devices

## 📋 Dashboard Pages

### Page 1: Executive Summary

- Total revenue, customers, products metrics
- Revenue trend visualization (2015-2017)
- Model performance scorecard
- Category performance breakdown
- Estimated business impact ($480K-$1.28M annually)

### Page 2: Revenue Forecasting

- Historical revenue + forecast visualization
- Forecast accuracy table (actual vs predicted)
- Feature importance analysis
- Model details and parameters
- **Key Metric:** 11.58% MAPE (25% improvement vs baseline)

### Page 3: Customer Churn Analysis

- Churn distribution pie chart
- Churn risk by recency bins
- Top 20 high-risk customers table
- RFM (Recency, Frequency, Monetary) histograms
- Retention recommendations
- **Key Metric:** 100% accuracy, 65.9% churn rate identified

### Page 4: Product Return Risk

- Return rate by category bar chart
- High-risk product distribution
- Top 20 high-risk products table
- Subcategory analysis
- Quality improvement recommendations
- **Key Metric:** 100% accuracy, 31 high-risk products flagged

### Page 5: Model Performance

- All models overview table
- Phase-by-phase comparison cards
- Estimated business value breakdown
- Technical stack summary
- Deployment recommendations

## 📁 Dashboard Structure

```
dashboards/
├── adventureworks_dashboard.py   # Main Streamlit app
├── README.md                      # This file
└── automated_reports.py           # Scheduled report generator (upcoming)
```

## 🎨 Customization

### Changing Theme

Edit `adventureworks_dashboard.py` and modify the custom CSS in the `st.markdown()` section:

```python
st.markdown("""
<style>
    .main-header {
        color: #1f77b4;  # Change header color
    }
</style>
""", unsafe_allow_html=True)
```

### Adding New Pages

1. Add new page option to sidebar:
```python
page = st.sidebar.radio(
    "Navigate to:",
    ["Existing Pages...", "🆕 New Page"]
)
```

2. Add page logic:
```python
elif page == "🆕 New Page":
    st.markdown("# New Page Title")
    # Your code here
```

## 📊 Data Sources

The dashboard reads from:

- `data/processed/AdventureWorks_Sales_Enriched.csv` - Sales transactions
- `data/processed/Customer_Churn_Features.csv` - Customer churn data
- `data/processed/Product_Return_Risk_Features.csv` - Product return data
- `data/processed/Revenue_Monthly_Features.csv` - Revenue time series
- `models/phase5_xgboost_revenue_optimized.pkl` - Revenue forecasting model
- `models/phase5_ensemble_churn.pkl` - Churn prediction model (optional)
- `models/phase5_ensemble_return.pkl` - Return risk model (optional)

## 🔧 Troubleshooting

### Issue: Dashboard won't start

**Solution:**
```bash
# Check if streamlit is installed
pip list | grep streamlit

# Reinstall if needed
pip install streamlit plotly
```

### Issue: Data not loading

**Solution:**
- Verify all CSV files exist in `data/processed/`
- Run Phase 1-5 scripts to generate data if missing
- Check file paths are correct (absolute vs relative)

### Issue: Models not found

**Solution:**
```bash
# Check models directory
ls -l ../models/phase5*.pkl

# Run Phase 5 optimization if models missing
python ../scripts/phase5_optimization.py
python ../scripts/phase5_ensemble.py
```

### Issue: Port 8501 already in use

**Solution:**
```bash
# Run on different port
streamlit run adventureworks_dashboard.py --server.port 8502
```

## 📈 Performance Tips

1. **Large Datasets:** Dashboard loads ~70K+ rows. Consider:
   - Adding date filters to limit data range
   - Using `@st.cache_data` decorator for data loading
   - Sampling data for visualizations

2. **Slow Loading:** Optimize by:
   - Loading only required columns
   - Pre-aggregating data in scripts
   - Using Plotly's `scattergl` for large scatter plots

3. **Memory Usage:** Monitor with:
```python
import psutil
st.sidebar.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
```

## 🚀 Deployment Options

### Option 1: Streamlit Cloud (Easiest)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo and select `dashboards/adventureworks_dashboard.py`
4. Deploy!

### Option 2: Docker

```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "dashboards/adventureworks_dashboard.py"]
```

### Option 3: Local Server

```bash
# Run in background
nohup streamlit run adventureworks_dashboard.py &

# Access from network
streamlit run adventureworks_dashboard.py --server.address 0.0.0.0
```

## 📧 Automated Reports

For scheduled email reports, see `automated_reports.py` (upcoming feature).

## 🆘 Support

- **Documentation:** [Streamlit Docs](https://docs.streamlit.io)
- **Plotly Docs:** [Plotly Python](https://plotly.com/python/)
- **Issues:** Open an issue in the project repo

## 📝 License

Internal use only. AdventureWorks Analytics Team.

---

**Last Updated:** October 24, 2025
**Version:** 1.0
**Maintainer:** Data Science Team
