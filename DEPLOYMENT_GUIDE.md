# AdventureWorks Analytics - Production Deployment Guide

**Version:** 1.0
**Date:** October 24, 2025
**Status:** Production Ready ✅

---

## Table of Contents

1. [Quick Start (5 Minutes)](#quick-start-5-minutes)
2. [Local Deployment](#local-deployment)
3. [Automated Reports Setup](#automated-reports-setup)
4. [Streamlit Cloud Deployment (Recommended)](#streamlit-cloud-deployment-recommended)
5. [Alternative Deployments](#alternative-deployments)
6. [User Training](#user-training)
7. [Maintenance](#maintenance)

---

## Quick Start (5 Minutes)

### Prerequisites Checklist

- [x] Python 3.13 installed
- [x] Virtual environment activated
- [x] All packages installed (`pip install -r requirements.txt`)
- [x] Data processed (Phases 1-5 complete)
- [x] Models trained and saved in `models/` directory

### Test Locally

```bash
# 1. Navigate to project
cd /Users/gaganjit/Documents/AdventureWorks

# 2. Activate virtual environment
source venv/bin/activate

# 3. Run dashboard
streamlit run dashboards/adventureworks_dashboard.py

# 4. Open browser to: http://localhost:8501
```

**Expected Result:**
- Dashboard loads in < 2 seconds
- All 5 pages accessible via sidebar
- Charts render without errors
- Metrics display correct values

**If issues occur, see [Troubleshooting](#troubleshooting) section**

---

## Local Deployment

### Option 1: Foreground (Development)

```bash
# Run in terminal (blocks terminal)
streamlit run dashboards/adventureworks_dashboard.py
```

**Pros:** See logs in real-time, easy to stop (Ctrl+C)
**Cons:** Blocks terminal, stops when terminal closes

### Option 2: Background (Production)

```bash
# Run in background
nohup streamlit run dashboards/adventureworks_dashboard.py > streamlit.log 2>&1 &

# Check if running
ps aux | grep streamlit

# View logs
tail -f streamlit.log

# Stop
pkill -f streamlit
```

**Pros:** Runs independently, survives terminal close
**Cons:** Must check logs file for errors

### Option 3: Network Access

```bash
# Allow access from other devices on your network
streamlit run dashboards/adventureworks_dashboard.py --server.address 0.0.0.0

# Find your IP address
ifconfig | grep "inet " | grep -v 127.0.0.1

# Access from phone/tablet: http://YOUR_IP:8501
```

**Use Case:** Demo dashboard to team in meeting

---

## Automated Reports Setup

### Schedule Weekly Reports (Recommended)

**Every Monday at 9 AM:**

#### macOS/Linux (crontab)

```bash
# Edit crontab
crontab -e

# Add this line (replace paths):
0 9 * * 1 /Users/gaganjit/Documents/AdventureWorks/venv/bin/python /Users/gaganjit/Documents/AdventureWorks/dashboards/automated_reports.py >> /Users/gaganjit/Documents/AdventureWorks/reports.log 2>&1

# Save and exit
# Verify:
crontab -l
```

**Cron Schedule Examples:**
```
0 9 * * 1      # Every Monday 9 AM
0 8 1 * *      # First day of month 8 AM
0 9 * * 1-5    # Weekdays 9 AM
0 18 * * 5     # Every Friday 6 PM
```

#### Windows (Task Scheduler)

1. Open **Task Scheduler** (search in Start menu)
2. Click **Create Basic Task**
3. Name: "AdventureWorks Weekly Report"
4. Trigger: **Weekly**, Monday, 9:00 AM
5. Action: **Start a program**
   - Program: `C:\Users\YourName\Documents\AdventureWorks\venv\Scripts\python.exe`
   - Arguments: `dashboards\automated_reports.py`
   - Start in: `C:\Users\YourName\Documents\AdventureWorks`
6. Click **Finish**

#### Test Scheduled Script

```bash
# Run manually to test
python dashboards/automated_reports.py

# Check output
ls -lh outputs/automated_reports/

# Expected files:
# - Executive_Summary_2025-10-24.html
# - Summary_Metrics_2025-10-24.csv
```

### Email Distribution (Optional)

Create `dashboards/email_reports.py`:

```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from datetime import datetime

# Configuration
SMTP_SERVER = "smtp.gmail.com"  # or your company SMTP
SMTP_PORT = 587
SENDER_EMAIL = "analytics@adventureworks.com"
SENDER_PASSWORD = "your_app_password"  # Use app-specific password
RECIPIENTS = ["ceo@adventureworks.com", "cfo@adventureworks.com"]

# Email content
subject = f"AdventureWorks Weekly Analytics Report - {datetime.now().strftime('%Y-%m-%d')}"
body = """
Hello,

Please find attached this week's AdventureWorks Analytics Executive Summary.

Key Highlights:
- Revenue Forecast Accuracy: 11.58% MAPE
- Churn Prediction: 100% Accuracy
- High-Risk Products Identified: 31

View the full interactive dashboard at: http://your-dashboard-url.com

Best regards,
AdventureWorks Analytics Team
"""

# Find latest report
reports_path = Path("outputs/automated_reports")
latest_html = max(reports_path.glob("Executive_Summary_*.html"))

# Create message
msg = MIMEMultipart()
msg['From'] = SENDER_EMAIL
msg['To'] = ", ".join(RECIPIENTS)
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))

# Attach HTML report
with open(latest_html, 'rb') as f:
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename={latest_html.name}')
    msg.attach(part)

# Send email
with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
    server.starttls()
    server.login(SENDER_EMAIL, SENDER_PASSWORD)
    server.send_message(msg)

print(f"✓ Email sent to {len(RECIPIENTS)} recipients")
```

**Schedule this after report generation:**
```bash
# crontab: Run report, then email it
0 9 * * 1 /path/to/python automated_reports.py && /path/to/python email_reports.py
```

---

## Streamlit Cloud Deployment (Recommended)

### Why Streamlit Cloud?

✅ **Free** for public apps (or $20/month for private)
✅ **Zero DevOps** - no server management
✅ **Auto-deploy** on git push
✅ **HTTPS** included
✅ **Fast** - global CDN
✅ **5-minute setup**

### Step-by-Step Instructions

#### Step 1: Prepare Repository

```bash
# 1. Initialize git (if not already done)
cd /Users/gaganjit/Documents/AdventureWorks
git init

# 2. Create .gitignore
cat > .gitignore << EOF
venv/
__pycache__/
*.pyc
.DS_Store
*.log
.streamlit/secrets.toml
EOF

# 3. Add all files
git add .

# 4. Commit
git commit -m "Initial commit: AdventureWorks Analytics Dashboard"
```

#### Step 2: Push to GitHub

```bash
# 1. Create repo on GitHub.com
# Go to: https://github.com/new
# Name: adventureworks-analytics
# Visibility: Private (recommended for business data)

# 2. Add remote
git remote add origin https://github.com/YOUR_USERNAME/adventureworks-analytics.git

# 3. Push
git branch -M main
git push -u origin main
```

#### Step 3: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. **Repository:** Select `your-username/adventureworks-analytics`
4. **Branch:** `main`
5. **Main file path:** `dashboards/adventureworks_dashboard.py`
6. **App URL:** Choose custom name (e.g., `adventureworks-analytics`)
7. Click **"Deploy!"**

**Wait 2-3 minutes... Done!** ✅

Your dashboard is now live at:
`https://adventureworks-analytics.streamlit.app`

#### Step 4: Configure Privacy (If Needed)

**For Private Apps ($20/month):**
1. In Streamlit Cloud dashboard, click app settings
2. Go to **"Sharing"** tab
3. Set to **"Private"**
4. Add authorized email addresses

**Free Alternative (Password Protection):**

Add to top of `dashboards/adventureworks_dashboard.py`:

```python
import streamlit as st
import hashlib

def check_password():
    """Returns `True` if the user had the correct password."""
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hashlib.sha256(st.session_state["password"].encode()).hexdigest() == "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8":  # "password"
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input("Password", type="password", on_change=password_entered, key="password")
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if not check_password():
    st.stop()

# Rest of dashboard code...
```

**Generate password hash:**
```python
import hashlib
password = "your_secure_password"
hashed = hashlib.sha256(password.encode()).hexdigest()
print(hashed)  # Use this in code above
```

#### Step 5: Update on Code Changes

```bash
# Make changes to dashboard
nano dashboards/adventureworks_dashboard.py

# Commit and push
git add .
git commit -m "Update dashboard with new features"
git push

# Streamlit Cloud auto-deploys in ~2 minutes!
```

---

## Alternative Deployments

### Docker Deployment

#### Create Dockerfile

```dockerfile
FROM python:3.13-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run dashboard
CMD ["streamlit", "run", "dashboards/adventureworks_dashboard.py", "--server.address", "0.0.0.0"]
```

#### Build and Run

```bash
# Build image
docker build -t adventureworks-dashboard .

# Run container
docker run -d -p 8501:8501 --name aw-dashboard adventureworks-dashboard

# Check logs
docker logs aw-dashboard

# Stop
docker stop aw-dashboard
```

#### Deploy to Cloud

**AWS ECS / Google Cloud Run / Azure Container Instances:**

```bash
# Tag for registry
docker tag adventureworks-dashboard:latest YOUR_REGISTRY/adventureworks-dashboard:latest

# Push to registry
docker push YOUR_REGISTRY/adventureworks-dashboard:latest

# Deploy (platform-specific commands)
```

### Cloud VM Deployment

**AWS EC2 / GCP Compute Engine / Azure VM:**

```bash
# SSH into VM
ssh user@your-vm-ip

# Install dependencies
sudo apt-get update
sudo apt-get install python3-pip git

# Clone repo
git clone https://github.com/your-username/adventureworks-analytics.git
cd adventureworks-analytics

# Install packages
pip3 install -r requirements.txt

# Create systemd service
sudo nano /etc/systemd/system/streamlit-dashboard.service
```

**systemd service file:**
```ini
[Unit]
Description=AdventureWorks Streamlit Dashboard
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/adventureworks-analytics
ExecStart=/usr/local/bin/streamlit run dashboards/adventureworks_dashboard.py --server.address 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable streamlit-dashboard
sudo systemctl start streamlit-dashboard

# Check status
sudo systemctl status streamlit-dashboard
```

---

## User Training

### For Executives (Non-Technical)

#### Quick Start Guide

**Accessing the Dashboard:**
1. Open web browser (Chrome, Safari, or Edge recommended)
2. Go to: `http://YOUR-DASHBOARD-URL`
3. No login required (or enter password if protected)

**Navigation:**
- **Sidebar (left):** Click page names to switch views
- **Charts:** Hover mouse over data points to see details
- **Tables:** Scroll to see all rows

**Understanding the Pages:**

**1. Executive Summary (Home):**
- **Top cards:** Total revenue, customers, products, avg order value
- **Revenue trend:** 3-year sales history chart
- **Model performance:** How accurate our predictions are
- **Categories:** Which product categories perform best

**2. Revenue Forecasting:**
- **Main chart:** Blue = past, Orange = recent actual, Green = prediction
- **Table:** Compare our forecast to what actually happened
- **Bottom chart:** Which factors most influence revenue

**3. Customer Churn:**
- **Donut chart:** How many customers stopped buying (red) vs still active (green)
- **Bar chart:** Risk increases with days since last purchase
- **Table:** Top 20 customers we're losing - consider win-back campaigns

**4. Product Return Risk:**
- **Charts:** Which products/categories have highest return rates
- **Red highlighted products:** Need quality review
- **Green highlighted products:** Quality benchmarks to follow

**5. Model Performance:**
- **Technical details:** For data science team
- **Business value:** $480K-$1.28M annual benefit estimate

**Exporting Information:**
- **Charts:** Right-click → "Download plot as a PNG"
- **Tables:** Select cells → Copy (Ctrl+C / Cmd+C) → Paste in Excel

### For Analysts (Technical)

#### Advanced Usage

**Refreshing Data:**
```bash
# 1. Run latest data processing
python scripts/phase1_data_preparation.py

# 2. Dashboard auto-reloads on page refresh (F5)
```

**Modifying Dashboard:**
```bash
# 1. Edit dashboard file
nano dashboards/adventureworks_dashboard.py

# 2. Test locally
streamlit run dashboards/adventureworks_dashboard.py

# 3. Deploy (git push for cloud, or restart for local)
```

**Adding Custom Filters:**
```python
# Add to sidebar in dashboard code
selected_category = st.sidebar.selectbox(
    "Filter by Category:",
    options=["All"] + list(sales_data['CategoryName'].unique())
)

if selected_category != "All":
    sales_data = sales_data[sales_data['CategoryName'] == selected_category]
```

---

## Maintenance

### Daily Tasks

- [ ] Check dashboard is accessible (automated monitoring recommended)
- [ ] Review error logs if any (check streamlit.log or cloud logs)

### Weekly Tasks

- [ ] Review automated report output
- [ ] Check report delivery to stakeholders
- [ ] Monitor dashboard usage (Streamlit Cloud provides analytics)

### Monthly Tasks

- [ ] Retrain revenue forecasting model with new month's data
- [ ] Update churn model if accuracy drops below 98%
- [ ] Review and update high-risk products list
- [ ] Collect user feedback for dashboard improvements

### Quarterly Tasks

- [ ] Full model retraining (Phases 2-5)
- [ ] Dashboard performance review (load times, errors)
- [ ] User training refresher sessions
- [ ] Evaluate new feature requests

### Retraining Models

```bash
# Activate environment
source venv/bin/activate

# Run full pipeline
python scripts/phase1_data_preparation.py
python scripts/phase2_models_training.py
python scripts/phase3_models_training.py
python scripts/phase4_models_training.py
python scripts/phase5_optimization.py
python scripts/phase5_ensemble.py

# Dashboard will use updated models on next load
```

---

## Troubleshooting

### Dashboard Won't Start

**Error:** `ModuleNotFoundError: No module named 'streamlit'`

**Solution:**
```bash
# Verify virtual environment activated
which python  # Should show venv path

# Reinstall if needed
pip install -r requirements.txt
```

### Data Not Loading

**Error:** `FileNotFoundError`

**Solution:**
```bash
# Check data files exist
ls -lh data/processed/

# Run preparation if missing
python scripts/phase1_data_preparation.py
```

### Charts Not Rendering

**Error:** Blank spaces where charts should be

**Solution:**
1. Check browser console (F12)
2. Try different browser (Chrome recommended)
3. Clear cache: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port
lsof -i :8501

# Kill it
kill -9 <PID>

# Or use different port
streamlit run dashboard.py --server.port 8502
```

### Slow Performance

**Symptoms:** Dashboard takes >5 seconds to load

**Solution:**
1. Check data size: `wc -l data/processed/*.csv`
2. If >100K rows, consider:
   - Adding date range filters
   - Pre-aggregating data
   - Using database instead of CSV

### Automated Reports Not Generating

**Check cron is running:**
```bash
# View cron logs
grep CRON /var/log/syslog  # Linux
cat /var/log/cron  # macOS

# Test manually
python dashboards/automated_reports.py

# Check output directory
ls -lh outputs/automated_reports/
```

---

## Support & Contact

**Dashboard Issues:**
- Check logs: `streamlit.log` or cloud logs
- GitHub Issues: https://github.com/your-repo/issues

**Data Questions:**
- Review Phase 1-6 reports in `outputs/reports/`
- Contact: analytics@adventureworks.com

**Feature Requests:**
- Submit via GitHub Issues with "enhancement" tag
- Monthly review of requests

---

## Success Metrics

Track these to measure deployment success:

- ✅ **Dashboard Uptime:** Target 99.5%
- ✅ **Page Load Time:** Target < 2 seconds
- ✅ **Weekly Active Users:** Track adoption
- ✅ **Report Delivery Rate:** Target 100% (all scheduled reports sent)
- ✅ **User Satisfaction:** Quarterly survey, target >4/5 stars
- ✅ **Business Impact:** Track actual ROI vs estimated $480K-$1.28M

---

## Changelog

**v1.0 (2025-10-24):**
- Initial production deployment
- 5-page Streamlit dashboard
- Automated HTML report generation
- Deployment to Streamlit Cloud

**Future Versions:**
- v1.1: Add user authentication
- v1.2: Email report distribution
- v1.3: Real-time predictions API
- v2.0: Mobile app

---

**Document Version:** 1.0
**Last Updated:** October 24, 2025
**Maintained By:** Data Science Team
**Status:** ✅ Production Ready

*For questions or support, contact: analytics@adventureworks.com*
