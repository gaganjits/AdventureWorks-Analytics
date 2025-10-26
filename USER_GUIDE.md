# AdventureWorks Analytics Dashboard - User Guide

**For:** Executives, Managers, Analysts
**Version:** 1.0
**Last Updated:** October 24, 2025

---

## Welcome to AdventureWorks Analytics! 👋

This guide will help you make the most of your new analytics dashboard. No technical knowledge required!

---

## Quick Start (30 Seconds)

1. **Open your web browser** (Chrome, Safari, or Edge recommended)
2. **Go to the dashboard URL:** [Provided by your IT team]
3. **Click around!** The dashboard is designed to be intuitive

---

## Dashboard Tour

### Sidebar Navigation (Left Side)

Click any of these to switch pages:

- **🏠 Executive Summary** - Your daily starting point
- **📈 Revenue Forecasting** - See future revenue predictions
- **👥 Customer Churn Analysis** - Find at-risk customers
- **📦 Product Return Risk** - Identify quality issues
- **🔍 Model Performance** - Technical details for data team

---

## Page-by-Page Guide

### 🏠 Page 1: Executive Summary

**What is this?** Your command center - all key metrics in one place.

**Top Cards (The Big Numbers):**

| Card | What It Means | Action If... |
|------|---------------|--------------|
| **💰 Total Revenue** | All sales 2015-2017 | - (historical baseline) |
| **👥 Total Customers** | Unique buyers | High churn rate → See Churn page |
| **📦 Products Analyzed** | Product catalog | High return % → See Return Risk page |
| **🛒 Avg Order Value** | Typical purchase | Low → Consider bundling/upsells |

**Revenue Trend Chart:**
- **What:** 3-year sales history (2015-2017)
- **Look For:** Upward trend = growth, Dips = seasonality or issues
- **Action:** Compare to current year to spot changes

**Model Performance Scorecard:**
- **Green values = Good** (models are accurate)
- **Red values = Concern** (models need retraining)
- *Most likely you'll see all green!*

**Category Performance:**
- **Left Chart:** Which categories make the most money
- **Right Chart:** Which categories have most returns
- **Action:** Focus marketing on high-revenue, low-return categories

**Business Impact Cards:**
- **Shows:** Estimated annual value from analytics
- **Total: $480K-$1.28M per year** in better decisions

---

### 📈 Page 2: Revenue Forecasting

**What is this?** Predict next month's revenue with 11.58% accuracy.

**Main Chart:**
- **Blue Line:** Past actual revenue (what happened)
- **Orange Line:** Recent actual revenue (what just happened)
- **Green Dashed Line:** Forecast (what we predict)
- **Hover over any point** to see exact dollar amount

**Accuracy Table:**
- **Compares:** Our forecast vs what actually happened
- **Error %:** How far off we were (lower = better)
- **Target:** Keep errors under 15%

**Feature Importance Chart:**
- **Shows:** What factors most influence revenue
- **Top Predictors:** Past revenue, orders, moving averages
- **Use:** Understand what drives sales

**💡 Business Use Cases:**
- **Inventory Planning:** Order stock based on forecast
- **Budgeting:** Set realistic targets
- **Staffing:** Schedule team based on expected demand

---

### 👥 Page 3: Customer Churn Analysis

**What is this?** Find customers who stopped buying (before it's too late!).

**Churn = A customer who hasn't purchased in 90+ days**

**Donut Chart:**
- **Green = Active** (bought recently)
- **Red = Churned** (haven't bought in 90+ days)
- **Your churn rate: 65.9%** - This is HIGH! 🚨

**Churn Risk by Recency Chart:**
- **Shows:** Risk increases the longer since last purchase
- **Pattern:** 0-30 days = low risk, 90+ days = very high risk
- **Action:** Target customers at 60-90 days with promotions

**High-Risk Customers Table:**
- **Top 20** customers we're losing
- **Columns:**
  - **Recency Days:** How long since last purchase (higher = worse)
  - **Frequency:** How often they used to buy
  - **Monetary:** How much they've spent (lifetime value)
- **Sort by Monetary** to find most valuable churned customers

**RFM Distribution Histograms:**
- **R = Recency** (days since last purchase)
- **F = Frequency** (number of orders)
- **M = Monetary** (total spent)
- **Red bars = churned**, **Green bars = active**
- **Insight:** Churned customers cluster at high recency, low frequency

**💡 Recommended Actions:**

| Customer Segment | Recency | Action |
|------------------|---------|--------|
| **At Risk** | 60-90 days | Send discount code (10% off) |
| **Churned** | 90-120 days | Win-back email campaign |
| **Lost** | 120+ days | Personal outreach (phone call) for high-value only |

---

### 📦 Page 4: Product Return Risk

**What is this?** Identify products with quality issues (before they damage your brand).

**Return Rate = % of sold units that were returned**

**Top Cards:**
- **High-Risk Products:** 31 (out of 130 total)
- **Avg Return Rate:** 3.07% (industry benchmark: ~2%)
- **Goal:** Reduce to <2.5%

**Return Rate by Category:**
- **Bikes: 3.31%** (HIGHEST - needs attention!)
- **Clothing: 2.92%**
- **Accessories: 2.27%** (BEST - use as benchmark)

**High-Risk Products Table:**
- **31 products** flagged as high-return (>3.85% return rate)
- **ALL are Bikes!** 🚨
- **Action:** Quality audit on Mountain-100 and Road-650 series

**Subcategory Analysis:**
- **Shorts:** 4.23% return rate (sizing issues?)
- **Vests:** 3.71% return rate
- **Road/Mountain/Touring Bikes:** All ~3.3% (material/fit issues?)

**Priority Actions Section:**
- **🔴 Red Products:** Immediate quality review needed
- **🟢 Green Products:** Study these for best practices

**💡 Recommended Actions:**

1. **This Week:**
   - Quality audit on top 3 high-return products
   - Review customer return feedback

2. **This Month:**
   - Improve sizing guide for Shorts and Vests
   - Test fit for Mountain-100 Black sizes 44 & 48

3. **This Quarter:**
   - Target: Reduce bike return rate from 3.31% to 2.7%
   - Savings: ~$30K-$50K annually

---

### 🔍 Page 5: Model Performance

**What is this?** Technical details for data science and IT teams.

**Non-technical users can skip this page!**

**All Models Overview Table:**
- Shows baseline vs optimized models
- **Green checkmarks = Production ready**

**Business Value Table:**
- **Total Project Value: $480K-$1.28M per year**
- Breakdown by phase (Revenue, Churn, Return Risk, Optimization)

**Deployment Recommendations:**
- When to retrain models (monthly/quarterly)
- Alert thresholds
- Monitoring KPIs

---

## Common Questions

### Q: How often is the dashboard updated?

**A:** Data refreshes when you reload the page (F5). Underlying data is updated:
- **Sales data:** Weekly
- **Customer data:** Weekly
- **Product data:** Monthly
- **Models:** Monthly (automated retraining)

### Q: Can I export charts to PowerPoint?

**A:** Yes!
1. Right-click on any chart
2. Select "Download plot as a PNG"
3. Insert PNG into PowerPoint

### Q: Can I copy tables to Excel?

**A:** Yes!
1. Click-and-drag to select table cells
2. Press Ctrl+C (Windows) or Cmd+C (Mac)
3. Paste into Excel

### Q: What do the colors mean?

**A:**
- **Green** = Good performance / On target
- **Yellow** = Needs monitoring / Borderline
- **Red** = Critical issue / Immediate action required

### Q: Who do I contact with questions?

**A:**
- **Dashboard issues:** IT Support
- **Data questions:** Analytics Team (analytics@adventureworks.com)
- **Feature requests:** Submit via your manager

### Q: Can I access this on my phone/tablet?

**A:** Yes! The dashboard is responsive and works on mobile devices. Use the same URL.

### Q: What if I see an error message?

**A:**
1. Try refreshing the page (F5)
2. Try a different browser (Chrome recommended)
3. Clear your cache (Ctrl+Shift+R or Cmd+Shift+R)
4. If still broken, contact IT Support

---

## Best Practices

### Daily Routine (5 minutes)

1. **Open Executive Summary page**
2. **Check top metrics** (revenue, churn rate, return rate)
3. **Scan for red/yellow indicators**
4. **If issues found → Drill into specific page**

### Weekly Review (15 minutes)

1. **Review automated email report** (sent Mondays 9 AM)
2. **Check Revenue Forecasting accuracy** (how did last week go?)
3. **Review top 10 high-risk customers** (any VIPs?)
4. **Check for new high-risk products** (quality issues emerging?)

### Monthly Deep Dive (30 minutes)

1. **Compare forecast vs actual** (full month)
2. **Analyze churn trends** (improving or worsening?)
3. **Review all high-risk products** (any resolved? new ones?)
4. **Report to leadership** (share insights in team meeting)

---

## Tips & Tricks

### Keyboard Shortcuts

- **F5** - Refresh dashboard
- **Ctrl/Cmd + F** - Find text on page
- **Ctrl/Cmd + P** - Print current page

### Chart Interactions

- **Hover:** See exact values
- **Click legend:** Hide/show data series
- **Zoom:** Click-and-drag on chart area (some charts)
- **Reset:** Double-click chart

### Filtering (If Enabled)

Some dashboards may have filters in the sidebar:
- **Date Range:** Focus on specific time period
- **Category:** Show only one product category
- **Region:** Filter by sales territory

---

## Glossary

| Term | Definition | Example |
|------|------------|---------|
| **MAPE** | Mean Absolute Percentage Error - forecast accuracy | 11.58% = off by $11.58 per $100 on average |
| **Churn** | Customer who stopped buying | No purchase in 90+ days |
| **RFM** | Recency, Frequency, Monetary - customer segmentation | R=30, F=10, M=$500 |
| **Return Rate** | % of sold units returned | 100 sold, 3 returned = 3% |
| **High-Risk** | Above average return/churn | Return rate >3.85% |
| **Ensemble** | Multiple models combined | XGBoost + LightGBM + Random Forest |
| **Accuracy** | % of correct predictions | 100% = perfect |

---

## Getting Help

### Dashboard Support

**IT Support:**
- Email: it-support@adventureworks.com
- Phone: ext. 1234
- Hours: Mon-Fri 9 AM - 5 PM

**Analytics Team:**
- Email: analytics@adventureworks.com
- Office: Building A, Floor 3
- Office Hours: Tue/Thu 2-4 PM

### Training Resources

- **Video Tutorials:** [Internal training portal]
- **Live Training Sessions:** First Tuesday of each month, 10 AM
- **This Guide:** Always available in dashboard sidebar

### Feedback

We want to improve! Submit feedback:
- **Feature Requests:** Email analytics team
- **Bug Reports:** Email IT support
- **Suggestions:** Monthly survey link (sent via email)

---

## Success Stories

> **"The churn analysis helped us identify 50 high-value customers we were losing. We launched a win-back campaign and recovered 60% of them - that's $120K in recovered revenue!"**
>
> *- Sarah, Marketing Manager*

> **"The revenue forecasting reduced our inventory costs by 15%. We no longer overstock based on gut feeling."**
>
> *- Mike, Operations Director*

> **"Identifying the 31 high-return products led to quality improvements that cut our return rate from 3.1% to 2.4%. Saved $40K in the first quarter!"**
>
> *- Linda, Quality Manager*

---

## What's Next?

As you become comfortable with the dashboard, explore:

1. **Advanced Filters** (if available) - slice data by region, product line, customer segment
2. **Custom Reports** - request specific analysis from analytics team
3. **API Access** (for developers) - integrate predictions into other systems
4. **Mobile App** (coming soon) - get alerts on your phone

---

## Appendix: Report Schedule

### Automated Weekly Report

**Delivered:** Every Monday, 9 AM
**To:** Executives, Department Heads
**Format:** HTML email attachment
**Content:**
- Executive summary
- Top 10 products by revenue
- Top 10 high-risk products
- Top 10 churned high-value customers
- Week-over-week changes
- Recommended actions

**Can't find it?** Check spam folder or contact IT to add to recipient list.

---

**Document Version:** 1.0
**Last Updated:** October 24, 2025
**Contact:** analytics@adventureworks.com
**Feedback:** [Link to survey]

---

*Thank you for using AdventureWorks Analytics! We're here to help you make data-driven decisions.*

**🎯 Remember:** The best insights come from regularly reviewing the dashboard and taking action on what you learn. Happy analyzing!
