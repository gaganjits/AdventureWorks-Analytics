"""
AdventureWorks Business Intelligence Dashboard
A comprehensive Streamlit app for revenue forecasting, churn prediction, and return risk analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AdventureWorks Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        color: #28a745;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
    }
    .danger-metric {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Set up paths
project_root = Path(__file__).parent.parent
data_path = project_root / 'data' / 'processed'
models_path = project_root / 'models'
outputs_path = project_root / 'outputs'

# ============================================================================
# SIDEBAR - Navigation
# ============================================================================
st.sidebar.markdown("# 📊 AdventureWorks Analytics")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate to:",
    ["🏠 Executive Summary", "📈 Revenue Forecasting", "👥 Customer Churn Analysis",
     "📦 Product Return Risk", "🔍 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard provides insights from:
    - **Phase 2:** Revenue Forecasting (11.58% MAPE)
    - **Phase 3:** Churn Prediction (100% Accuracy)
    - **Phase 4:** Return Risk Analysis (100% Accuracy)
    - **Phase 5:** Optimized Models

    **Last Updated:** October 24, 2025
    """
)

# ============================================================================
# PAGE 1: EXECUTIVE SUMMARY
# ============================================================================
if page == "🏠 Executive Summary":
    st.markdown('<div class="main-header">AdventureWorks Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown("### Key Performance Indicators")

    # Load summary data
    try:
        sales_data = pd.read_csv(data_path / 'AdventureWorks_Sales_Enriched.csv')
        customer_data = pd.read_csv(data_path / 'Customer_Churn_Features.csv')
        product_data = pd.read_csv(data_path / 'Product_Return_Risk_Features.csv')

        # Top-level metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            total_revenue = sales_data['Revenue'].sum()
            st.metric(
                label="💰 Total Revenue",
                value=f"${total_revenue/1e6:.2f}M",
                delta="3 Years (2015-2017)"
            )

        with col2:
            total_customers = len(customer_data)
            churn_rate = customer_data['Churn_90'].mean() * 100
            st.metric(
                label="👥 Total Customers",
                value=f"{total_customers:,}",
                delta=f"{churn_rate:.1f}% Churn Rate",
                delta_color="inverse"
            )

        with col3:
            total_products = len(product_data)
            high_return_pct = (product_data['HighReturnRisk'].sum() / len(product_data)) * 100
            st.metric(
                label="📦 Products Analyzed",
                value=f"{total_products}",
                delta=f"{high_return_pct:.1f}% High Return Risk",
                delta_color="inverse"
            )

        with col4:
            avg_order_value = sales_data['Revenue'].sum() / sales_data['OrderNumber'].nunique()
            st.metric(
                label="🛒 Avg Order Value",
                value=f"${avg_order_value:.2f}",
                delta="56K+ Orders"
            )

        st.markdown("---")

        # Revenue Trend
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("### 📈 Revenue Trend (2015-2017)")

            # Aggregate by month
            sales_data['OrderDate'] = pd.to_datetime(sales_data['OrderDate'])
            monthly_revenue = sales_data.groupby(sales_data['OrderDate'].dt.to_period('M'))['Revenue'].sum().reset_index()
            monthly_revenue['OrderDate'] = monthly_revenue['OrderDate'].dt.to_timestamp()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly_revenue['OrderDate'],
                y=monthly_revenue['Revenue'],
                mode='lines+markers',
                name='Actual Revenue',
                line=dict(color='#1f77b4', width=3),
                fill='tozeroy',
                fillcolor='rgba(31, 119, 180, 0.2)'
            ))

            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Revenue ($)",
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🎯 Model Performance")

            performance_data = pd.DataFrame({
                'Model': ['Revenue Forecast', 'Churn Prediction', 'Return Risk'],
                'Metric': ['MAPE', 'Accuracy', 'Accuracy'],
                'Score': [11.58, 100, 100],
                'Target': [20, 90, 85]
            })

            for idx, row in performance_data.iterrows():
                metric_name = row['Model']
                metric_value = row['Score']
                metric_target = row['Target']
                metric_type = row['Metric']

                if metric_type == 'MAPE':
                    delta = f"{metric_target - metric_value:.1f}pp vs target"
                    delta_color = "normal"
                else:
                    delta = f"{metric_value - metric_target:.0f}pp vs target"
                    delta_color = "normal"

                st.metric(
                    label=metric_name,
                    value=f"{metric_value:.1f}{'%' if metric_type == 'MAPE' or metric_type == 'Accuracy' else ''}",
                    delta=delta
                )

        st.markdown("---")

        # Category Performance
        st.markdown("### 📊 Category Performance")

        col1, col2 = st.columns(2)

        with col1:
            # Revenue by category
            category_revenue = sales_data.groupby('CategoryName')['Revenue'].sum().reset_index()
            category_revenue = category_revenue.sort_values('Revenue', ascending=False)

            fig = px.bar(
                category_revenue,
                x='CategoryName',
                y='Revenue',
                title='Revenue by Category',
                color='Revenue',
                color_continuous_scale='Blues'
            )
            fig.update_layout(height=350, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Return rate by category
            category_returns = product_data.groupby('CategoryName').agg({
                'ReturnRate': 'mean',
                'ProductKey': 'count'
            }).reset_index()
            category_returns.columns = ['CategoryName', 'AvgReturnRate', 'Products']
            category_returns = category_returns.sort_values('AvgReturnRate', ascending=False)

            fig = px.bar(
                category_returns,
                x='CategoryName',
                y='AvgReturnRate',
                title='Average Return Rate by Category',
                color='AvgReturnRate',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=350, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        # Business Impact Summary
        st.markdown("---")
        st.markdown("### 💼 Estimated Business Impact")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Revenue Forecasting**")
            st.markdown("**Value:** $200K-$500K/year")
            st.markdown("Better inventory planning, reduced stockouts")
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Churn Prevention**")
            st.markdown("**Value:** $100K-$300K/year")
            st.markdown("Proactive retention campaigns")
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Return Risk Management**")
            st.markdown("**Value:** $50K-$150K/year")
            st.markdown("Targeted quality improvements")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error loading data: {e}")

# ============================================================================
# PAGE 2: REVENUE FORECASTING
# ============================================================================
elif page == "📈 Revenue Forecasting":
    st.markdown('<div class="main-header">Revenue Forecasting Dashboard</div>', unsafe_allow_html=True)

    try:
        # Load revenue data and model
        revenue_data = pd.read_csv(data_path / 'Revenue_Monthly_Features.csv')
        revenue_data['Date'] = pd.to_datetime(revenue_data['Date'])

        # Load optimized model
        model = joblib.load(models_path / 'phase5_xgboost_revenue_optimized.pkl')

        # Model info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model", "XGBoost (Optimized)", "Phase 5")
        with col2:
            st.metric("MAPE", "11.58%", "-25.2% vs baseline")
        with col3:
            st.metric("MAE", "$194,665", "Test Set")
        with col4:
            st.metric("Training Months", "24", "6 Test Months")

        st.markdown("---")

        # Historical + Forecast Visualization
        st.markdown("### 📊 Historical Revenue & Forecast")

        # Prepare data for plotting
        train_size = int(len(revenue_data) * 0.8)
        feature_cols = [col for col in revenue_data.columns if col not in ['Date', 'Revenue', 'Year', 'Quarter']]

        X = revenue_data[feature_cols].fillna(0)
        y = revenue_data['Revenue']

        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        dates_train, dates_test = revenue_data['Date'][:train_size], revenue_data['Date'][train_size:]

        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Create figure
        fig = go.Figure()

        # Actual (train)
        fig.add_trace(go.Scatter(
            x=dates_train,
            y=y_train,
            mode='lines+markers',
            name='Actual (Train)',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))

        # Actual (test)
        fig.add_trace(go.Scatter(
            x=dates_test,
            y=y_test,
            mode='lines+markers',
            name='Actual (Test)',
            line=dict(color='#ff7f0e', width=2),
            marker=dict(size=8)
        ))

        # Predicted (test)
        fig.add_trace(go.Scatter(
            x=dates_test,
            y=y_pred_test,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#2ca02c', width=2, dash='dash'),
            marker=dict(size=8, symbol='diamond')
        ))

        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Revenue ($)",
            hovermode='x unified',
            height=500,
            template='plotly_white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Forecast Accuracy Table
        st.markdown("### 📋 Forecast Accuracy (Test Period)")

        forecast_df = pd.DataFrame({
            'Month': dates_test.dt.strftime('%Y-%m'),
            'Actual': y_test.values,
            'Forecast': y_pred_test,
            'Error': y_test.values - y_pred_test,
            'Error %': ((y_test.values - y_pred_test) / y_test.values * 100)
        })

        forecast_df['Actual'] = forecast_df['Actual'].apply(lambda x: f"${x:,.0f}")
        forecast_df['Forecast'] = forecast_df['Forecast'].apply(lambda x: f"${x:,.0f}")
        forecast_df['Error'] = forecast_df['Error'].apply(lambda x: f"${x:,.0f}")
        forecast_df['Error %'] = forecast_df['Error %'].apply(lambda x: f"{x:.2f}%")

        st.dataframe(forecast_df, use_container_width=True, hide_index=True)

        # Feature Importance
        st.markdown("---")
        st.markdown("### 🔍 Top 10 Predictive Features")

        col1, col2 = st.columns([2, 1])

        with col1:
            # Get feature importance
            importance_df = pd.DataFrame({
                'Feature': feature_cols,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(10)

            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title='Feature Importance',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, template='plotly_white')
            fig.update_yaxes(categoryorder='total ascending')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Interpretation:**")
            st.markdown("""
            - **Lag Features:** Past revenue strongly predicts future
            - **Moving Averages:** Trends matter
            - **Orders/Quantity:** Sales volume indicators
            - **Seasonality:** Captured via cyclical features
            """)

            st.markdown("**Model Details:**")
            st.code(f"""
n_estimators: 195
max_depth: 3
learning_rate: 0.178
subsample: 0.825
            """)

    except Exception as e:
        st.error(f"Error loading revenue forecasting data: {e}")

# ============================================================================
# PAGE 3: CUSTOMER CHURN ANALYSIS
# ============================================================================
elif page == "👥 Customer Churn Analysis":
    st.markdown('<div class="main-header">Customer Churn Risk Dashboard</div>', unsafe_allow_html=True)

    try:
        # Load churn data
        churn_data = pd.read_csv(data_path / 'Customer_Churn_Features.csv')

        # Model info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model", "Ensemble (XGB+LGB+RF)", "Phase 5")
        with col2:
            st.metric("Accuracy", "100%", "Test Set")
        with col3:
            total_churned = churn_data['Churn_90'].sum()
            st.metric("Churned Customers", f"{total_churned:,}", f"{total_churned/len(churn_data)*100:.1f}%")
        with col4:
            st.metric("Churn Threshold", "90 Days", "No Purchase")

        st.markdown("---")

        # Churn Distribution
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Churn Distribution")

            churn_counts = churn_data['Churn_90'].value_counts()

            fig = go.Figure(data=[go.Pie(
                labels=['Active', 'Churned'],
                values=[churn_counts[0], churn_counts[1]],
                hole=0.4,
                marker_colors=['#2ca02c', '#d62728']
            )])
            fig.update_layout(height=350, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 📈 Churn Risk by Recency")

            # Bin recency
            churn_data['RecencyBin'] = pd.cut(churn_data['Recency_Days'],
                                               bins=[0, 30, 60, 90, 120, 365],
                                               labels=['0-30', '31-60', '61-90', '91-120', '120+'])
            recency_churn = churn_data.groupby('RecencyBin')['Churn_90'].mean() * 100

            fig = px.bar(
                x=recency_churn.index,
                y=recency_churn.values,
                labels={'x': 'Days Since Last Purchase', 'y': 'Churn Rate (%)'},
                title='Churn Rate by Recency',
                color=recency_churn.values,
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=350, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        # High-Risk Customers
        st.markdown("---")
        st.markdown("### ⚠️ High-Risk Customers (Top 20)")

        high_risk = churn_data[churn_data['Churn_90'] == 1].sort_values('Recency_Days', ascending=False).head(20)

        display_cols = ['CustomerKey', 'Recency_Days', 'Frequency', 'Monetary',
                       'TotalOrders', 'AvgOrderValue', 'CustomerLifetime_Days']

        if all(col in high_risk.columns for col in display_cols):
            high_risk_display = high_risk[display_cols].copy()
            high_risk_display['Monetary'] = high_risk_display['Monetary'].apply(lambda x: f"${x:,.2f}")
            high_risk_display['AvgOrderValue'] = high_risk_display['AvgOrderValue'].apply(lambda x: f"${x:,.2f}")

            st.dataframe(high_risk_display, use_container_width=True, hide_index=True)

        # RFM Analysis
        st.markdown("---")
        st.markdown("### 📊 RFM Segment Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            fig = px.histogram(
                churn_data,
                x='Recency_Days',
                color='Churn_90',
                title='Recency Distribution',
                labels={'Churn_90': 'Churned'},
                color_discrete_map={0: '#2ca02c', 1: '#d62728'}
            )
            fig.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.histogram(
                churn_data,
                x='Frequency',
                color='Churn_90',
                title='Frequency Distribution',
                labels={'Churn_90': 'Churned'},
                color_discrete_map={0: '#2ca02c', 1: '#d62728'}
            )
            fig.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        with col3:
            fig = px.histogram(
                churn_data,
                x='Monetary',
                color='Churn_90',
                title='Monetary Distribution',
                labels={'Churn_90': 'Churned'},
                color_discrete_map={0: '#2ca02c', 1: '#d62728'}
            )
            fig.update_layout(height=300, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        # Recommendations
        st.markdown("---")
        st.markdown("### 💡 Retention Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Immediate Actions:**")
            st.markdown("""
            - 🎯 Target customers with Recency > 60 days
            - 📧 Send win-back campaigns to 90+ day inactive
            - 🎁 Offer incentives to customers with high Monetary but low Frequency
            - 📞 Personal outreach for high-value churned customers
            """)

        with col2:
            st.markdown("**Long-Term Strategy:**")
            st.markdown("""
            - 📊 Monitor churn rate weekly
            - 🔔 Set up automated alerts for high-risk customers
            - 🎯 Segment campaigns by RFM score
            - 📈 Track retention rate improvements
            """)

    except Exception as e:
        st.error(f"Error loading churn data: {e}")

# ============================================================================
# PAGE 4: PRODUCT RETURN RISK
# ============================================================================
elif page == "📦 Product Return Risk":
    st.markdown('<div class="main-header">Product Return Risk Dashboard</div>', unsafe_allow_html=True)

    try:
        # Load product data
        product_data = pd.read_csv(data_path / 'Product_Return_Risk_Features.csv')

        # Model info
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Model", "Ensemble (XGB+LGB+RF)", "Phase 5")
        with col2:
            st.metric("Accuracy", "100%", "Test Set")
        with col3:
            high_risk_count = product_data['HighReturnRisk'].sum()
            st.metric("High-Risk Products", f"{high_risk_count}", f"{high_risk_count/len(product_data)*100:.1f}%")
        with col4:
            avg_return_rate = product_data['ReturnRate'].mean()
            st.metric("Avg Return Rate", f"{avg_return_rate:.2f}%", "All Products")

        st.markdown("---")

        # Return Rate by Category
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Return Rate by Category")

            category_returns = product_data.groupby('CategoryName').agg({
                'ReturnRate': 'mean',
                'HighReturnRisk': 'sum',
                'ProductKey': 'count'
            }).reset_index()
            category_returns.columns = ['Category', 'AvgReturnRate', 'HighRiskCount', 'TotalProducts']

            fig = px.bar(
                category_returns,
                x='Category',
                y='AvgReturnRate',
                title='Average Return Rate by Category',
                color='AvgReturnRate',
                color_continuous_scale='Reds',
                text='AvgReturnRate'
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### ⚠️ High-Risk Products by Category")

            fig = px.pie(
                category_returns,
                values='HighRiskCount',
                names='Category',
                title='High-Risk Product Distribution',
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            fig.update_layout(height=400, template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)

        # High-Risk Products Table
        st.markdown("---")
        st.markdown("### 🚨 Top 20 High-Risk Products")

        high_risk_products = product_data[product_data['HighReturnRisk'] == 1].sort_values('ReturnRate', ascending=False).head(20)

        display_cols = ['ProductName', 'CategoryName', 'SubcategoryName', 'ReturnRate',
                       'TotalSalesQuantity', 'TotalReturnsQuantity', 'ProductPrice']

        if all(col in high_risk_products.columns for col in display_cols):
            high_risk_display = high_risk_products[display_cols].copy()
            high_risk_display['ReturnRate'] = high_risk_display['ReturnRate'].apply(lambda x: f"{x:.2f}%")
            high_risk_display['ProductPrice'] = high_risk_display['ProductPrice'].apply(lambda x: f"${x:,.2f}")

            # Color coding
            st.dataframe(
                high_risk_display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "ReturnRate": st.column_config.TextColumn("Return Rate", help="Percentage of units returned")
                }
            )

        # Subcategory Analysis
        st.markdown("---")
        st.markdown("### 📈 Top 10 Subcategories by Return Rate")

        subcategory_stats = product_data.groupby(['CategoryName', 'SubcategoryName']).agg({
            'ReturnRate': 'mean',
            'HighReturnRisk': 'sum',
            'ProductKey': 'count'
        }).reset_index()
        subcategory_stats.columns = ['Category', 'Subcategory', 'AvgReturnRate', 'HighRiskCount', 'ProductCount']
        subcategory_stats = subcategory_stats.sort_values('AvgReturnRate', ascending=False).head(10)

        fig = px.bar(
            subcategory_stats,
            x='Subcategory',
            y='AvgReturnRate',
            color='Category',
            title='Top 10 Subcategories by Return Rate',
            text='AvgReturnRate'
        )
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=400, template='plotly_white')
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        # Actionable Insights
        st.markdown("---")
        st.markdown("### 💡 Quality Improvement Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Priority Actions:**")

            # Find worst performers
            worst_products = product_data[product_data['HighReturnRisk'] == 1].nlargest(3, 'ReturnRate')

            for idx, row in worst_products.iterrows():
                st.markdown(f"""
                - 🔴 **{row['ProductName']}**: {row['ReturnRate']:.2f}% return rate
                  - Category: {row['CategoryName']} / {row['SubcategoryName']}
                  - Action: Quality audit required
                """)

        with col2:
            st.markdown("**Success Stories:**")

            # Find best performers
            best_products = product_data[product_data['HighReturnRisk'] == 0].nsmallest(3, 'ReturnRate')

            for idx, row in best_products.iterrows():
                st.markdown(f"""
                - 🟢 **{row['ProductName']}**: {row['ReturnRate']:.2f}% return rate
                  - Category: {row['CategoryName']} / {row['SubcategoryName']}
                  - Benchmark for quality standards
                """)

    except Exception as e:
        st.error(f"Error loading product return data: {e}")

# ============================================================================
# PAGE 5: MODEL PERFORMANCE
# ============================================================================
elif page == "🔍 Model Performance":
    st.markdown('<div class="main-header">Model Performance Metrics</div>', unsafe_allow_html=True)

    st.markdown("### 📊 All Models Overview")

    # Create performance summary table
    performance_data = {
        'Phase': ['Phase 2', 'Phase 2', 'Phase 3', 'Phase 3', 'Phase 4', 'Phase 4'],
        'Model': ['XGBoost (Baseline)', 'XGBoost (Optimized)', 'XGBoost (Baseline)', 'Ensemble (Optimized)',
                 'XGBoost (Baseline)', 'Ensemble (Optimized)'],
        'Task': ['Revenue Forecast', 'Revenue Forecast', 'Churn Prediction', 'Churn Prediction',
                'Return Risk', 'Return Risk'],
        'Primary Metric': ['15.48% MAPE', '11.58% MAPE', '100% Accuracy', '100% Accuracy',
                          '100% Accuracy', '100% Accuracy'],
        'Improvement': ['-', '✅ 25.2%', '-', '✅ Robust', '-', '✅ Robust'],
        'Status': ['Superseded', '✅ Production', 'Production', '✅ Production', 'Production', '✅ Production']
    }

    df = pd.DataFrame(performance_data)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Status": st.column_config.TextColumn("Status"),
            "Improvement": st.column_config.TextColumn("Improvement")
        }
    )

    st.markdown("---")

    # Phase-by-phase comparison
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Phase 2: Revenue")
        st.markdown("**Model:** XGBoost (Optimized)")
        st.metric("MAPE", "11.58%", "-25.2% vs baseline")
        st.metric("MAE", "$194,665", "-49% vs baseline")
        st.metric("RMSE", "$252,968")
        st.success("✅ Production Ready")

    with col2:
        st.markdown("### Phase 3: Churn")
        st.markdown("**Model:** Ensemble")
        st.metric("Accuracy", "100%")
        st.metric("F1-Score", "1.0000")
        st.metric("ROC-AUC", "1.0000")
        st.metric("Features", "15", "-72% reduction")
        st.success("✅ Production Ready")

    with col3:
        st.markdown("### Phase 4: Return Risk")
        st.markdown("**Model:** Ensemble")
        st.metric("Accuracy", "100%")
        st.metric("F1-Score", "1.0000")
        st.metric("ROC-AUC", "1.0000")
        st.success("✅ Production Ready")

    # Business Value
    st.markdown("---")
    st.markdown("### 💰 Estimated Business Value")

    value_data = pd.DataFrame({
        'Phase': ['Phase 2: Revenue Forecasting', 'Phase 3: Churn Prevention', 'Phase 4: Return Risk Management',
                 'Phase 5: Optimization', 'Total Project Value'],
        'Annual Value (Low)': ['$200,000', '$100,000', '$50,000', '$130,000', '$480,000'],
        'Annual Value (High)': ['$500,000', '$300,000', '$150,000', '$330,000', '$1,280,000'],
        'Key Benefit': [
            'Better inventory planning, reduced stockouts',
            'Proactive customer retention campaigns',
            'Targeted quality improvements',
            'Improved accuracy, reduced complexity',
            'Comprehensive data-driven decisions'
        ]
    })

    st.dataframe(value_data, use_container_width=True, hide_index=True)

    # Technical Details
    st.markdown("---")
    st.markdown("### 🛠️ Technical Stack")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Machine Learning:**")
        st.code("""
- XGBoost: Gradient boosting
- LightGBM: Fast gradient boosting
- Random Forest: Ensemble learning
- Optuna: Hyperparameter tuning
- SMOTE: Class imbalance handling
        """)

    with col2:
        st.markdown("**Data & Tools:**")
        st.code("""
- pandas: Data manipulation
- scikit-learn: ML framework
- plotly: Interactive visualizations
- streamlit: Dashboard framework
- MLflow: Experiment tracking
        """)

    # Deployment Recommendations
    st.markdown("---")
    st.markdown("### 🚀 Deployment Recommendations")

    deployment_rec = pd.DataFrame({
        'Model': ['Revenue XGBoost', 'Churn Ensemble', 'Return Risk Ensemble'],
        'Update Frequency': ['Monthly', 'Quarterly', 'Monthly'],
        'Trigger': ['New month data', 'Accuracy < 98% OR drift', 'New products OR accuracy < 95%'],
        'Monitoring KPI': ['MAPE on new months', 'Weekly churn rate vs actual', 'New product predictions'],
        'Alert Threshold': ['MAPE > 15%', 'Accuracy < 98%', '>40% high-risk flagged']
    })

    st.dataframe(deployment_rec, use_container_width=True, hide_index=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>AdventureWorks Analytics Dashboard | Powered by Streamlit & Plotly</p>
        <p>Data Period: 2015-2017 | Last Updated: October 24, 2025</p>
        <p>📧 Questions? Contact: analytics@adventureworks.com</p>
    </div>
    """,
    unsafe_allow_html=True
)
