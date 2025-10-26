"""
Phase 8: Customer Segmentation
Advanced Analytics - K-Means Clustering

This script segments customers into distinct groups based on:
- RFM (Recency, Frequency, Monetary) values
- Demographics (Income, Children, Education)
- Purchase behavior patterns

Output:
- Customer segments with profiles
- Segment visualizations
- Actionable insights per segment

Author: AdventureWorks Data Science Team
Date: October 25, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "customer_segmentation"
OUTPUTS_DIR = BASE_DIR / "outputs" / "plots"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PHASE 8: CUSTOMER SEGMENTATION")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[1/7] Loading customer data...")

# Load customer RFM data
customer_rfm_path = DATA_DIR / "AdventureWorks_Customer_RFM.csv"
df = pd.read_csv(customer_rfm_path)

print(f"✓ Loaded {len(df):,} customers")
print(f"  Features: {list(df.columns)[:10]}...")

# ============================================================================
# STEP 2: FEATURE ENGINEERING FOR SEGMENTATION
# ============================================================================

print("\n[2/7] Engineering segmentation features...")

# Select features for clustering
segmentation_features = []

# RFM features
# Standardize column names
if 'Recency_Days' in df.columns:
    df['Recency'] = df['Recency_Days']
    segmentation_features.append('Recency')
elif 'Recency' in df.columns:
    segmentation_features.append('Recency')

if 'Frequency' in df.columns:
    segmentation_features.append('Frequency')
if 'Monetary' in df.columns:
    segmentation_features.append('Monetary')

# Derived RFM features
if 'AvgOrderValue' in df.columns:
    segmentation_features.append('AvgOrderValue')
elif 'Frequency' in df.columns and 'Monetary' in df.columns:
    df['AvgOrderValue'] = df['Monetary'] / df['Frequency'].replace(0, 1)
    segmentation_features.append('AvgOrderValue')

if 'TotalQuantity' in df.columns:
    segmentation_features.append('TotalQuantity')

# Demographic features
if 'AnnualIncome' in df.columns:
    # Convert from string format if needed
    if df['AnnualIncome'].dtype == 'object':
        df['AnnualIncome'] = df['AnnualIncome'].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
        df['AnnualIncome'] = pd.to_numeric(df['AnnualIncome'], errors='coerce')
    segmentation_features.append('AnnualIncome')

if 'TotalChildren' in df.columns:
    segmentation_features.append('TotalChildren')

# Create additional behavioral features
if 'Recency' in df.columns:
    df['DaysSinceLastPurchase_Log'] = np.log1p(df['Recency'])
    segmentation_features.append('DaysSinceLastPurchase_Log')

if 'Frequency' in df.columns:
    df['PurchaseFrequency_Log'] = np.log1p(df['Frequency'])
    segmentation_features.append('PurchaseFrequency_Log')

if 'Monetary' in df.columns:
    df['LifetimeValue_Log'] = np.log1p(df['Monetary'])
    segmentation_features.append('LifetimeValue_Log')

# Engagement score
if 'Recency' in df.columns and 'Frequency' in df.columns and 'Monetary' in df.columns:
    df['EngagementScore'] = (
        (df['Frequency'] * df['Monetary']) / (df['Recency'] + 1)
    )
    segmentation_features.append('EngagementScore')

print(f"✓ Created {len(segmentation_features)} segmentation features:")
for feat in segmentation_features:
    print(f"  - {feat}")

# Prepare data for clustering
df_segment = df[segmentation_features].copy()
df_segment = df_segment.fillna(df_segment.median())

# ============================================================================
# STEP 3: DETERMINE OPTIMAL NUMBER OF CLUSTERS
# ============================================================================

print("\n[3/7] Determining optimal number of clusters...")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_segment)

# Elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    print(f"  K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")

# Plot elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (K)')
ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)')
ax1.set_title('Elbow Method for Optimal K')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, silhouette_scores, 'go-')
ax2.set_xlabel('Number of Clusters (K)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score by K')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "customer_segmentation_elbow.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved elbow curve plot")

# Select optimal K (highest silhouette score)
optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"\n✓ Optimal number of clusters: {optimal_k}")

# ============================================================================
# STEP 4: PERFORM K-MEANS CLUSTERING
# ============================================================================

print(f"\n[4/7] Performing K-Means clustering with K={optimal_k}...")

# Final K-Means model
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=20, max_iter=500)
df['Segment'] = kmeans_final.fit_predict(X_scaled)

# Calculate clustering metrics
silhouette_avg = silhouette_score(X_scaled, df['Segment'])
davies_bouldin = davies_bouldin_score(X_scaled, df['Segment'])

print(f"✓ Clustering complete")
print(f"  Silhouette Score: {silhouette_avg:.3f} (higher is better)")
print(f"  Davies-Bouldin Index: {davies_bouldin:.3f} (lower is better)")

# Save model and scaler
joblib.dump(kmeans_final, MODELS_DIR / "kmeans_model.pkl")
joblib.dump(scaler, MODELS_DIR / "feature_scaler.pkl")
print(f"✓ Saved model to {MODELS_DIR}")

# ============================================================================
# STEP 5: ANALYZE SEGMENTS
# ============================================================================

print(f"\n[5/7] Analyzing customer segments...")

segment_profiles = []

for segment_id in range(optimal_k):
    segment_data = df[df['Segment'] == segment_id]

    profile = {
        'Segment': segment_id,
        'Count': len(segment_data),
        'Percentage': len(segment_data) / len(df) * 100,
        'Avg_Recency': segment_data['Recency'].mean() if 'Recency' in df.columns else None,
        'Avg_Frequency': segment_data['Frequency'].mean() if 'Frequency' in df.columns else None,
        'Avg_Monetary': segment_data['Monetary'].mean() if 'Monetary' in df.columns else None,
        'Avg_OrderValue': segment_data['AvgOrderValue'].mean() if 'AvgOrderValue' in df.columns else None,
        'Avg_Income': segment_data['AnnualIncome'].mean() if 'AnnualIncome' in df.columns else None,
        'Total_Revenue': segment_data['Monetary'].sum() if 'Monetary' in df.columns else None,
    }

    segment_profiles.append(profile)

segment_df = pd.DataFrame(segment_profiles)

# Assign segment names based on characteristics
def assign_segment_name(row):
    if row['Avg_Monetary'] > segment_df['Avg_Monetary'].quantile(0.75):
        if row['Avg_Recency'] < segment_df['Avg_Recency'].quantile(0.25):
            return "VIP Active"
        else:
            return "VIP At-Risk"
    elif row['Avg_Frequency'] > segment_df['Avg_Frequency'].quantile(0.75):
        return "Loyal Regulars"
    elif row['Avg_Recency'] < segment_df['Avg_Recency'].quantile(0.25):
        return "New Engaged"
    elif row['Avg_Recency'] > segment_df['Avg_Recency'].quantile(0.75):
        return "Hibernating"
    else:
        return "Potential Loyalists"

segment_df['Segment_Name'] = segment_df.apply(assign_segment_name, axis=1)

print("\n" + "="*90)
print("CUSTOMER SEGMENT PROFILES")
print("="*90)

for _, row in segment_df.iterrows():
    print(f"\nSegment {row['Segment']}: {row['Segment_Name']}")
    print(f"  Customers: {row['Count']:,} ({row['Percentage']:.1f}%)")
    if row['Avg_Recency'] is not None:
        print(f"  Avg Days Since Last Purchase: {row['Avg_Recency']:.0f} days")
    if row['Avg_Frequency'] is not None:
        print(f"  Avg Purchase Frequency: {row['Avg_Frequency']:.1f} orders")
    if row['Avg_Monetary'] is not None:
        print(f"  Avg Lifetime Value: ${row['Avg_Monetary']:,.2f}")
    if row['Avg_OrderValue'] is not None:
        print(f"  Avg Order Value: ${row['Avg_OrderValue']:,.2f}")
    if row['Avg_Income'] is not None:
        print(f"  Avg Annual Income: ${row['Avg_Income']:,.0f}")
    if row['Total_Revenue'] is not None:
        print(f"  Total Segment Revenue: ${row['Total_Revenue']:,.2f}")

# Save segment profiles
segment_df.to_csv(DATA_DIR / "Customer_Segment_Profiles.csv", index=False)
print(f"\n✓ Saved segment profiles to {DATA_DIR / 'Customer_Segment_Profiles.csv'}")

# ============================================================================
# STEP 6: VISUALIZE SEGMENTS
# ============================================================================

print(f"\n[6/7] Creating segment visualizations...")

# PCA for 2D visualization
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]

# Create segment name mapping
segment_name_map = dict(zip(segment_df['Segment'], segment_df['Segment_Name']))
df['Segment_Name'] = df['Segment'].map(segment_name_map)

# Plot 1: PCA scatter plot
plt.figure(figsize=(14, 10))

# Main scatter plot
plt.subplot(2, 2, 1)
for segment_id in range(optimal_k):
    segment_data = df[df['Segment'] == segment_id]
    plt.scatter(segment_data['PCA1'], segment_data['PCA2'],
                label=segment_name_map[segment_id], alpha=0.6, s=50)

plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('Customer Segments (PCA Visualization)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

# Plot 2: Segment size distribution
plt.subplot(2, 2, 2)
segment_sizes = df['Segment_Name'].value_counts()
colors = plt.cm.Set3(range(len(segment_sizes)))
plt.pie(segment_sizes, labels=segment_sizes.index, autopct='%1.1f%%', colors=colors)
plt.title('Customer Distribution by Segment')

# Plot 3: Recency vs Monetary by segment
plt.subplot(2, 2, 3)
if 'Recency' in df.columns and 'Monetary' in df.columns:
    for segment_id in range(optimal_k):
        segment_data = df[df['Segment'] == segment_id]
        plt.scatter(segment_data['Recency'], segment_data['Monetary'],
                    label=segment_name_map[segment_id], alpha=0.6, s=30)
    plt.xlabel('Recency (Days)')
    plt.ylabel('Monetary (Lifetime Value)')
    plt.title('Recency vs Monetary by Segment')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)

# Plot 4: Average metrics by segment
plt.subplot(2, 2, 4)
metrics = ['Avg_Frequency', 'Avg_Monetary']
x = np.arange(len(segment_df))
width = 0.35

if 'Avg_Frequency' in segment_df.columns and 'Avg_Monetary' in segment_df.columns:
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.bar(x - width/2, segment_df['Avg_Frequency'], width, label='Avg Frequency', color='skyblue')
    ax2.bar(x + width/2, segment_df['Avg_Monetary'], width, label='Avg Monetary ($)', color='coral')

    ax1.set_xlabel('Segment')
    ax1.set_ylabel('Average Frequency', color='skyblue')
    ax2.set_ylabel('Average Monetary ($)', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels([segment_name_map[i] for i in range(optimal_k)], rotation=45, ha='right')
    ax1.set_title('Segment Metrics Comparison')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "customer_segmentation_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved segmentation analysis plot")

# Additional visualization: Heatmap of segment characteristics
plt.figure(figsize=(12, 6))

# Prepare data for heatmap
heatmap_data = segment_df.set_index('Segment_Name')[['Avg_Recency', 'Avg_Frequency', 'Avg_Monetary', 'Avg_OrderValue']].T

# Normalize for better visualization
from sklearn.preprocessing import MinMaxScaler
heatmap_scaler = MinMaxScaler()
heatmap_normalized = pd.DataFrame(
    heatmap_scaler.fit_transform(heatmap_data),
    index=heatmap_data.index,
    columns=heatmap_data.columns
)

sns.heatmap(heatmap_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
            cbar_kws={'label': 'Normalized Score (0-1)'})
plt.title('Customer Segment Characteristics (Normalized)')
plt.ylabel('Metric')
plt.xlabel('Segment')
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "customer_segmentation_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved segmentation heatmap")

# ============================================================================
# STEP 7: GENERATE ACTIONABLE INSIGHTS
# ============================================================================

print(f"\n[7/7] Generating actionable insights...")

insights = []

for _, row in segment_df.iterrows():
    segment_name = row['Segment_Name']

    if segment_name == "VIP Active":
        insight = {
            'Segment': segment_name,
            'Priority': 'High',
            'Strategy': 'Retention & Rewards',
            'Actions': [
                'Offer VIP loyalty program with exclusive benefits',
                'Personalized service and early access to new products',
                'Quarterly appreciation gifts',
                'Direct account manager for top spenders'
            ],
            'Expected_ROI': 'High - Protecting high-value revenue stream'
        }
    elif segment_name == "VIP At-Risk":
        insight = {
            'Segment': segment_name,
            'Priority': 'Critical',
            'Strategy': 'Win-Back Campaign',
            'Actions': [
                'Personal outreach from account manager within 48 hours',
                '20% win-back discount on next purchase',
                'Survey to understand reasons for inactivity',
                'Exclusive re-engagement offer'
            ],
            'Expected_ROI': 'Very High - Preventing high-value churn'
        }
    elif segment_name == "Loyal Regulars":
        insight = {
            'Segment': segment_name,
            'Priority': 'Medium-High',
            'Strategy': 'Upsell & Cross-sell',
            'Actions': [
                'Product recommendations based on purchase history',
                'Bundle deals for frequently purchased items',
                'Referral program incentives',
                'Upgrade to VIP program eligibility'
            ],
            'Expected_ROI': 'High - Increase wallet share'
        }
    elif segment_name == "New Engaged":
        insight = {
            'Segment': segment_name,
            'Priority': 'Medium',
            'Strategy': 'Nurture & Convert',
            'Actions': [
                'Welcome series email campaign',
                'First-time buyer discount on second purchase',
                'Educational content about product usage',
                'Social proof and customer reviews'
            ],
            'Expected_ROI': 'Medium - Building future loyalists'
        }
    elif segment_name == "Hibernating":
        insight = {
            'Segment': segment_name,
            'Priority': 'Low-Medium',
            'Strategy': 'Re-activation',
            'Actions': [
                'Automated win-back email series',
                '15% discount incentive',
                'New product announcements',
                'Last chance offers before removal from active list'
            ],
            'Expected_ROI': 'Low-Medium - Cost-effective reactivation'
        }
    else:  # Potential Loyalists
        insight = {
            'Segment': segment_name,
            'Priority': 'Medium',
            'Strategy': 'Engagement & Frequency',
            'Actions': [
                'Frequency-based rewards program',
                'Personalized product recommendations',
                'Educational content and tips',
                'Social engagement campaigns'
            ],
            'Expected_ROI': 'Medium - Convert to loyal customers'
        }

    insights.append(insight)

print("\n" + "="*90)
print("ACTIONABLE INSIGHTS BY SEGMENT")
print("="*90)

for insight in insights:
    print(f"\n{insight['Segment']} - Priority: {insight['Priority']}")
    print(f"Strategy: {insight['Strategy']}")
    print(f"Recommended Actions:")
    for i, action in enumerate(insight['Actions'], 1):
        print(f"  {i}. {action}")
    print(f"Expected ROI: {insight['Expected_ROI']}")

# Save insights
insights_df = pd.DataFrame(insights)
insights_df.to_csv(DATA_DIR / "Customer_Segment_Insights.csv", index=False)
print(f"\n✓ Saved actionable insights to {DATA_DIR / 'Customer_Segment_Insights.csv'}")

# Save segmented customer data
df_output = df[['CustomerKey', 'Segment', 'Segment_Name'] + segmentation_features]
df_output.to_csv(DATA_DIR / "Customer_Segmentation_Results.csv", index=False)
print(f"✓ Saved customer segmentation results to {DATA_DIR / 'Customer_Segmentation_Results.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("PHASE 8: CUSTOMER SEGMENTATION - COMPLETE")
print("="*70)

print(f"\n✅ Segmented {len(df):,} customers into {optimal_k} distinct groups")
print(f"✅ Silhouette Score: {silhouette_avg:.3f}")
print(f"✅ Created {len(insights)} actionable strategies")

print(f"\nTop Segment by Revenue:")
top_segment = segment_df.loc[segment_df['Total_Revenue'].idxmax()]
print(f"  {top_segment['Segment_Name']}: ${top_segment['Total_Revenue']:,.2f} ({top_segment['Percentage']:.1f}% of customers)")

print(f"\nFiles Created:")
print(f"  1. models/customer_segmentation/kmeans_model.pkl")
print(f"  2. models/customer_segmentation/feature_scaler.pkl")
print(f"  3. data/processed/Customer_Segment_Profiles.csv")
print(f"  4. data/processed/Customer_Segment_Insights.csv")
print(f"  5. data/processed/Customer_Segmentation_Results.csv")
print(f"  6. outputs/plots/customer_segmentation_elbow.png")
print(f"  7. outputs/plots/customer_segmentation_analysis.png")
print(f"  8. outputs/plots/customer_segmentation_heatmap.png")

print("\n" + "="*70)
print("Next: Run python scripts/phase8_product_recommendations.py")
print("="*70 + "\n")
