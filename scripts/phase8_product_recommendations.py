"""
Phase 8: Product Recommendation System
Advanced Analytics - Collaborative Filtering & Association Rules

This script creates product recommendations based on:
- Collaborative filtering (user-item matrix)
- Market basket analysis (frequent itemsets)
- Content-based filtering (product similarity)
- Cross-sell and up-sell opportunities

Output:
- Product recommendations per customer
- Frequently bought together bundles
- Cross-sell opportunities
- Up-sell suggestions

Author: AdventureWorks Data Science Team
Date: October 25, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
import joblib
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "recommendations"
OUTPUTS_DIR = BASE_DIR / "outputs" / "plots"

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PHASE 8: PRODUCT RECOMMENDATION SYSTEM")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[1/6] Loading sales data...")

# Load enriched sales data
sales_path = DATA_DIR / "AdventureWorks_Sales_Enriched.csv"
df_sales = pd.read_csv(sales_path)

print(f"✓ Loaded {len(df_sales):,} transactions")
print(f"  Unique customers: {df_sales['CustomerKey'].nunique():,}")
print(f"  Unique products: {df_sales['ProductKey'].nunique():,}")

# ============================================================================
# STEP 2: CREATE USER-ITEM MATRIX (Collaborative Filtering)
# ============================================================================

print("\n[2/6] Creating user-item interaction matrix...")

# Create purchase frequency matrix
user_item_matrix = df_sales.groupby(['CustomerKey', 'ProductKey']).size().unstack(fill_value=0)

print(f"✓ Created matrix: {user_item_matrix.shape[0]:,} customers × {user_item_matrix.shape[1]:,} products")
print(f"  Sparsity: {(1 - user_item_matrix.astype(bool).sum().sum() / (user_item_matrix.shape[0] * user_item_matrix.shape[1])):.2%}")

# Calculate item-item similarity (products that are purchased together)
item_similarity = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(
    item_similarity,
    index=user_item_matrix.columns,
    columns=user_item_matrix.columns
)

print(f"✓ Calculated item-item similarity matrix")

# Save matrices
joblib.dump(user_item_matrix, MODELS_DIR / "user_item_matrix.pkl")
joblib.dump(item_similarity_df, MODELS_DIR / "item_similarity_matrix.pkl")
print(f"✓ Saved recommendation matrices")

# ============================================================================
# STEP 3: MARKET BASKET ANALYSIS (Frequently Bought Together)
# ============================================================================

print("\n[3/6] Performing market basket analysis...")

# Group products by order
order_column = 'OrderNumber' if 'OrderNumber' in df_sales.columns else 'SalesOrderNumber'
orders = df_sales.groupby(order_column)['ProductKey'].apply(list).reset_index()
orders.columns = ['OrderID', 'Products']

print(f"✓ Analyzing {len(orders):,} orders")

# Find frequently bought together products
product_pairs = defaultdict(int)
product_counts = Counter()

for products in orders['Products']:
    # Count individual products
    for product in products:
        product_counts[product] += 1

    # Count pairs
    unique_products = list(set(products))
    if len(unique_products) >= 2:
        for i in range(len(unique_products)):
            for j in range(i+1, len(unique_products)):
                pair = tuple(sorted([unique_products[i], unique_products[j]]))
                product_pairs[pair] += 1

# Calculate support for pairs
total_orders = len(orders)
frequent_pairs = []

for (prod1, prod2), count in product_pairs.items():
    support = count / total_orders
    if support >= 0.01:  # At least 1% of orders
        # Calculate confidence: P(prod2|prod1)
        confidence_1_to_2 = count / product_counts[prod1]
        confidence_2_to_1 = count / product_counts[prod2]

        frequent_pairs.append({
            'Product1': prod1,
            'Product2': prod2,
            'Co_Occurrences': count,
            'Support': support,
            'Confidence_1_to_2': confidence_1_to_2,
            'Confidence_2_to_1': confidence_2_to_1,
            'Lift': support / ((product_counts[prod1]/total_orders) * (product_counts[prod2]/total_orders))
        })

frequent_pairs_df = pd.DataFrame(frequent_pairs).sort_values('Lift', ascending=False)

print(f"✓ Found {len(frequent_pairs_df)} frequent product pairs")
print(f"  Top pair lift: {frequent_pairs_df['Lift'].max():.2f}")

# Save frequent pairs
frequent_pairs_df.to_csv(DATA_DIR / "Frequent_Product_Pairs.csv", index=False)
print(f"✓ Saved frequent product pairs")

# ============================================================================
# STEP 4: GENERATE RECOMMENDATIONS
# ============================================================================

print("\n[4/6] Generating product recommendations...")

def get_recommendations_for_customer(customer_id, user_item_matrix, item_similarity_df, top_n=5):
    """
    Generate product recommendations for a customer based on collaborative filtering
    """
    if customer_id not in user_item_matrix.index:
        return []

    # Get products customer has already purchased
    customer_purchases = user_item_matrix.loc[customer_id]
    purchased_products = customer_purchases[customer_purchases > 0].index.tolist()

    if not purchased_products:
        return []

    # Calculate recommendation scores
    recommendation_scores = {}

    for purchased_product in purchased_products:
        # Get similar products
        similar_products = item_similarity_df[purchased_product].sort_values(ascending=False)

        # Weight by purchase frequency
        purchase_weight = customer_purchases[purchased_product]

        for product, similarity in similar_products.items():
            if product not in purchased_products:  # Don't recommend already purchased
                if product not in recommendation_scores:
                    recommendation_scores[product] = 0
                recommendation_scores[product] += similarity * purchase_weight

    # Sort and return top N
    recommendations = sorted(recommendation_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

    return [(prod, score) for prod, score in recommendations]

# Generate recommendations for sample customers
sample_customers = user_item_matrix.index[:1000]  # First 1000 customers
customer_recommendations = {}

for customer_id in sample_customers:
    recs = get_recommendations_for_customer(customer_id, user_item_matrix, item_similarity_df, top_n=5)
    if recs:
        customer_recommendations[customer_id] = recs

print(f"✓ Generated recommendations for {len(customer_recommendations):,} customers")
print(f"  Avg recommendations per customer: {np.mean([len(v) for v in customer_recommendations.values()]):.1f}")

# Save recommendations
recommendations_list = []
for customer_id, recs in customer_recommendations.items():
    for rank, (product_id, score) in enumerate(recs, 1):
        recommendations_list.append({
            'CustomerKey': customer_id,
            'RecommendedProductKey': product_id,
            'Rank': rank,
            'RecommendationScore': score
        })

recommendations_df = pd.DataFrame(recommendations_list)
recommendations_df.to_csv(DATA_DIR / "Product_Recommendations.csv", index=False)
print(f"✓ Saved {len(recommendations_df):,} product recommendations")

# ============================================================================
# STEP 5: CROSS-SELL & UP-SELL OPPORTUNITIES
# ============================================================================

print("\n[5/6] Identifying cross-sell and up-sell opportunities...")

# Initialize empty dataframes
cross_sell_df = pd.DataFrame()
up_sell_df = pd.DataFrame()

# Load product information
if 'ProductName' in df_sales.columns and 'ListPrice' in df_sales.columns:
    products_info = df_sales[['ProductKey', 'ProductName', 'ListPrice', 'Category', 'SubCategory']].drop_duplicates()

    # Cross-sell: Products from different categories bought together
    cross_sell_opportunities = []

    for _, row in frequent_pairs_df.head(50).iterrows():  # Top 50 pairs
        prod1_info = products_info[products_info['ProductKey'] == row['Product1']].iloc[0]
        prod2_info = products_info[products_info['ProductKey'] == row['Product2']].iloc[0]

        if prod1_info['Category'] != prod2_info['Category']:
            cross_sell_opportunities.append({
                'Product1_Key': row['Product1'],
                'Product1_Name': prod1_info['ProductName'],
                'Product1_Category': prod1_info['Category'],
                'Product2_Key': row['Product2'],
                'Product2_Name': prod2_info['ProductName'],
                'Product2_Category': prod2_info['Category'],
                'Support': row['Support'],
                'Lift': row['Lift'],
                'Opportunity_Type': 'Cross-Sell'
            })

    cross_sell_df = pd.DataFrame(cross_sell_opportunities)

    print(f"✓ Identified {len(cross_sell_df)} cross-sell opportunities")

    # Up-sell: Higher-priced products in same category
    up_sell_opportunities = []

    # For each product, find higher-priced alternatives in same subcategory
    for _, product in products_info.iterrows():
        similar_products = products_info[
            (products_info['SubCategory'] == product['SubCategory']) &
            (products_info['ListPrice'] > product['ListPrice']) &
            (products_info['ProductKey'] != product['ProductKey'])
        ].sort_values('ListPrice')

        if len(similar_products) > 0:
            for _, up_sell_product in similar_products.head(3).iterrows():  # Top 3 upsells
                price_increase = up_sell_product['ListPrice'] - product['ListPrice']
                price_increase_pct = (price_increase / product['ListPrice']) * 100

                up_sell_opportunities.append({
                    'Current_Product_Key': product['ProductKey'],
                    'Current_Product_Name': product['ProductName'],
                    'Current_Price': product['ListPrice'],
                    'UpSell_Product_Key': up_sell_product['ProductKey'],
                    'UpSell_Product_Name': up_sell_product['ProductName'],
                    'UpSell_Price': up_sell_product['ListPrice'],
                    'Price_Increase': price_increase,
                    'Price_Increase_Pct': price_increase_pct,
                    'Category': product['Category'],
                    'SubCategory': product['SubCategory'],
                    'Opportunity_Type': 'Up-Sell'
                })

    up_sell_df = pd.DataFrame(up_sell_opportunities)

    print(f"✓ Identified {len(up_sell_df)} up-sell opportunities")

    # Save opportunities
    cross_sell_df.to_csv(DATA_DIR / "CrossSell_Opportunities.csv", index=False)
    up_sell_df.to_csv(DATA_DIR / "UpSell_Opportunities.csv", index=False)

    print(f"✓ Saved cross-sell and up-sell opportunities")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\n[6/6] Creating recommendation visualizations...")

# Plot 1: Top product pairs by lift
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top 15 product pairs
ax = axes[0, 0]
top_pairs = frequent_pairs_df.head(15).copy()
top_pairs['Pair'] = top_pairs.apply(lambda x: f"{x['Product1']}-{x['Product2']}", axis=1)
ax.barh(range(len(top_pairs)), top_pairs['Lift'], color='steelblue')
ax.set_yticks(range(len(top_pairs)))
ax.set_yticklabels(top_pairs['Pair'], fontsize=8)
ax.set_xlabel('Lift')
ax.set_title('Top 15 Product Pairs by Lift')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# Distribution of recommendation scores
ax = axes[0, 1]
if len(recommendations_df) > 0:
    ax.hist(recommendations_df['RecommendationScore'], bins=50, color='coral', edgecolor='black')
    ax.set_xlabel('Recommendation Score')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Recommendation Scores')
    ax.grid(axis='y', alpha=0.3)

# Recommendations per customer
ax = axes[1, 0]
recs_per_customer = recommendations_df.groupby('CustomerKey').size()
ax.hist(recs_per_customer, bins=30, color='lightgreen', edgecolor='black')
ax.set_xlabel('Number of Recommendations')
ax.set_ylabel('Number of Customers')
ax.set_title('Recommendations per Customer Distribution')
ax.axvline(recs_per_customer.mean(), color='red', linestyle='--', label=f'Mean: {recs_per_customer.mean():.1f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Up-sell price increases
ax = axes[1, 1]
if len(up_sell_df) > 0:
    ax.hist(up_sell_df['Price_Increase_Pct'], bins=50, color='plum', edgecolor='black')
    ax.set_xlabel('Price Increase (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Up-Sell Price Increase Distribution')
    ax.axvline(up_sell_df['Price_Increase_Pct'].median(), color='red', linestyle='--',
               label=f'Median: {up_sell_df["Price_Increase_Pct"].median():.1f}%')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "product_recommendations_analysis.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved recommendation analysis plot")

# Plot 2: Heatmap of top product similarities
plt.figure(figsize=(12, 10))

# Get top 20 most purchased products
top_products = user_item_matrix.sum().sort_values(ascending=False).head(20).index

# Create similarity matrix for top products
top_similarity = item_similarity_df.loc[top_products, top_products]

sns.heatmap(top_similarity, cmap='RdYlGn', center=0, annot=False,
            xticklabels=top_products, yticklabels=top_products,
            cbar_kws={'label': 'Cosine Similarity'})
plt.title('Product Similarity Matrix (Top 20 Products)')
plt.xlabel('Product Key')
plt.ylabel('Product Key')
plt.tight_layout()
plt.savefig(OUTPUTS_DIR / "product_similarity_heatmap.png", dpi=300, bbox_inches='tight')
print(f"✓ Saved product similarity heatmap")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("PHASE 8: PRODUCT RECOMMENDATIONS - COMPLETE")
print("="*70)

print(f"\n✅ Created recommendation system for {df_sales['CustomerKey'].nunique():,} customers")
print(f"✅ Analyzed {df_sales['ProductKey'].nunique():,} products")
print(f"✅ Generated {len(recommendations_df):,} product recommendations")
print(f"✅ Found {len(frequent_pairs_df)} frequent product pairs")
print(f"✅ Identified {len(cross_sell_df)} cross-sell opportunities")
print(f"✅ Identified {len(up_sell_df)} up-sell opportunities")

print(f"\nTop Recommendation Insights:")
if len(frequent_pairs_df) > 0:
    top_pair = frequent_pairs_df.iloc[0]
    print(f"  Best product pair: {top_pair['Product1']} + {top_pair['Product2']}")
    print(f"  Lift: {top_pair['Lift']:.2f}x (bought together {top_pair['Lift']:.1f}x more than expected)")

print(f"\nFiles Created:")
print(f"  1. models/recommendations/user_item_matrix.pkl")
print(f"  2. models/recommendations/item_similarity_matrix.pkl")
print(f"  3. data/processed/Frequent_Product_Pairs.csv")
print(f"  4. data/processed/Product_Recommendations.csv")
print(f"  5. data/processed/CrossSell_Opportunities.csv")
print(f"  6. data/processed/UpSell_Opportunities.csv")
print(f"  7. outputs/plots/product_recommendations_analysis.png")
print(f"  8. outputs/plots/product_similarity_heatmap.png")

print("\n" + "="*70)
print("Next: Run python scripts/phase8_anomaly_detection.py")
print("="*70 + "\n")
