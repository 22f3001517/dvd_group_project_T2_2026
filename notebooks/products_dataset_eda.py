"""Products_dataset_EDA

Original file is located at
    https://colab.research.google.com/drive/1TsFUx5e6nxGpVtIkxIuzRHMqaRohGpH6
"""

# ===== CELL 1: Upload the dataset =====
from google.colab import files
uploaded = files.upload()
print(uploaded.keys())

# ===== CELL 2: Unzip =====
import zipfile
import os

zip_filename = "E-Commerce Dataset -20260819T104554Z-1-001.zip"

with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('olist_data')

for root, dirs, filenames in os.walk('olist_data'):
    for f in filenames:
        print(os.path.join(root, f))

# ===== CELL 3: Load products dataset =====
import pandas as pd

products = pd.read_csv('olist_data/E-Commerce Dataset /products_dataset.csv')

print(products.shape)
print(products.columns.tolist())
products.head()

# ===== CELL 4: Also load translation table (needed later for merging) =====
translation = pd.read_csv('olist_data/E-Commerce Dataset /product_category_name_translation.csv')
translation.head()

# ===== CELL 5: Missing Values Check =====
missing = products.isnull().sum()
missing_pct = (missing / len(products) * 100).round(2)
missing_summary = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
print(missing_summary[missing_summary['missing_count'] > 0])

print("\nTotal rows:", len(products))
print("Duplicate product_ids:", products['product_id'].duplicated().sum())

# ===== CELL 6: Clean =====
# Drop rows with no category at all (usually a small number)
products_clean = products.dropna(subset=['product_category_name']).copy()

# Impute dimension/weight columns with median
dim_cols = ['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm']
for col in dim_cols:
    products_clean[col] = products_clean[col].fillna(products_clean[col].median())

# Impute description length / photo qty
products_clean['product_description_lenght'] = products_clean['product_description_lenght'].fillna(
    products_clean['product_description_lenght'].median())
products_clean['product_photos_qty'] = products_clean['product_photos_qty'].fillna(0)

print("Rows before:", len(products), "| Rows after cleaning:", len(products_clean))
products_clean.isnull().sum()

# ===== CELL 7: Merge with translation table =====
products_clean = products_clean.merge(
    translation, on='product_category_name', how='left'
)

untranslated = products_clean['product_category_name_english'].isnull().sum()
print("Untranslated categories:", untranslated)

products_clean['product_category_name_english'] = products_clean['product_category_name_english'].fillna(
    products_clean['product_category_name']
)

products_clean.head()

# ===== CELL 7a: Check which categories were untranslated =====
untranslated_cats = products_clean.loc[
    ~products_clean['product_category_name'].isin(translation['product_category_name']),
    'product_category_name'
].unique()

print("Categories with no English translation:")
print(untranslated_cats)

# ===== CELL 7b: Manually add translations for the 2 missing categories =====
manual_translations = {
    'pc_gamer': 'pc_gamer',
    'portateis_cozinha_e_preparadores_de_alimentos': 'kitchen_portables_and_food_preparers'
}

products_clean['product_category_name_english'] = products_clean.apply(
    lambda row: manual_translations.get(row['product_category_name'], row['product_category_name_english']),
    axis=1
)

still_missing = products_clean.loc[
    products_clean['product_category_name'].isin(manual_translations.keys()),
    ['product_category_name', 'product_category_name_english']
].drop_duplicates()

print(still_missing)

# ===== CELL 8: Load order_items and orders =====
order_items = pd.read_csv('olist_data/E-Commerce Dataset /order_items_dataset.csv')
orders = pd.read_csv('olist_data/E-Commerce Dataset /orders_dataset.csv')

print(order_items.shape)
print(orders.shape)

# ===== CELL 9: Merge products -> order_items -> orders =====
order_items_orders = order_items.merge(
    orders[['order_id', 'order_status', 'order_purchase_timestamp']],
    on='order_id', how='left'
)

products_sales = products_clean.merge(order_items_orders, on='product_id', how='left')

print(products_sales.shape)
products_sales.head()

# ===== CELL 10: Sanity check the merge =====
no_sales = products_sales['order_id'].isnull().sum()
print(f"Products with no matching order records: {no_sales}")
print(f"Total rows after merge: {len(products_sales)}")

# ===== CELL 11: Category-level summary =====
category_summary = (products_sales.groupby('product_category_name_english')
                     .agg(total_orders=('order_id', 'nunique'),
                          total_units=('order_item_id', 'count'),
                          total_revenue=('price', 'sum'),
                          avg_price=('price', 'mean'))
                     .sort_values('total_revenue', ascending=False))

print(category_summary.head(10))
print(f"\nTotal categories: {len(category_summary)}")

# ===== CELL 12: Chart 1 — Top 10 categories by revenue =====
import matplotlib.pyplot as plt
import seaborn as sns

top10 = category_summary.head(10).reset_index()

plt.figure(figsize=(10,6))
sns.barplot(data=top10, y='product_category_name_english', x='total_revenue', palette='viridis')
plt.title('Top 10 Product Categories by Revenue')
plt.xlabel('Total Revenue (R$)')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig('top_categories_revenue.png', dpi=150)
plt.show()

# ===== CELL 13: Revenue concentration =====
total_revenue_all = category_summary['total_revenue'].sum()
top10_revenue = category_summary.head(10)['total_revenue'].sum()
top10_share = (top10_revenue / total_revenue_all * 100).round(1)

print(f"Total revenue across all categories: R$ {total_revenue_all:,.2f}")
print(f"Top 10 categories revenue: R$ {top10_revenue:,.2f}")
print(f"Top 10 share of total revenue: {top10_share}%")

# ===== CELL 14 (fixed): Chart 2 — Price distribution, top 5 categories =====
top5_cats = category_summary.head(5).index.tolist()
subset = products_sales[products_sales['product_category_name_english'].isin(top5_cats)]

plt.figure(figsize=(10,6))
sns.boxplot(data=subset, y='product_category_name_english', x='price',
            hue='product_category_name_english', palette='Set2', legend=False)
plt.title('Price Distribution — Top 5 Categories by Revenue')
plt.xlabel('Price (R$)')
plt.ylabel('')
plt.xlim(0, subset['price'].quantile(0.95))
plt.tight_layout()
plt.savefig('price_distribution_top5.png', dpi=150)
plt.show()

# ===== CELL 15: Chart 3 — Weight vs Freight Value =====
merged_freight = products_sales.dropna(subset=['freight_value', 'product_weight_g'])

plt.figure(figsize=(8,6))
sns.scatterplot(data=merged_freight.sample(min(3000, len(merged_freight)), random_state=42),
                 x='product_weight_g', y='freight_value', alpha=0.4)
plt.title('Product Weight vs Freight Value')
plt.xlabel('Weight (g)')
plt.ylabel('Freight Value (R$)')
plt.tight_layout()
plt.savefig('weight_vs_freight.png', dpi=150)
plt.show()

# Correlation for your write-up
corr = merged_freight['product_weight_g'].corr(merged_freight['freight_value'])
print(f"Correlation between weight and freight value: {corr:.3f}")

# ===== CELL 16: Chart 4 — Number of products per category (catalog size vs revenue) =====
product_count_per_cat = products_clean['product_category_name_english'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(x=product_count_per_cat.values, y=product_count_per_cat.index, palette='mako')
plt.title('Top 10 Categories by Number of Products Listed')
plt.xlabel('Number of Products')
plt.ylabel('Category')
plt.tight_layout()
plt.savefig('products_per_category.png', dpi=150)
plt.show()

# ===== CELL 17: Chart 6 — Review score by category (requires reviews data) =====
reviews = pd.read_csv('olist_data/E-Commerce Dataset /order_reviews_dataset.csv')

products_reviews = products_sales.merge(
    reviews[['order_id', 'review_score']], on='order_id', how='left'
)

review_by_cat = (products_reviews.groupby('product_category_name_english')['review_score']
                  .mean()
                  .sort_values(ascending=False))

top_bottom = pd.concat([review_by_cat.head(5), review_by_cat.tail(5)])

plt.figure(figsize=(10,6))
colors = ['#2ca02c']*5 + ['#d62728']*5
sns.barplot(x=top_bottom.values, y=top_bottom.index, palette=colors)
plt.title('Highest vs Lowest Rated Categories (Avg Review Score)')
plt.xlabel('Average Review Score')
plt.ylabel('Category')
plt.xlim(0, 5)
plt.tight_layout()
plt.savefig('review_score_by_category.png', dpi=150)
plt.show()
