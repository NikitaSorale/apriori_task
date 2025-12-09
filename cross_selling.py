import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load dataset
df = pd.read_csv("online_retail_II.csv", encoding="ISO-8859-1")

# Step 2: Clean column names and data
df.columns = df.columns.str.strip()
df = df[df['Quantity'] > 0]
df = df.dropna(subset=['Invoice', 'Description'])

# Step 3: Create Invoice-Product basket (one-hot encoding)
basket = df.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Step 4: Apply Apriori algorithm to find frequent itemsets
frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
print("Frequent Itemsets:")
print(frequent_items.sort_values('support', ascending=False).head(10))

# Step 5: Generate association rules
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
rules = rules.sort_values('lift', ascending=False)
print("\nTop 10 Cross-selling Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Step 6: Optional - Filter rules for practical cross-selling
# Example: Only rules with 2+ items in antecedents
cross_sell_rules = rules[rules['antecedents'].apply(lambda x: len(x) >= 1)]
print("\nFiltered Cross-selling Rules:")
print(cross_sell_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))
