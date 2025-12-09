import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Load the dataset
df = pd.read_csv("online_retail_II.csv", encoding="ISO-8859-1")
print("Raw dataset shape:", df.shape)

# Step 2: Basic Cleaning
df = df[df['Quantity'] > 0]            # Remove negative quantities
df = df.dropna(subset=['Invoice', 'Description'])  # Remove missing values

# Step 3: Make Invoice-Description Basket Format
basket = df.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Step 4: Apply Apriori Algorithm
frequent_items = apriori(basket, min_support=0.02, use_colnames=True)
print("\nFrequent Itemsets:")
print(frequent_items)

# Step 5: Generate Association Rules
rules = association_rules(frequent_items, metric="lift", min_threshold=1)
print("\nAssociation Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
