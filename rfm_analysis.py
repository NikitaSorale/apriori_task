import pandas as pd
import datetime as dt

# Step 1: Load the dataset
df = pd.read_csv("online_retail_II.csv", encoding="ISO-8859-1")

# Step 2: Clean column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Step 3: Check columns
print("Columns in dataset:", df.columns.tolist())

# Step 4: Basic cleaning
df = df[df['Quantity'] > 0]                       # Remove negative quantities

# Drop missing CustomerID and Invoice
if 'CustomerID' in df.columns:
    df = df.dropna(subset=['Invoice', 'CustomerID'])
else:
    raise KeyError("Column 'CustomerID' not found in dataset")

# Step 5: Calculate TotalPrice using the correct price column
if 'UnitPrice' in df.columns:
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
else:
    raise KeyError("Column 'UnitPrice' not found in dataset")

# Step 6: Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Step 7: Define snapshot date for Recency
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# Step 8: Calculate RFM metrics
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalPrice': 'sum'
})

# Step 9: Rename columns
rfm.rename(columns={
    'InvoiceDate': 'Recency',
    'Invoice': 'Frequency',
    'TotalPrice': 'Monetary'
}, inplace=True)

# Step 10: Display top 10 customers by Monetary value
print("\nTop 10 customers by Monetary value:")
print(rfm.sort_values('Monetary', ascending=False).head(10))
