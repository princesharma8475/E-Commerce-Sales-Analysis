import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("ecommerce_sales.csv")

# Data Cleaning
df.drop_duplicates(inplace=True)

df["Rating"] = df["Rating"].fillna(df["Rating"].mean())
df["Discount_Percent"] = df["Discount_Percent"].fillna(0)
df["City"] = df["City"].fillna("Unknown")

df["Order_Date"] = pd.to_datetime(df["Order_Date"])

# Dataset Information
print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

# Basic Statistics
print("\nBasic Statistics:")
print(df.describe())

# Key Business Metrics
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order_ID"].nunique()
total_quantity = df["Quantity"].sum()

print("\nTotal Sales:", round(total_sales, 2))
print("Total Profit:", round(total_profit, 2))
print("Total Orders:", total_orders)
print("Total Quantity Sold:", total_quantity)

# Category Sales
category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
print("\nSales by Category:")
print(category_sales)

# Category Profit
category_profit = df.groupby("Category")["Profit"].sum().sort_values(ascending=False)
print("\nProfit by Category:")
print(category_profit)

# Top Products
top_products = df.groupby("Product")["Sales"].sum().sort_values(ascending=False).head(10)
print("\nTop 10 Products:")
print(top_products)

# Regional Sales
region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
print("\nSales by Region:")
print(region_sales)

# Monthly Sales
df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

monthly_sales = df.groupby("Month")["Sales"].sum()

print("\nMonthly Sales:")
print(monthly_sales)

# Payment Mode Analysis
payment_sales = df.groupby("Payment_Mode")["Sales"].sum().sort_values(ascending=False)

print("\nSales by Payment Mode:")
print(payment_sales)

# Order Status
order_status = df["Order_Status"].value_counts()

print("\nOrder Status:")
print(order_status)


# =========================
# VISUALIZATION
# =========================

# 1. Sales by Category
plt.figure(figsize=(8, 5))
category_sales.plot(kind="bar")
plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("charts/chart.png")
plt.close()

# 2. Monthly Sales Trend
plt.figure(figsize=(12, 5))
monthly_sales.plot(kind="line", marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("charts/monthly_sales.png")
plt.close()


# 3. Sales by Region
plt.figure(figsize=(8, 5))
region_sales.plot(kind="bar")
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig("charts/sales_by_region.png")
plt.close()


# 4. Top 10 Products
plt.figure(figsize=(10, 5))
top_products.sort_values().plot(kind="barh")
plt.title("Top 10 Products by Sales")
plt.xlabel("Sales")
plt.ylabel("Product")
plt.tight_layout()
plt.savefig("charts/top_10_products.png")
plt.close()


# 5. Payment Mode
plt.figure(figsize=(8, 5))
payment_sales.plot(kind="pie", autopct="%1.1f%%")
plt.title("Sales by Payment Mode")
plt.ylabel("")
plt.tight_layout()
plt.savefig("charts/payment_mode.png")
plt.close()


# 6. Sales vs Profit
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="Sales", y="Profit")
plt.title("Sales vs Profit")
plt.tight_layout()
plt.savefig("charts/sales_vs_profit.png")
plt.close()


print("\n====================================")
print("E-COMMERCE SALES ANALYSIS COMPLETED")
print("====================================")

# ==========================================
# AI / ML - SALES PREDICTION
# ==========================================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create features
df["Year"] = df["Order_Date"].dt.year
df["Month_Number"] = df["Order_Date"].dt.month

# Monthly sales dataset
monthly_data = df.groupby(
    ["Year", "Month_Number"]
)["Sales"].sum().reset_index()

# Features and target
X = monthly_data[["Year", "Month_Number"]]
y = monthly_data["Sales"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LinearRegression()

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n====================================")
print("AI SALES PREDICTION")
print("====================================")

print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Predict next month
last_year = monthly_data["Year"].max()
last_month = monthly_data[
    monthly_data["Year"] == last_year
]["Month_Number"].max()

if last_month == 12:
    next_year = last_year + 1
    next_month = 1
else:
    next_year = last_year
    next_month = last_month + 1

next_month_prediction = model.predict(
    pd.DataFrame({
        "Year": [next_year],
        "Month_Number": [next_month]
    })
)

print(
    "Predicted Sales for Next Month:",
    round(next_month_prediction[0], 2)
)

print("\n====================================")
print("AI SALES PREDICTION COMPLETED")
print("====================================")

# ==========================================
# IMPROVED AI / ML - SALES PREDICTION
# ==========================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Features
features = [
    "Quantity",
    "Unit_Price",
    "Discount_Percent",
    "Age",
    "Rating"
]

X = df[features]
y = df["Sales"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Random Forest Model
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

# Train Model
rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)

# Evaluation
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

print("\n====================================")
print("IMPROVED AI SALES PREDICTION")
print("====================================")

print("Mean Absolute Error:", round(rf_mae, 2))
print("R2 Score:", round(rf_r2, 2))

# Example future sales prediction
future_data = pd.DataFrame({
    "Quantity": [2],
    "Unit_Price": [5000],
    "Discount_Percent": [10],
    "Age": [30],
    "Rating": [4.0]
})

future_prediction = rf_model.predict(future_data)

print(
    "Predicted Sales for Example Order:",
    round(future_prediction[0], 2)
)

print("\n====================================")
print("IMPROVED AI MODEL COMPLETED")
print("====================================")