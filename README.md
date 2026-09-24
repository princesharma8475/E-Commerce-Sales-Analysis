# E-Commerce Sales Data Analysis and Business Insights Using Python

## 1. Project Overview

This project analyzes e-commerce sales data using Python to identify sales trends, customer behavior, product performance, regional performance, and business insights.

The project also applies Machine Learning techniques to predict sales and compare model performance.

## 2. Objectives

- Analyze e-commerce sales data.
- Clean and preprocess the dataset.
- Identify top-performing products and categories.
- Analyze regional and monthly sales trends.
- Analyze payment modes and order status.
- Create meaningful data visualizations.
- Apply Machine Learning for sales prediction.
- Evaluate and improve the prediction model.

## 3. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## 4. Dataset

The dataset contains 5,000 cleaned e-commerce transactions and 19 attributes including:

- Order details
- Customer information
- Product information
- Sales
- Profit
- Quantity
- Discount
- Payment mode
- Order status
- Rating

## 5. Data Cleaning

The following data preprocessing steps were performed:

- Removed duplicate records.
- Handled missing ratings using the mean rating.
- Replaced missing discount values with 0.
- Replaced missing city values with "Unknown".
- Converted Order_Date into datetime format.

After cleaning:

- Records: 5,000
- Missing values: 0
- Duplicate rows: 0

## 6. Exploratory Data Analysis

The project analyzed:

- Sales by category
- Profit by category
- Top 10 products
- Sales by region
- Monthly sales
- Payment mode
- Order status

## 7. Key Business Results

Total Sales: ₹94,666,866.75

Total Profit: ₹13,115,166.84

Total Orders: 5,000

Total Quantity Sold: 9,256

The Electronics category generated the highest sales.

Laptop was the highest-selling product by sales value.

North region recorded the highest regional sales.

UPI was the highest sales-generating payment mode.

## 8. Data Visualization

The project created visualizations for:

- Sales by Category
- Monthly Sales Trend
- Sales by Region
- Top 10 Products
- Sales by Payment Mode
- Sales vs Profit

## 9. Machine Learning

Two regression approaches were implemented:

### Linear Regression

Mean Absolute Error: 781,941.65

R² Score: -1.08

### Random Forest Regression

Mean Absolute Error: 787.14

R² Score: 1.00

The Random Forest model provided a much stronger fit on the held-out test data for this dataset.

## 10. Example Prediction

The Random Forest model predicted sales of approximately:

₹9,141.41

for an example order with selected input features.

## 11. Conclusion

The project demonstrates how Python and Machine Learning can be used to analyze e-commerce sales data and generate business insights.

The analysis identifies important sales patterns, product performance, regional performance, payment preferences, and profitability.

Machine Learning was also applied to demonstrate sales prediction using customer and transaction-related features.

## 12. Future Scope

- Build an interactive dashboard using Power BI or Streamlit.
- Use larger real-world datasets.
- Apply advanced time-series forecasting.
- Add customer segmentation.
- Develop a real-time sales prediction system.