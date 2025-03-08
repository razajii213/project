# project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV Data (Provide your CSV path here)
data = pd.read_csv("sales_data.csv")

# Clean Column Names (Removing extra spaces if any)
data.columns = data.columns.str.strip()

# Check if 'Sales' column exists
if 'Sales' not in data.columns:
    print("'Sales' column not found. Creating it using 'Quantity' * 'Price'...")
    data['Sales'] = data['Quantity'] * data['Price']
else:
    print("'Sales' column found!")

# Handle Missing Values
data.fillna(0, inplace=True)

# Data Overview
print("Data Info:")
print(data.info())
print("\nData Sample:")
print(data.head())

# Feature Engineering: Convert Date to Numeric Format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['DateNumeric'] = data['Date'].dt.day

# Remove rows with invalid date formats
data = data.dropna(subset=['Date'])

# Features and Target
features = ['DateNumeric', 'Month', 'Year', 'Quantity', 'Price']
target = 'Sales'

X = data[features]
y = data[target]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model Training
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Model Performance
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Feature Importance
importance = model.feature_importances_
print("\nFeature Importance:")
for i, v in enumerate(importance):
    print(f"{features[i]}: {v:.2f}")

# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nCross-Validation R² Scores:", cv_scores)
print("Average R² Score:", np.mean(cv_scores))

# Hyperparameter Tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='r2')
grid_search.fit(X, y)

print("\nBest Parameters:", grid_search.best_params_)
print("Best R² Score from Grid Search:", grid_search.best_score_)

# Final Model Training with Best Parameters
final_model = RandomForestRegressor(**grid_search.best_params_, random_state=42)
final_model.fit(X_train, y_train)

y_final_pred = final_model.predict(X_test)
final_mse = mean_squared_error(y_test, y_final_pred)
final_r2 = r2_score(y_test, y_final_pred)
print(f"\nFinal Model Mean Squared Error: {final_mse:.2f}")
print(f"Final Model R² Score: {final_r2:.2f}")

# Data Visualization
sns.set(style='whitegrid')

# Sales Trend Over Time
plt.figure(figsize=(10, 6))
data.groupby('Date')['Sales'].sum().plot(marker='o', color='skyblue')
plt.title('Sales Trend Over Time')
plt.xlabel('Date')
plt.ylabel('Total Sales')
plt.show()

# Top 5 Products by Sales
top_products = data.groupby('Product')['Sales'].sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_products.values, y=top_products.index, palette='viridis')
plt.title('Top 5 Products by Sales')
plt.xlabel('Total Sales')
plt.show()

# Sales by Category
if 'Category' in data.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='Category', y='Sales', data=data, palette='Set2')
    plt.title('Sales Distribution by Category')
    plt.xlabel('Category')
    plt.ylabel('Sales')
    plt.show()
else:
    print("'Category' column not found. Skipping Sales by Category plot.")

# Region-wise Sales Distribution
if 'Region' in data.columns:
    plt.figure(figsize=(8, 5))
    sns.countplot(x='Region', data=data, palette='coolwarm')
    plt.title('Region-wise Sales Distribution')
    plt.xlabel('Region')
    plt.ylabel('Count of Sales')
    plt.show()
else:
    print("'Region' column not found. Skipping Region-wise Sales Distribution plot.")

# Correlation Heatmap (Only for Numeric Data)
numeric_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_data.corr(), annot=True, cmap='YlGnBu', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()
