import pandas as pd

# Load the dataset
df = pd.read_csv("car data.csv")

# Display first 5 rows
print("First 5 rows:")
print(df.head())

# Display dataset shape
print("\nDataset Shape:")
print(df.shape)

# Display column names
print("\nColumn Names:")
print(df.columns)

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Statistical summary
print("\nStatistical Summary:")
print(df.describe())

# Remove duplicate rows
df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)

# Convert categorical columns into numerical values
df = pd.get_dummies(
    df,
    columns=["Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True
)

print("\nColumns after encoding:")
print(df.columns)

print("\nFirst 5 rows after encoding:")
print(df.head())

# Remove Car_Name because it is a text identifier
df = df.drop("Car_Name", axis=1)

print("\nFinal columns after preprocessing:")
print(df.columns)

print("\nFinal dataset shape:")
print(df.shape)

# Feature Engineering: Calculate car age
current_year = 2026
df["Car_Age"] = current_year - df["Year"]

print("\nDataset after Feature Engineering:")
print(df.head())

print("\nColumns after Feature Engineering:")
print(df.columns)

# Separate features and target
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining and Testing Data:")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
model = LinearRegression()

# Train the model using training data
model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully!")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict car prices for test data
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R²):", r2)

import matplotlib.pyplot as plt

# Present Price vs Selling Price
plt.figure(figsize=(8, 5))
plt.scatter(df["Present_Price"], df["Selling_Price"])
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.title("Present Price vs Selling Price")
plt.show()

# Year vs Selling Price
plt.figure(figsize=(8, 5))
plt.scatter(df["Year"], df["Selling_Price"])
plt.xlabel("Year")
plt.ylabel("Selling Price")
plt.title("Year vs Selling Price")
plt.show()

# Driven Kms vs Selling Price
plt.figure(figsize=(8, 5))
plt.scatter(df["Driven_kms"], df["Selling_Price"])
plt.xlabel("Driven Kms")
plt.ylabel("Selling Price")
plt.title("Driven Kms vs Selling Price")
plt.show()

# Actual vs Predicted Selling Price
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Selling Price")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 7))

correlation = df.corr(numeric_only=True)

plt.imshow(correlation, cmap="coolwarm")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# Predict Selling Price for a New Car

new_car = pd.DataFrame({
    "Year": [2018],
    "Present_Price": [10.0],
    "Driven_kms": [20000],
    "Owner": [0],
    "Fuel_Type_Diesel": [0],
    "Fuel_Type_Petrol": [1],
    "Selling_type_Individual": [0],
    "Transmission_Manual": [1],
    "Car_Age": [2026 - 2018]
})

# Arrange new car columns in the same order used during model training
new_car = new_car[X.columns]

# Predict the selling price
predicted_price = model.predict(new_car)

print("\nPredicted Selling Price for New Car:")
print(predicted_price[0])

# Actionable Business Insights

print("\nActionable Business Insights:")
print("1. Present Price is an important factor in predicting Selling Price.")
print("2. Older vehicles generally have lower selling prices.")
print("3. Driven Kms can affect the resale value of a vehicle.")
print("4. Vehicle age, usage, fuel type, selling type and transmission can influence price.")
print("5. The trained model can be used to estimate the selling price of a new vehicle.")

# Final Project Summary

print("\nFinal Project Summary:")
print("Car Price Prediction using Machine Learning was completed successfully.")
print("Linear Regression was used to predict vehicle selling prices.")
print("The model was evaluated using MAE, RMSE and R-squared metrics.")
print("The model can be used as a basic tool for estimating used vehicle prices.")