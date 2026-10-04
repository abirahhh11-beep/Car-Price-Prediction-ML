# Car Price Prediction using Machine Learning

## Project Overview

This project predicts the selling price of used cars using Machine Learning.

The dataset contains information such as year, present price, driven kilometres, fuel type, selling type, transmission, and number of previous owners.

A Linear Regression model is trained to predict the car's selling price based on these features.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

## Dataset

The project uses a used-car dataset containing 301 records and 9 columns.

The target variable is:

* `Selling_Price`

## Data Preprocessing

The following preprocessing steps were performed:

* Checked the dataset structure and shape
* Checked missing values
* Checked duplicate records
* Removed duplicate rows
* Converted categorical variables into numerical values using one-hot encoding
* Removed `Car_Name` because it is a text identifier
* Created a new `Car_Age` feature

## Machine Learning Model

**Linear Regression** was used to predict the selling price of cars.

The dataset was divided into:

* 80% Training Data
* 20% Testing Data

## Model Evaluation

The model was evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* R-squared (R²)

Test-set results:

* **MAE:** 1.47
* **RMSE:** 2.52
* **R²:** 0.753

The model explained approximately 75.3% of the variation in the test-set selling prices.

## Visualizations

The project includes visualizations for:

* Present Price vs Selling Price
* Year vs Selling Price
* Driven Kms vs Selling Price
* Actual vs Predicted Selling Price
* Correlation Heatmap

## New Car Price Prediction

The trained model was also used to estimate the selling price of a new car based on its input features.

## Business Insights

* Present Price is an important factor in predicting Selling Price.
* Older vehicles generally have lower selling prices.
* Driven kilometres can affect resale value.
* Fuel type, selling type, transmission, ownership and vehicle age can influence price.
* Machine Learning can be used as a basic tool for estimating used-car selling prices.

## Project Files

* `car price prediction.py` – Python source code
* `car data.csv` – Dataset
* `README.md` – Project documentation

## Conclusion

This project demonstrates a complete Machine Learning workflow for car price prediction, including data preprocessing, feature engineering, model training, evaluation, visualization, and price prediction.
