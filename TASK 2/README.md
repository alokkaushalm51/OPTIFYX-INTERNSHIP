# Car Price Prediction

This repository contains a Jupyter Notebook for predicting car prices using various machine learning models. The dataset used is `car data.csv`, which includes features such as car name, year, selling price, present price, kilometers driven, fuel type, selling type, transmission, and owner information.

## Introduction
The goal of this project is to predict the selling price of cars based on various features. We use several machine learning models, including Linear Regression, Random Forest, and Gradient Boosting, to achieve this. The models are evaluated using metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R²).

## Requirements
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Dataset
The dataset `car data.csv` contains the following columns:
- `Car_Name`: Name of the car
- `Year`: Year of purchase
- `Selling_Price`: Selling price of the car
- `Present_Price`: Present price of the car
- `Driven_kms`: Kilometers driven
- `Fuel_Type`: Type of fuel (Petrol, Diesel, CNG)
- `Selling_type`: Type of selling (Dealer, Individual)
- `Transmission`: Transmission type (Manual, Automatic)
- `Owner`: Number of previous owners

## Data Preprocessing
The data preprocessing steps include:
- Handling missing values
- Encoding categorical variables
- Creating new features such as the age of the car
- Splitting the data into training and testing sets

## Model Training
We train three different models:
1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**

The models are trained on the training dataset and evaluated on the testing dataset.

## Model Evaluation
The performance of the models is evaluated using the following metrics:
- **Mean Squared Error (MSE)**
- **Mean Absolute Error (MAE)**
- **R-squared (R²)**

## Feature Importance
We analyze the importance of each feature in predicting the car price using the Random Forest model. The feature importance plot helps us understand which features contribute the most to the model's predictions.

## Visualization
We visualize the predictions of the Random Forest model against the actual prices using a scatter plot. The plot shows a strong correlation between the predicted and actual prices, indicating the model's accuracy.

## Conclusion
The Random Forest model performs the best among the three models, with the lowest MSE and MAE and the highest R² value. This model can be used to predict car prices with high accuracy based on the given features.

