import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

## load your dataset(replace "synthetic_healthcare_data.csv")
data = pd.read_csv("Healthcare.csv")


## Assuming "readmission" is the target column and the rest are features

x = data.drop("readmission", axis=1) ## features
y = data["readmission"] ## Target variables


## split the data into training and the testing sets (80% train, 20% test)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

## standardize the data (important for Ridge regression)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


## Initialize the Ridge regression model

Ridge_regressor = Ridge(alpha=1.0) ## alpha is the regularization parameter

## Train the model
Ridge_regressor.fit(x_train_scaled, y_train)

## predict on the test set 
y_pred = Ridge_regressor.predict(x_test_scaled)

## evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')




# 1. Scatter plot of predicted vs actual values
plt. figure(figsize=(8, 6))
plt. scatter (y_test, y_pred, color='blue', edgecolor='k', alpha=0.7)
plt. title( 'Predicted vs Actual Values')
plt. xlabel( 'Actual Values (Readmission) ')
plt. ylabel( 'Predicted Values')
plt. grid (True)
plt. show


# 2. Histogram of residuals (errors)
residuals = y_test - y_pred
plt. figure(figsize=(8, 6))
sns. histplot(residuals, kde=True, color='red', bins=20)
plt. title( 'Residuals Distribution')
plt. xlabel('|Residuals')
plt. ylabel ('Frequency')
plt. grid (True)
plt. show()


# 3. Plot the coefficients (importance of features)
coefficients = ridge_regressor.coef_
features = X. columns
plt. figure(figsize=(8, 6))
plt. barh(features, coefficients, color='green')
plt. title('Feature Coefficients (Ridge Regression)')
plt. xlabel( 'Coefficient Value')
plt. ylabel ( 'Features')
plt. grid (True)
plt. show()
