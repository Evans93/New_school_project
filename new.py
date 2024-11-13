import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

#load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ["mpg", "cylinders", "displacement", "horsepower", "weighgt","acceleration", "model_year", "origin"]
data = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep= " ", skipinitialspace=True)

# handle missing values
data.dropna(inplace=True)
#this code drops aka deletes the missing values 

#one-hot encode 'origin' feature
data = pd.get_dummies(data, columns=['origin'], drop_first=True)

# define features and target
x = data.drop('mpg', axis=1)
y = data['mpg']

#split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# initialize and train model
model = LinearRegression()
model.fit(x_train, y_train)

#predictions on test set
predictions= model.predict(x_test)

#calculate metrics
mse= mean_squared_error (y_test, predictions)
r2= r2_score(y_test, predictions)

# display results
print("mean squared error:" ,mse)
print ("R-squared:" ,r2)
print("predictions:",predictions[:5])