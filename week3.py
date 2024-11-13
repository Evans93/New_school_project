import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split

auto_mpg = fetch_ucirepo(id = 9)

X = auto_mpg.data.features # rest
Y = auto_mpg.data.targets # mpg

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = HistGradientBoostingRegressor()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

mse = mean_squared_error(Y_test, predictions)
r2 = r2_score(Y_test, predictions)

print("mse: ", mse)
print("r-squared: ", r2)
print("predictions: ", predictions)
print("score: ", model.score(X_test, Y_test))