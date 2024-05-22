Tips
# Using sklearn
regression = linear_model.LinearRegression()
regression.fit(x, y)
predictions_sklearn = regression.predict(x)
print("Intercept: \n", regression.intercept_)
print("Coefficients: \n", regression.coef_)
