import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data
diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_predict = model.predict(diabetes_X_test)

print(f"Mean squared error: {mean_squared_error(diabetes_y_test, diabetes_y_predict)}")

print(f"Weight: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
# plt.plot(diabetes_X_test, diabetes_y_predict, color='blue', linewidth=3)
#
# plt.show()
