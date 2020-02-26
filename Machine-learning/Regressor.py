# import sys
import numpy as np

filename = 'data_multivar_1.txt'
X = []
y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        temp = [float(i) for i in line.split(',')]
        X.append(temp[:-1])
        y.append(temp[-1])

num_training = int(0.8 * len(X))
num_test = len(X) - num_training
#___ Training data
X_train = np.array(X[:num_training]).reshape((num_training, 3))
y_train = np.array(y[:num_training])
#___ Test data
X_test = np.array(X[num_training:]).reshape((num_test,3))
y_test = np.array(y[num_training:])

from sklearn import linear_model
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)
y_train_pred = linear_regressor.predict(X_train)
y_test_pred = linear_regressor.predict(X_test)

# # Plotting
# import matplotlib.pyplot as plt
# plt.figure()
# plt.scatter(X_train, y_train, color='green')
# plt.plot(X_train, y_train_pred, color='black', linewidth=4)
# plt.title('Training data')
# plt.show()
#

# plt.scatter(X_test, y_test, color='red')
# plt.plot(X_test, y_test_pred, color='black', linewidth=4)
# plt.title('Test data')
# plt.show()

# Computing regression accuracy
import sklearn.metrics as sm
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score = ", round(sm.r2_score(y_test, y_test_pred), 2))

# Achieving model persistence
import pickle
output_model_file = 'saved_model.pk1'
with open(output_model_file, 'wb') as f:
    pickle.dump(linear_regressor, f)
with open(output_model_file, 'rb') as f:
    model_linregr = pickle.load(f)

y_test_pred_new = model_linregr.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))

# Building ridge regressor
ridge_regressor = linear_model.Ridge(alpha=0.01, fit_intercept=True, max_iter=10000)
ridge_regressor.fit(X_train, y_train)
y_test_pred_ridge = ridge_regressor.predict(X_test)
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_ridge), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred_ridge), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred_ridge), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred_ridge), 2))
print("R2 score = ", round(sm.r2_score(y_test, y_test_pred_ridge), 2))

# Building polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[0.39, 2.78, 7.11]]
poly_datapoint = polynomial.fit_transform(datapoint)

poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("\nLinear regression:", linear_regressor.predict(datapoint)[0])
print("\nPolynomial regression:", poly_linear_model.predict(poly_datapoint)[0])