import numpy as np
import pandas as pd
from pandas import DataFrame
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import textwrap
import pylab

# Define dataset loading function
def data_load(filename):
    file = csv.reader(open(filename, 'rt'), delimiter=',')
    data = []
    # Extract X, y
    X, y = [], []
    for row in file:
        data.append(row)
    # data1 = preprocessing.normalize(data[1:], norm='l1')
    data_scaler = preprocessing.MinMaxScaler(feature_range=(0,10))
    data_scaled = data_scaler.fit_transform(data[1:])
    for row in data_scaled:
        X.append(row[0:11])
        y.append(row[-1])
    # Separate features' names
    labels = np.array(data[0])
    feature_names = labels[0:11]
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names

# Input and normalize data
X, y, feature_names = data_load('winequality-red.csv')

# # Normalize features
# X = preprocessing.normalize(X, norm='l1')

# Separate training and testing data
num_training = int(0.7 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Train the random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=100, min_impurity_decrease=0)
rf_regressor.fit(X_train, y_train)

# # Train multivariate linear regressor
# ln_regressor = linear_model.LinearRegression()
# ln_regressor.fit(X_train, y_train)


# Evaluate regressor's performance
y_pred = rf_regressor.predict(X_test)
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 3))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 3))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 3))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_pred), 3))
print("R2 score = ", round(sm.r2_score(y_test, y_pred), 3))

# Define feature importances plotting function
def plot_importances(regressor, title, feature_names):
    importances = rf_regressor.feature_importances_
    # importances = 100.0 * (importances/max(importances))

    # Summarize feature importance
    importance_list = []
    for idx1, f in enumerate(feature_names):
        for idx2, i in enumerate(importances):
            if idx1 == idx2:
                print("Feature:", f, "\n|", "Score:", round(i, 5))

    # Normalize feature importance
    importances = rf_regressor.feature_importances_
    importances = 100 * (importances/max(importances))

    # Sort importances by descending order
    idx_asc = np.flipud(np.argsort(importances)[::-1][:12])

    # Center labels
    label_position = np.arange(idx_asc.shape[0]) + 0.5
    # labels = ['\n'.join(wrap(l, 20)) for l in feature_names]

    # Plot the bar chart of feature importances
    plt.figure()
    ax = plt.subplots()
    plt.barh(label_position, importances[idx_asc], align='center')
    plt.yticks(label_position, feature_names[idx_asc])
    plt.xlabel('Red Wine Relative Feature Importance')
    plt.title(title)

    # # Sort importances by descending order
    # idx_dsc = np.flipud(np.argsort(importances))
    #
    # # Center labels
    # label_position = np.arange(idx_dsc.shape[0]) + 0.5
    # # labels = ['\n'.join(wrap(l, 20)) for l in feature_names]
    #
    # # Plot the bar chart of feature importances
    # plt.figure()
    # ax = plt.subplots()
    # plt.bar(label_position, importances[idx_dsc], align='center')
    # plt.xticks(label_position, feature_names[idx_dsc], rotation=60)
    # plt.ylabel('Red Wine Relative Feature Importance')
    # plt.title(title)


    plt.show()

# Plot features' importances
plot_importances(rf_regressor, 'Random Forest regressor', feature_names)