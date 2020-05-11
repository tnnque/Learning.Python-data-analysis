import math, statistics
import csv
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Define dataset loading function
def data_load(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :11]
    y = df.iloc[:, -1:]

    return df, X, y

# Input and separate X, y for training and testing data
df, X, y = data_load('winequality-red.csv')
data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
X = pd.DataFrame(data_scaler.fit_transform(X), columns=df.columns[:11])
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# Train the random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=100, min_impurity_decrease=0)
rf_regressor.fit(X_train, y_train)

# Evaluate RF regressor's performance
print('Training Score: {}'.format(rf_regressor.score(X_train, y_train)))
print('Test Score: {}'.format(rf_regressor.score(X_test, y_test)))
y_pred_rf = rf_regressor.predict(X_test)
print("RF Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred_rf), 5))
print("RF Mean squared error =", round(sm.mean_squared_error(y_test, y_pred_rf), 5))
print("RF Median absolute error =", round(sm.median_absolute_error(y_test, y_pred_rf), 5))
print("RF Explained variance score =", round(sm.explained_variance_score(y_test, y_pred_rf), 5))
print("RF R2 score = ", round(sm.r2_score(y_test, y_pred_rf), 3))

# Define feature importances plotting function
def plot_importances(regressor, title, feature_names):
    # Normalize feature importance
    importances = rf_regressor.feature_importances_
    importances = 100 * (importances/max(importances))

    # Summarize feature importance
    for idx1, f in enumerate(feature_names):
        for idx2, i in enumerate(importances):
            if idx1 == idx2:
                print("Feature:", f, "\n|", "Score:", round(i, 5))

    # Sort importances by descending order
    idx_asc = np.flipud(np.argsort(importances)[::-1][:12])

    # Center labels
    label_position = np.arange(idx_asc.shape[0]) + 0.5

    # Plot the bar chart of feature importances
    plt.figure()
    ax = plt.subplots()
    plt.barh(label_position, importances[idx_asc], align='center')
    plt.yticks(label_position, feature_names[idx_asc])
    plt.xlabel('Red Wine Relative Feature Importance')
    plt.title(title)
    plt.show()

# Plot features' importances
plot_importances(rf_regressor, 'Random Forest regressor', feature_names)

#======================================================================
# Train multivariate linear regressor
ln_regressor = linear_model.LinearRegression()
ln_regressor.fit(X_train, y_train)

# Print out LN model score
print('LN training score: {}'.format(ln_regressor.score(X_train, y_train)))
print('LN test score: {}'.format(ln_regressor.score(X_test, y_test)))

# Evaluate LN regressor's performance
y_pred_ln = ln_regressor.predict(X_test)
mse = round(sm.mean_squared_error(y_test, y_pred_ln), 5)
rmse = round(math.sqrt(mse), 5)
print("LN mean squared error =", mse)
print("LN mean squared error =", rmse)

### The training seems to be overfitting since linear regression is quite sensitive to outliers
# Outlier detection using visualization
# Density plot and histogram (see features'distribution)
for i in X:
    sns.distplot(X[i], hist=True, kde=True, bins=10, color='red',
        hist_kws={'color':'gray', 'edgecolor':'black'},
        kde_kws={'linewidth': 3})
    plt.show()

# Box plot
ax = sns.boxplot(data=X, orient="h", palette="Set2")
plt.show()

# Introduce Ridge regressor for regularization

# steps = [
#     ('scalar', preprocessing.StandardScaler()),
#     ('poly', preprocessing.PolynomialFeatures(degree=2)),
#     ('model', Ridge(alpha=10, fit_intercept=True))
# ]
#
# ridge_pipe = Pipeline(steps)
# ridge_pipe.fit(X_train, y_train)
#
# print('Ridge training Score: {}'.format(ridge_pipe.score(X_train, y_train)))
# print('Ridge test Score: {}'.format(ridge_pipe.score(X_test, y_test)))
#
# steps = [
#     ('scalar', preprocessing.StandardScaler()),
#     ('poly', preprocessing.PolynomialFeatures(degree=2)),
#     ('model', Lasso(alpha=0.3, fit_intercept=True))
# ]
#
# lasso_pipe = Pipeline(steps)
#
# lasso_pipe.fit(X_train, y_train)
#
# print('Training score: {}'.format(lasso_pipe.score(X_train, y_train)))
# print('Test score: {}'.format(lasso_pipe.score(X_test, y_test)))
