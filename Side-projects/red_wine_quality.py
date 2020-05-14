import math, statistics
import csv
import numpy as np
import pandas as pd
import ppscore as pps
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn import preprocessing, linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
import seaborn as sns

classifiers = {}
parameters = {}

# Create and update dictionary with Gradient Boosting
classifiers.update({"Gradient Boosting": GradientBoostingClassifier()})
parameters.update({"Gradient Boosting": {
                                        "classifier__learning_rate":[0.15,0.1,0.05,0.01,0.005,0.001],
                                        "classifier__n_estimators": [200],
                                        "classifier__max_depth": [2,3,4,5,6],
                                        "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                        "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                        "classifier__max_features": ["auto", "sqrt", "log2"],
                                        "classifier__subsample": [0.8, 0.9, 1]
                                         }})

# Create and update dictionary with Extra Trees Ensemble
classifiers.update({"Extra Trees Ensemble": ExtraTreesClassifier()})
parameters.update({"Extra Trees Ensemble": {
                                            "classifier__n_estimators": [200],
                                            "classifier__class_weight": [None, "balanced"],
                                            "classifier__max_features": ["auto", "sqrt", "log2"],
                                            "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                            "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                            "classifier__criterion" :["gini", "entropy"]     ,
                                            "classifier__n_jobs": [-1]
                                             }})

# Create and update dictionary with Random Forest Classifier
classifiers.update({"Random Forest": RandomForestClassifier()})
parameters.update({"Random Forest": {
                                    "classifier__n_estimators": [200],
                                    "classifier__class_weight": [None, "balanced"],
                                    "classifier__max_features": ["auto", "sqrt", "log2"],
                                    "classifier__max_depth" : [3, 4, 5, 6, 7, 8],
                                    "classifier__min_samples_split": [0.005, 0.01, 0.05, 0.10],
                                    "classifier__min_samples_leaf": [0.005, 0.01, 0.05, 0.10],
                                    "classifier__criterion" :["gini", "entropy"]     ,
                                    "classifier__n_jobs": [-1]
                                     }})

# Define dataset loading function
def data_load(filename):
    df = pd.read_csv(filename)
    X = df.iloc[:, :11]
    y = df.iloc[:, -1:]

    return df, X, y

# Input and separate X, y for training and testing data
df, X, y = data_load('winequality-red.csv')
# data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
# X = pd.DataFrame(data_scaler.fit_transform(X), columns=df.columns[:11])
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

## Take a look at the features
#Density plot and histogram (see features'distribution)
for i in X:
    sns.distplot(X[i], hist=True, kde=True, bins=10, color='red',
        hist_kws={'color':'gray', 'edgecolor':'black'},
        kde_kws={'linewidth': 3})
    plt.show()

# Box plot to see each feature's outliers
ax = sns.boxplot(data=X, orient="h", palette="Set2")
plt.show()

# Use correlation matrix to detect highly correlated features
correlation = X_train.corr(method='spearman').abs()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(round(correlation,5), square=True, ax=ax, cmap="Blues", linewidths=.5)
fig.tight_layout()
plt.show()

# Remove feature columns where correlation > 0.95
up_triangle = correlation.where(np.triu(np.ones(correlation.shape), k = 1).astype(np.bool))
remv = [column for column in up_triangle.columns if any(up_triangle[column] > 0.95)]

X_train = X_train.drop(remv, axis = 1)
X_test = X_test.drop(remv, axis = 1)


# ### Start the work with some 'feature importance' classifiers
# Create a dictionary for the classifiers
FEATURE_IMPORTANCE = {"Gradient Boosting", "Extra Trees Ensemble", "Random Forest"}

# # Train the random forest regressor
# rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_impurity_decrease=0)
# rf_regressor.fit(X_train, y_train)

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

# Evaluate RF regressor's performance
print('RF training Score: {}'.format(rf_regressor.score(X_train, y_train)))
print('RF test Score: {}'.format(rf_regressor.score(X_test, y_test)))
y_pred_rf = rf_regressor.predict(X_test)

print("RF Mean squared error =", round(sm.mean_squared_error(y_test, y_pred_rf), 5))
print("RF Explained variance score =", round(sm.explained_variance_score(y_test, y_pred_rf), 5))

##

======================================================================
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
print("LN root mean squared error =", rmse)