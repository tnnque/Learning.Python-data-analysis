import math
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing, linear_model


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
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

## Take a look at the features
# Density plot and histogram (see features'distribution)
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

# fig, ax = plt.subplots(figsize=(10, 6))
# sns.heatmap(round(correlation,5), square=True, ax=ax, cmap="Blues", linewidths=.5)
# fig.tight_layout()
# plt.show()

# Remove feature columns where correlation > 0.95
up_triangle = correlation.where(np.triu(np.ones(correlation.shape), k = 1).astype(np.bool))
remv = [column for column in up_triangle.columns if any(up_triangle[column] > 0.9)]

X_train = X_train.drop(remv, axis = 1)
X_test = X_test.drop(remv, axis = 1)

# ======================================================================
# Train random forest regressor
rf_regressor = RandomForestRegressor(n_estimators=1000, max_depth=10, min_impurity_decrease=0)
rf_regressor.fit(X_train, y_train)

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

# ======================================================================
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

# ======================================================================
# Train polynomial regressor

polynomial = preprocessing.PolynomialFeatures(degree=2)
X_train_poly = polynomial.fit_transform(X_train)
X_test_poly = polynomial.fit_transform(X_test)
pl_regressor = linear_model.LinearRegression()
pl_regressor.fit(X_train_poly, y_train)

print('PL training score: {}'.format(pl_regressor.score(X_train_poly, y_train)))
print('PL test score: {}'.format(pl_regressor.score(X_test_poly, y_test)))

# Evaluate LN regressor's performance
y_pred_pl = pl_regressor.predict(X_test_poly)
mse = round(sm.mean_squared_error(y_test, y_pred_pl), 5)
rmse = round(math.sqrt(mse), 5)
print("PL mean squared error =", mse)
print("PL root mean squared error =", rmse)

# ======================================================================
# Train logistic regression classirifier
lg_classifier = linear_model.LogisticRegression(solver='liblinear', C=100)
lg_classifier.fit(X_train, y_train)

y_pred_lg = lg_classifier.predict(X_train)
f1 = cross_val_score(lg_classifier, X, y, scoring='f1_weighted', cv=5)
print("F1:" + str(round(f1.mean(), 5)))

# Evaluate RF regressor's performance
print('LG training Score: {}'.format(lg_classifier.score(X_train, y_train)))
print('LG test Score: {}'.format(lg_classifier.score(X_test, y_test)))

### The regressors'results are not as good as expected, need to check for the shape of these features.
#### Because now our data has 11 dimensions, which is not good for visualization, PCA needs to be applied first.
# ======================================================================
# Apply PCA

