import csv
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

path = input("Enter path:")
X_range = input("Enter X_range:")
y_range = input("Enter y_range:")

# Define dataset loading function
def data_load(filename, X_range, y_range):
    file_reader = csv.reader(open(filename, 'rb'), delimiter=',')
    X, y = [], []
    for row in file_reader:
        X.append(row[X_range])
        y.append(row[y_range])
# Separate features' names
    feature_names = np.array(X[0])
    return np.array(X[1:]).astype(np.float32), np.array(y[1:]).astype(np.float32), feature_names



