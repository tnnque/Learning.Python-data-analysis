import math, statistics
import csv
import numpy as np
import pandas as pd
import ppscore as pps
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
# data_scaler = preprocessing.MinMaxScaler(feature_range=(0, 10))
# X = pd.DataFrame(data_scaler.fit_transform(X), columns=df.columns[:11])
feature_names = X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Use Predictive Power Score to find patterns and select features
fig, ax = plt.subplots(figsize=(10, 6))
df_matrix = pps.matrix(df)
sns.heatmap(round(df_matrix,5), annot=True, vmin=0, vmax=1, ax=ax, cmap="BuGn", linewidths=0.5)
fig.subplots_adjust(top=0.5)
t= fig.suptitle('Predictive Power Score Heatmap', fontsize=14)
plt.show()

fig1, ax1 = plt.subplots(figsize=(10, 6))
correlation = df.corr()
sns.heatmap(round(correlation,5), annot=True, ax=ax1, cmap="coolwarm", linewidths=.5)
fig1.subplots_adjust(top=0.5)
t1= fig1.suptitle('Wine Attributes Correlation Heatmap', fontsize=14)
plt.show()