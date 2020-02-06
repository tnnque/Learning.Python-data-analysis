import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame

#Data
path = '/Users/tnnque/PycharmProjects/data-analytics-ex/que/train_small_cleaned.csv'
col_name = input("Enter variable:")

def extract(path, col_name):
    df = pd.read_csv(path)
    current_row = None
    old_row = None
    sum_row = 0
    n = 0
    y = []
    X = []
    for index, row in df.iterrows():
        if index == 0:
            sum_row = row[col_name]
            n = 1
            old_row = row
        else:
            current_row = row
            if current_row['dayofweek'] == old_row['dayofweek']:
                sum_row += row[col_name]
                n += 1
                old_row = row
            else:
                y.append(sum_row/n)
                X.append(len(X) + 1)
                sum_row = row[col_name]
                n = 1
                old_row = row

    return X/np.max(X), y/np.max(y)

def hypothesis(theta0, theta1, x):
    return theta0 + (theta1 * x)

def plotline(theta0, theta1, X, y):
    max_x = np.max(X)
    min_x = np.min(X)

    xplot = np.linspace(max_x, min_x, 2)
    yplot = theta0 + (theta1 * xplot)

    plt.plot(xplot, yplot, color = '#ff0000', label= 'Regression Line')
    plt.scatter(X, y, label= col_name + " Price")
    plt.legend()
    plt.show()

def derivatives(theta0, theta1, X, y):
    dtheta0 = 0
    dtheta1 = 0
    for xi, yi in zip(X, y):
        dtheta0 += hypothesis(theta0, theta1, xi) - yi
        dtheta1 += (hypothesis(theta0, theta1, xi) - yi) * xi

    dtheta0 /= len(X)
    dtheta1 /= len(X)

    return dtheta0, dtheta1

def updateParameters(theta0, theta1, X, y, alpha):
    dtheta0, dtheta1 = derivatives(theta0, theta1, X, y)
    theta0 = theta0 - (alpha * dtheta0)
    theta1 = theta1 - (alpha * dtheta1)

    return theta0, theta1

def LinearRegression(path, col_name):
    X, y = extract(path, col_name)

    theta0 = -np.random.rand()
    theta1 = np.random.rand()

    for i in range(0, 10000):
        # if i % 1000 == 0:
        #     plotline(theta0, theta1, X, y)
        theta0, theta1 = updateParameters(theta0, theta1, X, y, 0.01)
    plotline(theta0, theta1, X, y)

LinearRegression(path, col_name)