#The Data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features= 1, noise= 0.4, bias= 50)

#Hypothesis Function
def hypothesis(theta0, theta1, x):
    return theta0 + (theta1 * x)

def plotline(theta0, theta1, X, y):
    max_x = np.max(X) + 100
    min_x = np.min(X) - 100
    xplot = np.linspace(min_x, max_x, 1000)
    yplot = theta0 + (theta1 * xplot)
    plt.plot(xplot, yplot, color = '#ff0000', label= 'Regression Line')

    plt.scatter(X, y)
    plt.axis([-10, 10, 0, 200])
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
    theta0 = theta0 - (dtheta0 * alpha)
    theta1 = theta1 - (dtheta1 * alpha)

    return theta0, theta1

def LinearRegression(X, y):
    theta0 = np.random.rand()
    theta1 = np.random.rand()

    for i in range(0, 1000):
        if i % 100 == 0:
            plotline(theta0, theta1, X, y)
        theta0, theta1 = updateParameters(theta0, theta1, X, y, 0.005)

LinearRegression(X, y)