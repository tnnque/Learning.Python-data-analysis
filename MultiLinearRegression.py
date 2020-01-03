import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_excel('/Users/tnnque/PycharmProjects/data-analytics-ex/datasets/CCPP/Folds5x2_pp.xlsx')
# print(data.head(5))
X = data.iloc[:, :4]
y = data.iloc[:, -1:]

X = StandardScaler().fit_transform(X)

def cost_function(X, Y, B):
    m = len(Y)
    t = X.dot(B)
    y = np.transpose(np.array(Y)).reshape((7000, ))
    J = np.sum((X.dot(B) - y) ** 2)/(2 * m)
    return J

def batch_gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    n = len(Y)

    for iterations in range(iterations):
        #Hypothesis:
        h = X.dot(B)
        #Different between Hypothesis vs Actual Y:
        temp = np.transpose(np.array(Y)).reshape((7000, ))
        # loss = h - np.array(Y)
        loss = h - temp
        #Gradient calculation:
        gradient = X.T.dot(loss) / n
        #Gradient update:
        B = B - alpha * gradient
        #New cost gradient
        cost = cost_function(X, Y, B)
        cost_history[iterations] = cost
    return B, cost_history

m = 7000
f = 2
X_train = X[:m, :f]
X_train = np.c_[np.ones(len(X_train), dtype='int64'), X_train]
y_train = y[:m]
X_test = X[m:, :f]
X_test = np.c_[np.ones(len(X_test), dtype='int64'), X_test]
y_test = y[m:]

#Initial coefficients:
B = np.zeros(X_train.shape[1])
alpha = 0.005
iter_ = 2000

newB, cost_history = batch_gradient_descent(X_train, y_train, B, alpha, iter_)

def pred(X_test, newB):
    y_pred = X_test.dot(newB)
    return y_pred

y_ = pred(X_test, newB)

def r2(y, y_):
    y_ = np.transpose(np.array(y_)).reshape((2568, ))
    sst = np.sum((y - y.mean())**2)
    ssr = np.sum((y - y_)**2)

    r2 = 1 - (ssr/sst)
    return r2

print(r2(y_, y_test))

# d = abs(y_pred - y_test)