import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.array([[4, 7], [3.5, 8], [3.1, 6.2], [0.5, 1], [1, 2], [1.2, 1.9], [6, 2], [5.7, 1.5], [5.4, 2.2]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])

classifier = linear_model.LogisticRegression(solver='liblinear', C= 10000)
classifier.fit(X, y)

def plot_classifier(classifier, X, y):
    # define ranges
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0
    # denotes step size
    step_size = 0.1
    # define mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))
    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)
    # plot
    plt.figure()
    plt.pcolormesh(x_values, y_values, mesh_output, cmap= plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c= y, s= 80, edgecolors= 'black', linewidths= 1, cmap= plt.cm.Paired)
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    plt.xticks((np.arange(int(min(X[:, 0]) - 1), int(max(X[:, 0]) + 1), 1.0)))
    plt.xticks((np.arange(int(min(X[:, 1]) - 1), int(max(X[:, 1]) + 1), 1.0)))
    plt.show()

plot_classifier(classifier, X, y)