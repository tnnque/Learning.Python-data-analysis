import numpy as np
import matplotlib.pyplot as plt

def load_data(input_file):
    X = []
    y = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(i) for i in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1])
    X = np.array(X)
    y = np.array(y)
    return X, y

input_file = 'data_multivar_3.txt'
X, y = load_data(input_file)
class_0 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
class_1 = np.array([X[i] for i in range(len(X)) if y[i] == 1])


# Plot data
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], facecolors= 'black', edgecolors= 'black', marker= 's')
plt.scatter(class_1[:, 0], class_1[:, 1], facecolors= 'None', edgecolors= 'black', marker= 's')
plt.title('Input Data')
plt.show()