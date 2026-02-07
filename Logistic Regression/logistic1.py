import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(z):
    g = 1/(1 + np.exp(-z))
    return g



X_train = np.array([
    [0.5, 1.5],
    [1, 1],
    [1.5, 0.5],
    [3, 0.5],
    [2, 2],
    [1, 2.5]
])

y_train = np.array([0, 0, 0, 1, 1, 1])

# Separate points by class
X0 = X_train[y_train == 0]   # class 0
X1 = X_train[y_train == 1]   # class 1

# Plot
plt.figure(figsize=(4,4))

plt.scatter(X0[:,0], X0[:,1], marker='o', color='blue', label='y = 0')
# This upper line syntax means X0[:,0] = [0.5, 1.0, 1.5] and X0[:,1] = [1.5, 1.0, 0.5]
# Where X0 means where y is 0 means not
# X1 means where y is 1
# X1[:,0] = [3.0, 2.0, 1.0] and X1[:,1] = [0.5, 2.0, 2.5] 
plt.scatter(X1[:,0], X1[:,1], marker='x', color='red',  label='y = 1')


def compute_logistic_cost(x, y, w, b):

    m = x.shape[0]

    z = np.zeros(m)
    f_wb = np.zeros(m)
    cost = 0

    for i in range(m):
        z[i] = np.dot(x[i],w) + b
        f_wb[i] = sigmoid(z[i])
        cost += (-y[i]*np.log(f_wb[i]) - (1 - y[i])*np.log(1 - f_wb[i]))
    cost = ((1)/m)*cost
    return cost
 
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_logistic_cost(X_train, y_train, w_tmp, b_tmp))

plt.xlabel('x0')
plt.ylabel('x1')
plt.legend()
plt.grid(True)
plt.xlim(0, 4)
plt.ylim(0, 3.5)
