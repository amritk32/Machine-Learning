import numpy as np
import matplotlib.pyplot as plt
import math
import copy

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

X_features = ['size(sqft)','bedrooms','floors','age']





b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

def prediction(x, w, b):
    m = len(x)

    
    f_wb = np.dot(x,w) + b
    return f_wb

def compute_cost(X, y, w, b):
    m = len(X)
    cost = 0

    for i in range(m):
        f_wb = np.dot(X[i],w) + b
        cost += (f_wb - y[i])**2
    cost = cost/(2*m)

    return cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')

def gradient(X,y,w,b):
    m,n = X.shape

    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        err = (np.dot(X[i],w) + b) - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err*X[i,j]
        dj_db += err
    dj_db = dj_db/m
    dj_dw = dj_dw/m

    return dj_db , dj_dw

def gradient_descent(X, y, w_in, b_in, compute_cost, gradient, alpha, iteration):

    cost = []
    w_final = copy.deepcopy(w_in)
    b_final = b_in
    iters = []

    for i in range(iteration):
        dj_db,dj_dw = gradient(X,y,w_final,b_final)

        w_final = w_final - alpha*dj_dw
        b_final = b_final - alpha*dj_db

        if i%math.ceil(iteration/10) == 0:
            iters.append(i)
            cost.append(compute_cost(X, y, w_final, b_final))
            print(f"Iteration {i:4d} , Cost {cost[-1]:8.2f}")

    return w_final, b_final, cost,iters

def zscore_normalize_features(X):
  
    # find the mean of each column/feature
    mu     = np.mean(X, axis=0)                 # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma  = np.std(X, axis=0)                  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    X_norm = (X - mu) / sigma      

    return (X_norm, mu, sigma)



inital_w = np.zeros_like(w_init)
initial_b = 0

iterations = 1000
alpha = 5.0e-7



w_f,b_f,J_his, iters = gradient_descent(X_train,y_train,inital_w,initial_b,compute_cost,gradient,alpha,iterations)
# print(f"b,w found by gradient descent: {b_f:0.2f},{w_f} ")
m,_ = X_train.shape
# for i in range(m):
#     print(f"prediction: {np.dot(X_train[i], w_f) + b_f:0.2f}, target value: {y_train[i]}")

val,_,_ = zscore_normalize_features(X_train)

size_original = X_train[:,0]
size_norm = val[:,0]

# 1️⃣ Normalize features
X_norm, mu, sigma = zscore_normalize_features(X_train)

# 2️⃣ Plot histograms before and after normalization
fig, ax = plt.subplots(2, 4, figsize=(16, 6))  # 2 rows: original vs normalized, 4 columns for features

for i in range(4):
    # Row 0: original features
    ax[0, i].hist(X_train[:, i], bins=3, color='blue')
    ax[0, i].set_xlabel(X_features[i])
    if i == 0:
        ax[0, i].set_ylabel("Count")
    ax[0, i].set_title("Original Feature")
    ax[0, i].grid(True)

    # Row 1: normalized features
    ax[1, i].hist(X_norm[:, i], bins=3, color='orange', alpha=0.7)
    ax[1, i].set_xlabel(X_features[i])
    if i == 0:
        ax[1, i].set_ylabel("Count")
    ax[1, i].set_title("Z-score Normalized Feature")
    ax[1, i].grid(True)

fig.suptitle("Feature Distributions: Original vs Z-score Normalized", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
