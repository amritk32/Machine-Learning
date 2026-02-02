import math, copy
import numpy as np
import matplotlib.pyplot as plt

# Load our data set
x_train = np.array([1.0, 2.0])   #features
y_train = np.array([300.0, 500.0])   #target value

def compute_cost(x , y , w , b):
    cost = 0
    m = len(x)
    total_cost = 0

    for i in range(m):
        f_wb = w*x[i] + b
        cost += (f_wb - y[i])**2
    total_cost = ((1)/(2*m))*cost
    return total_cost

def gradient(x, y, w, b):
    m = len(x)
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = x[i] * w + b
        error = f_wb - y[i]  # Calculate error once
        dj_dw += error * x[i]
        dj_db += error
        
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    return dj_dw, dj_db

w_initial = 0
b_initial = 0

iteration = 10000
tmp_alpha = 0.01

def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_func, gradient_func):
    # Use local copies so we don't modify the initials
    w = w_in
    b = b_in
    J_history = []
    p_history = []

    for i in range(num_iters):
        # 1. Calculate the gradients using the function passed in
        dj_dw, dj_db = gradient_func(x, y, w, b) 

        # 2. Update parameters simultaneously
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # 3. Save history for plotting
        if i < 100000:
            J_history.append(cost_func(x, y, w, b))
            p_history.append([w, b])

        # Print progress
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: Cost {J_history[-1]:0.2e} "
                  f"w: {w: 0.3e}, b:{b: 0.5e}")
 
    return w, b, J_history, p_history

# Corrected Function Call
w_final, b_final, J_hist, p_hist = gradient_descent(x_train, y_train, w_initial, b_initial, 
                                                   tmp_alpha, iteration, compute_cost, gradient)

print(f"Price of 1000 Square feet {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"Price of 2000 Square feet {w_final*2.0 + b_final:0.1f} Thousand dollars")
print(f"Price of 2587 Square feet {w_final*2.587 + b_final:0.1f} Thousand dollars")