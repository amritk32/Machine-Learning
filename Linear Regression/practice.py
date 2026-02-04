import numpy as np
import math
import matplotlib.pyplot as plt

# Dataset 1

x_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
y_train = np.array([35, 38, 40, 43, 47, 50, 54, 57, 61, 64, 68, 70, 73, 76, 80, 83, 86, 88, 91, 93, 95, 96, 97, 98, 100])

def compute_model_output(x, y, w, b):
    m = len(x)
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w*x[i] + b
    
    return f_wb

def calculate_error(x, y, w, b):
    total_error = 0
    current_error = 0

    m = len(x)

    for i in range(m):
        f_wb = w*x[i] + b
        current_error += (f_wb - y[i])**2
    total_error = (1/(2*m))*current_error
    return total_error

def find_gradient(x, y, w, b):
    
    dj_db = 0
    dj_dw = 0

    m = len(x)

    temp_db = 0
    temp_dw = 0

    for i in range(m):
        temp_dw += (w*x[i] + b - y[i])*x[i]
        temp_db += (w*x[i] + b - y[i])
    dj_dw = (1/m)*temp_dw
    dj_db = (1/m)*temp_db

    return dj_dw , dj_db

def gradient_desc(x, y, w_initial, b_initial, calculate_error, find_gradient, compute_model_output, iterations, alpha):
    
    w_final = w_initial
    b_final = b_initial

    cost_history = []
    parameter_history = []

    for i in range(iterations):
        dj_dw , dj_db = find_gradient(x, y, w_final, b_final)

        w_final = w_final - alpha*dj_dw
        b_final = b_final - alpha*dj_db

        if i < 100000:
            cost_history.append(calculate_error(x, y, w_final, b_final))
            parameter_history.append([w_final,b_final])
        
        # if i % math.ceil(iterations / 10) == 0:
        #     print(f"Iteration {i:4}: Cost {cost_history[-1]:0.2e} "
        #           f"w: {w_final: 0.3e}, b:{b_final: 0.5e}")
    
    return w_final , b_final , cost_history, parameter_history

w_init = 0
b_init = 0
alpha = 0.001
iter1 = 100000
w_f, b_f, C_his, P_his = gradient_desc(x_train, y_train, w_init, b_init, calculate_error, find_gradient, compute_model_output, iter1,alpha)

pred_line = compute_model_output(x_train, y_train, w_f,b_f)
print(f"X = 27 {w_f*27 + b_f:0.1f} final values of w and b {w_f:0.1f} , {b_f:0.1f}")
print(f"X = 26 {w_f*26 + b_f:0.1f} final values of w and b {w_f:0.1f} , {b_f:0.1f}")
print(f"X = 30 {w_f*30 + b_f:0.1f} final values of w and b {w_f:0.1f} , {b_f:0.1f}")

plt.scatter(x_train,y_train,marker='x',color = 'red',s = 50,label = "Actual Data")
plt.plot(x_train,w_f*x_train+b_f,color = 'blue',label = "Prediction line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
# plt.show()