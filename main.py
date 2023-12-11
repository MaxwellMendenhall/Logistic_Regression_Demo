import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w)+b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)

    cost = cost/m
    return cost


def derivative(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i],w)+b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        dj_db = dj_db + err_i
    dj_dw = dj_dw/m
    dj_db = dj_db / m

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = derivative(X, y, w, b)

        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            J_history.append(cost_function(X, y, w, b))

        if i % math.ceil(num_iters/10) == 0:
            print(f"Interation {i:4d}: Cost {J_history[-1]}    ")

    return w, b, J_history


def get_slope(data, point, y):
    return (data.iloc[point + 1][y] - data.iloc[point - 1][y])/2


if __name__ == "__main__":
    """
    Sample data for logistic regression
    """
    np.random.seed(0)
    X_train_single = np.random.rand(100, 1) * 10 
    y_train_single = np.random.randint(0, 2, size=X_train_single.shape[0])

    w_tmp = np.zeros(X_train_single.shape[1])
    b_tmp = 0.
    alph = 0.01
    iters = 10000

    w_out, b_out, _ = gradient_descent(X_train_single, y_train_single, w_tmp, b_tmp, alph, iters)

    print(f"\nUpdated parameters: w: {w_out}, b: {b_out}")

    plt.scatter(X_train_single, y_train_single, color='red', label='Data Points')
    x_values = np.linspace(0, 10, 300)
    y_values = 1 / (1 + np.exp(-(w_out * x_values + b_out)))  # Sigmoid function
    plt.plot(x_values, y_values, color='blue', label='Logistic Regression')
    plt.xlabel('Feature')
    plt.ylabel('Probability/Class')
    plt.legend()
    plt.title('Single Feature Logistic Regression')
    plt.show()

    """
    Sample data for finding angles
    """
    np.random.seed(0)
    data = np.random.randn(100).cumsum() + 100
    df_with_smas = pd.DataFrame(data, columns=['Price'])
    df_with_smas['SMA_10'] = df_with_smas['Price'].rolling(window=10).mean()
    df_with_smas['SMA_20'] = df_with_smas['Price'].rolling(window=20).mean()

    m1 = get_slope(df_with_smas, 29, 'SMA_10')
    m2 = get_slope(df_with_smas, 29, 'SMA_20')

    tan_theta_second = abs(m1 - m2) / (1 + m1 * m2)
    angle_radians_second = math.atan(tan_theta_second)
    angle_degrees_second = math.degrees(angle_radians_second)

    print(angle_degrees_second)
