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
    df1 = pd.read_csv('alldata.csv')
    df2 = pd.read_csv('data.csv')
    df1['Time'] = pd.to_datetime(df1['Time'].str.strip('{}'), format='%m/%d/%Y %I:%M:%S %p')
    df2['Time'] = pd.to_datetime(df2['Time'].str.strip('{}'), format='%m/%d/%Y %I:%M:%S %p')
    matching_rows = df1[df1['Time'].isin(df2['Time'])]
    df = df1.replace(to_replace=r'\{|\}', value='', regex=True)
    df['SMA12'] = df['SMA12'].astype(float)
    df['SMA31'] = df['SMA31'].astype(float)
    my_dict = {}
    for index, row in matching_rows.iterrows():
        matching_row = df2[df2['Time'] == row['Time']]
        m1 = get_slope(df, index, 'SMA12')
        m2 = get_slope(df, index, 'SMA31')
        tan_theta_second = (m1 - m2) / (1 + m1 * m2)
        tan_theta_second = abs(tan_theta_second)
        angle_radians_second = math.atan(tan_theta_second)
        angle_degrees_second = math.degrees(angle_radians_second)
        my_dict[angle_degrees_second] = float(matching_row['Win'].str.strip('{}').item())
    """
    Model Training
    """
    X_train = np.array(list(my_dict.keys()))
    y_train = np.array(list(my_dict.values()))
    X_train_reshaped = X_train.reshape(-1, 1)

    w_tmp = np.zeros(X_train_reshaped.shape[1])
    b_tmp = 0.
    alph = 0.001
    iters = 10000
    w_out, b_out, _ = gradient_descent(X_train_reshaped, y_train, w_tmp, b_tmp, alph, iters)
    print(f"\nUpdated parameters: w: {w_out}, b: {b_out}")
    plt.scatter(X_train_reshaped, y_train, color='red', label='Data Points')
    x_values = np.linspace(0, 10, 300)
    y_values = 1 / (1 + np.exp(-(w_out * x_values + b_out)))  # Sigmoid function
    plt.plot(x_values, y_values, color='blue', label='Logistic Regression')
    plt.xlabel('Feature')
    plt.ylabel('Probability/Class')
    plt.legend()
    plt.title('Single Feature Logistic Regression')
    plt.show()

