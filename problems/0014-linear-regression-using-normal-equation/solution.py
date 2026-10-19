import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    x, y = np.array(X), np.array(y).reshape(-1, 1)
    x_t = x.T
    theta = np.linalg.inv(x_t.dot(x)).dot(x_t).dot(y)
    return np.round(theta.flatten(), 4).tolist()