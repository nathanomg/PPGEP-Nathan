import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def objective_regression_function(theta, X, y):
    theta_0 = theta[0]
    theta_coeffs = theta[1:]
    
    y_pred = theta_0 + X @ theta_coeffs
    error  = y - y_pred
    sse    = np.sum(error**2)
    
    return sse

def objective_ridge(theta, X, y, lambda_val):
    theta_coeffs = theta[1:]
    sse          =  objective_regression_function(theta, X, y)
    l2_penalty   = lambda_val * np.sum(theta_coeffs**2)
    
    return sse + l2_penalty

def univariate_regression():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([2.0, 2.8, 3.6, 4.5])

    initial_theta = np.array([0.0, 0.0])

    result = minimize(objective_regression_function, initial_theta, args=(X, y), method='Nelder-Mead')

    theta_0, theta_1 = result.x

    print("--- Optimization Results ---")
    print(f"The optimization was successful: {result.success}")
    print(f"Optimal theta_0: {theta_0:.4f}")
    print(f"Optimal theta_1: {theta_1:.4f}")
    print(f"Final SSE: {result.fun:.4f}")

    print(f"\nModel: y = {theta_0:.2f} + {theta_1:.2f}*x1")

    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='red', label='Observed Data Points')

    x_line = np.linspace(min(X), max(X), 100)
    y_line = theta_0 + theta_1 * x_line
    plt.plot(x_line, y_line, color='blue', label=f'Fitted Line: y = {theta_0:.2f} + {theta_1:.2f}x')

    plt.title('Univariate Linear Regression')
    plt.xlabel('$x_i$')
    plt.ylabel('$y_i$')
    plt.legend()
    plt.grid(True)
    plt.show()

def bivariate_regression():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0]
    ])
    y = np.array([2.1, 2.5, 4.9, 5.1])

    initial_theta = np.array([0.0, 0.0, 0.0])
    result_std = minimize(objective_regression_function, initial_theta, args=(X, y), method='Nelder-Mead')

    theta_0, theta_1, theta_2 = result_std.x
    print("--- Standard Bivariate Model Results (with theta_2) ---")
    print(f"Optimal theta_0: {theta_0:.4f}")
    print(f"Optimal theta_1: {theta_1:.4f}")
    print(f"Optimal theta_2: {theta_2:.4f}")
    print(f"Final SSE: {result_std.fun:.4f}")
    print(f"\nModel: y = {theta_0:.2f} + {theta_1:.2f}*x1 + {theta_2:.2f}*x2")

def ridge_regression():
    X = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
        [3.0, 4.0],
        [4.0, 3.0]
    ])
    y = np.array([2.1, 2.5, 4.9, 5.1])

    initial_theta = np.array([0.0, 0.0, 0.0])
    lambda_val    = 1.0 

    result_ridge = minimize(objective_ridge, initial_theta, args=(X, y, lambda_val), method='BFGS')

    theta_opt_ridge = result_ridge.x
    print(f"--- Ridge Regression Results (lambda = {lambda_val}) ---")
    print(f"Optimal theta_0: {theta_opt_ridge[0]:.4f}")
    print(f"Optimal theta_1: {theta_opt_ridge[1]:.4f}")
    print(f"Optimal theta_2: {theta_opt_ridge[2]:.4f}")
    print(f"\nModel: y = {theta_opt_ridge[0]:.2f} + {theta_opt_ridge[1]:.2f}*x1 + {theta_opt_ridge[2]:.2f}*x2")

univariate_regression()
bivariate_regression()
ridge_regression()