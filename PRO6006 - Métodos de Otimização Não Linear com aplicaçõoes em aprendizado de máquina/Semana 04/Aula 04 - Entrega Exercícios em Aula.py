import numpy as np
from scipy.optimize import line_search

def levenberg_marquardt_modification(f, grad, hess, x0, delta=1e-3, max_iter=100, tol=1e-6):
    """
    Implements the Levenberg-Marquardt modification of Newton's method.

    Args:
        f (callable): The objective function to be minimized. f(x) -> float.
        grad (callable): The gradient of the objective function. grad(x) -> np.ndarray.
        hess (callable): The Hessian of the objective function. hess(x) -> np.ndarray.
        x0 (np.ndarray): The initial guess for the minimum.
        delta (float): A small positive constant to ensure the modified Hessian is positive definite.
        max_iter (int): The maximum number of iterations.
        tol (float): The tolerance for the norm of the gradient to determine convergence.

    Returns:
        np.ndarray: The point at which the minimum is found.
    """
    x_k = np.asarray(x0, dtype=float)

    for k in range(max_iter):
        # 1. Calculate the gradient and Hessian at the current point x_k
        grad_k = grad(x_k)
        hess_k = hess(x_k)

        # Check for convergence
        if np.linalg.norm(grad_k) < tol:
            print(f"Convergence reached at iteration {k}.")
            break

        # 2. Calculate the eigenvalues of the Hessian H(x_k)
        eigenvalues = np.linalg.eigvalsh(hess_k)
        min_eigenvalue = np.min(eigenvalues)

        # 3. Determine epsilon_k
        # epsilon_k is the smallest non-negative constant such that the eigenvalues of
        # (epsilon_k * I + H(x_k)) are >= delta.
        # This means: min_eigenvalue + epsilon_k >= delta
        epsilon_k = 0.0
        if min_eigenvalue < delta:
            epsilon_k = delta - min_eigenvalue

        # 4. Define the modified Hessian M_k
        M_k = epsilon_k * np.eye(len(x_k)) + hess_k

        # 5. Calculate the search direction d_k = -inv(M_k) * grad(f(x_k))
        # It's more numerically stable to solve the linear system M_k * d_k = -grad_k
        try:
            d_k = np.linalg.solve(M_k, -grad_k)
        except np.linalg.LinAlgError:
            print("Error: Singular matrix. Could not solve for the search direction.")
            # As a fallback, one could use the steepest descent direction
            d_k = -grad_k


        # 6. Find alpha_k that minimizes f(x_k + alpha * d_k) using a line search
        # scipy.optimize.line_search aims to find an alpha satisfying Wolfe conditions
        alpha_k, _, _, _, _, _ = line_search(f, grad, x_k, d_k)

        if alpha_k is None:
            # If line search fails, take a small fixed step or stop.
            print("Line search failed. Using a small fixed step size.")
            alpha_k = 1e-3


        # 7. Update x_{k+1} = x_k + alpha_k * d_k
        x_k = x_k + alpha_k * d_k

        print(f"Iteration {k+1}: x = {x_k}, f(x) = {f(x_k)}")

    else:
        print("Maximum number of iterations reached.")

    return x_k

if __name__ == '__main__':
    # --- Example Usage ---
    # We will use a classic test function: the Rosenbrock function.
    # f(x, y) = (a - x)^2 + b * (y - x^2)^2
    # For this example, we use a=1 and b=100. The minimum is at (1, 1).

    def rosenbrock(x):
        """The Rosenbrock function."""
        return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

    def rosenbrock_grad(x):
        """Gradient of the Rosenbrock function."""
        grad = np.zeros_like(x)
        grad[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
        grad[1] = 200 * (x[1] - x[0]**2)
        return grad

    def rosenbrock_hess(x):
        """Hessian of the Rosenbrock function."""
        hess = np.zeros((2, 2))
        hess[0, 0] = 2 - 400 * (x[1] - 3 * x[0]**2)
        hess[0, 1] = -400 * x[0]
        hess[1, 0] = -400 * x[0]
        hess[1, 1] = 200
        return hess

    # Initial guess
    initial_guess = np.array([0.0, 0.0])

    # Run the optimization
    # The choice of delta is an "art". A small delta is used here.
    minimum = levenberg_marquardt_modification(
        f=rosenbrock,
        grad=rosenbrock_grad,
        hess=rosenbrock_hess,
        x0=initial_guess,
        delta=1e-4
    )

    print("\n--- Results ---")
    print(f"The found minimum is at: {minimum}")
    print(f"The value of the function at the minimum is: {rosenbrock(minimum)}")
    print("The true minimum is at: [1. 1.]")