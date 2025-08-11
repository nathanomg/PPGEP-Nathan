import numpy as np
from scipy.optimize import line_search

class Optimizer():
    def __init__(self, f, grad, tol=1e-6, max_iter=1000):
        self.f = f
        self.grad = grad
        self.tol = tol
        self.max_iter = max_iter

    def minimize(self, x0):
        pass

# ==============================================================================
# Conjugate Gradient Methods
# ==============================================================================

class FletcherReevesCG(Optimizer):
    def minimize(self, x0):
        x = np.copy(x0)
        g = self.grad(x)
        p = -g
        g_dot_g = np.dot(g, g)
        k = 0

        for k in range(self.max_iter):
            if np.linalg.norm(g) < self.tol:
                break
            
            alpha_search = line_search(self.f, self.grad, x, p)
            alpha = alpha_search[0]
            
            if alpha is None:
                alpha = 1e-3

            x = x + alpha * p
            g_new = self.grad(x)
            
            g_dot_g_new = np.dot(g_new, g_new)
            beta = g_dot_g_new / g_dot_g
            
            p = -g_new + beta * p
            g = g_new
            g_dot_g = g_dot_g_new
            
        return x, self.f(x), k + 1

# ==============================================================================
# Quasi-Newton Methods
# ==============================================================================

class QuasiNewtonOptimizer(Optimizer):
    def minimize(self, x0):
        x = np.copy(x0)
        n = len(x0)
        H = np.eye(n)
        g = self.grad(x)
        k = 0

        for k in range(self.max_iter):
            if np.linalg.norm(g) < self.tol:
                break
            
            p = -np.dot(H, g)

            alpha_search = line_search(self.f, self.grad, x, p)
            alpha = alpha_search[0]
            
            if alpha is None:
                 print(f"{self.__class__.__name__}: Line search failed at iteration {k}.")
                 break

            s = alpha * p
            x_new = x + s
            g_new = self.grad(x_new)
            y = g_new - g
            
            H = self._update_H(H, s, y)
            
            x = x_new
            g = g_new  # CRITICAL FIX
        
        return x, self.f(x), k + 1

    def _update_H(self, H, s, y):
        pass

class BFGS(QuasiNewtonOptimizer):
    def _update_H(self, H, s, y):
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)
        
        sy_dot = np.dot(y.T, s)
        if sy_dot <= 1e-8:
            return H 
        
        rho = 1.0 / sy_dot
        I = np.eye(len(s))
        
        term1 = I - rho * np.dot(s, y.T)
        term2 = I - rho * np.dot(y, s.T)
        term3 = rho * np.dot(s, s.T)
        
        return np.dot(term1, np.dot(H, term2)) + term3

class DFP(QuasiNewtonOptimizer):
    def _update_H(self, H, s, y):
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)

        sy_dot = np.dot(s.T, y)
        if sy_dot <= 1e-8:
            return H
            
        term1 = np.dot(s, s.T) / sy_dot
        
        Hy = np.dot(H, y)
        yHy_dot = np.dot(y.T, Hy)
        if yHy_dot <= 1e-8:
            return H
            
        term2 = -np.dot(Hy, Hy.T) / yHy_dot
        
        return H + term1 + term2

class SR1(QuasiNewtonOptimizer):
    def _update_H(self, H, s, y):
        s = s.reshape(-1, 1)
        y = y.reshape(-1, 1)

        s_minus_Hy = s - np.dot(H, y)
        denominator = np.dot(s_minus_Hy.T, y)
        
        if np.abs(denominator) < 1e-8 * np.linalg.norm(y) * np.linalg.norm(s_minus_Hy):
            return H
            
        return H + np.dot(s_minus_Hy, s_minus_Hy.T) / denominator