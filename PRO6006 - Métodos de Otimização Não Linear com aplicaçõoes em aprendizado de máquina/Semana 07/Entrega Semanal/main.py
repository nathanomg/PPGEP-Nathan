import numpy as np
from optimizers import BFGS, DFP, SR1, FletcherReevesCG

def get_test_suite():

    # Gradient for Rotated Hyper-Ellipsoid
    def grad_rotated_ellipsoid(x):
        n = len(x)
        grad = np.zeros_like(x)
        S = np.cumsum(x)
        for k in range(n):
            grad[k] = 2 * np.sum(S[k:])
        return grad
        
    # Gradient for Sum of Different Powers
    def grad_sum_powers(x):
        n = len(x)
        i = np.arange(1, n + 1)
        return (i + 1) * np.sign(x) * (np.abs(x)**i)
        
    # Gradient for Zakharov
    def grad_zakharov(x):
        n = len(x)
        i = np.arange(1, n + 1)
        B = np.sum(0.5 * i * x)
        term2_deriv = 0.5 * i * (2 * B + 4 * B**3)
        return 2 * x + term2_deriv

    test_functions = [
        {
            "name": "Sphere Function (2D)",
            "func": lambda x: np.sum(x**2),
            "grad": lambda x: 2 * x,
            "x0": np.array([2.0, -1.5]),
            "xmin": np.array([0.0, 0.0]),
        },
        {
            "name": "Rotated Hyper-Ellipsoid (3D)",
            "func": lambda x: np.sum(np.cumsum(x)**2),
            "grad": grad_rotated_ellipsoid,
            "x0": np.array([1.0, 2.0, -1.0]),
            "xmin": np.zeros(3),
        },
        {
            "name": "Bohachevsky Function (2D)",
            "func": lambda x: x[0]**2 + 2*x[1]**2 - 0.3*np.cos(3*np.pi*x[0]) - 0.4*np.cos(4*np.pi*x[1]) + 0.7,
            "grad": lambda x: np.array([
                2*x[0] + 0.3 * 3*np.pi * np.sin(3*np.pi*x[0]),
                4*x[1] + 0.4 * 4*np.pi * np.sin(4*np.pi*x[1])
            ]),
            "x0": np.array([1.0, 1.0]),
            "xmin": np.array([0.0, 0.0]),
        },
        {
            "name": "Sum of Different Powers (3D)",
            "func": lambda x: np.sum(np.abs(x)**(np.arange(1, len(x) + 1) + 1)),
            "grad": grad_sum_powers,
            "x0": np.array([1.0, -1.0, 1.5]),
            "xmin": np.zeros(3),
        },
        {
            "name": "Zakharov Function (3D)",
            "func": lambda x: np.sum(x**2) + (np.sum(0.5 * np.arange(1, len(x) + 1) * x))**2 + (np.sum(0.5 * np.arange(1, len(x) + 1) * x))**4,
            "grad": grad_zakharov,
            "x0": np.array([1.0, 1.0, 1.0]),
            "xmin": np.zeros(3),
        },
        {
            "name": "Matyas Function (2D)",
            "func": lambda x: 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1],
            "grad": lambda x: np.array([
                0.52 * x[0] - 0.48 * x[1],
                0.52 * x[1] - 0.48 * x[0]
            ]),
            "x0": np.array([5.0, 10.0]),
            "xmin": np.array([0.0, 0.0]),
        },
    ]
    return test_functions


test_suite = get_test_suite()

for problem in test_suite:
    print(f"===== TESTING: {problem['name']} =====")
    print(f"Starting Point x0 = {problem['x0']}")
    
    f = problem['func']
    grad = problem['grad']
    x0 = problem['x0']
    
    print(f"True Minimum at f({problem['xmin']}) = {f(problem['xmin']):.6f}\n")
    
    optimizers_to_test = {
        "BFGS": BFGS(f, grad),
        "SR1": SR1(f, grad),
        "Fletcher-Reeves CG": FletcherReevesCG(f, grad),
        "DFP": DFP(f, grad),
    }
    
    for name, optimizer in optimizers_to_test.items():
        print(f"  --- Running {name} ---")
        
        x_min, f_min, iterations = optimizer.minimize(x0)
        error = np.linalg.norm(x_min - problem['xmin'])
        
        print(f"    Result: f(x_min) = {f_min:.6f}")
        print(f"    Iterations: {iterations}")
        print(f"    Error (||x_min - x_true||): {error:.6f}")
    
    print("=" * (len(problem['name']) + 16))
    print("\n")