import numpy as np

# Maximize area, but with more constraints

"""
Input: numpy array
"""
def objective(x):
    return x[0] * x[1]

def penalty(x, rho=1000.0):
    """
    Penalty for constraint violation: square the perim cost
    """
    x1, x2 = x
    penalties = 0.0

    eq = 2 * x1 + 2 * x2
    penalties += rho * eq ** 2

    # Penalize various inequality constraints
    if x1 < 0:
        penalties += rho * (x1) ** 2
    if x2 < 0:
        penalties += rho * (x2) ** 2
    if x1 > 40:
        penalties += rho * (x1 - 40) ** 2
    
    return penalties

def augmented_objective(x, rho=1000.0):
    return -objective(x) + penalty(x, rho)

def gradient_descent(f, x0, lr=0.01, max_iter=1000, tol=1e-6):
    x = np.array(x0, dtype=float)
    for i in range(max_iter):
        grad = np.zeros_like(x)
        fx = f(x)

        eps = 1e-6
        for j in range(len(x)):
            # Finite difference derivative approx
            # Done in lieu of exact automatic differentiation
            x_eps = x.copy()
            x_eps[j] += eps
            grad[j] = (f(x_eps) - fx) / eps
        
        x_new = x - lr * grad
        if np.linalg.norm(x_new - x) < tol:
            break
        x = x_new

    return x, -objective(x), i+1
    
# Run solution
x_opt, area, iters = gradient_descent(augmented_objective, [10, 10])
print(f"Optimal x: {x_opt}")
print(f"Max area: {area}")
print(f"Iterations: {iters}")