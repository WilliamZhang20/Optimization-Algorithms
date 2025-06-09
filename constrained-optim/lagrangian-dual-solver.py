import numpy as np
from scipy.optimize import minimize

# Lagrangian dual variables: λ1, λ2, λ3 for constraints
# Constraint 1: -2x - y ≤ -8   (Protein)
# Constraint 2:  x + y = 4     (Fat, equality → ν)
# Constraint 3: -3x - 2y ≤ -10 (Carbs)

def dual_objective(lam):
    λ1, λ3, ν = lam

    # Objective coefficients:
    # L(x, y, λ1, ν, λ3) = (2 + 2λ1 - ν + 3λ3)x + (1 + λ1 - ν + 2λ3)y -8λ1 - 10λ3 + 4ν
    # At minimum, gradient wrt x and y = 0
    # Solve for x,y such that ∂L/∂x = ∂L/∂y = 0
    cx = 2 + 2*λ1 - ν + 3*λ3
    cy = 1 + λ1 - ν + 2*λ3

    # If gradient conditions can't be satisfied, return +∞ (invalid dual point)
    if not np.isclose(cx, 0) or not np.isclose(cy, 0):
        return 1e6  # penalty

    # Otherwise, valid dual function value (infimum over x,y is L evaluated at min)
    return -8*λ1 - 10*λ3 + 4*ν  # This is g(λ, ν), which we **maximize**

# Maximize g(λ, ν) subject to λ1 ≥ 0, λ3 ≥ 0
bounds = [(0, None),  # λ1 ≥ 0
          (0, None),  # λ3 ≥ 0
          (None, None)]  # ν unrestricted

res = minimize(lambda lam: -dual_objective(lam), x0=[1, 1, 0], bounds=bounds)

λ1_opt, λ3_opt, ν_opt = res.x
dual_val = -res.fun

print(f"Best dual value: {dual_val:.4f}")
print(f"Optimal dual variables: λ1 = {λ1_opt:.4f}, λ3 = {λ3_opt:.4f}, ν = {ν_opt:.4f}")

# === Recover primal variables using KKT stationarity ===
# ∂L/∂x = 2 + 2λ1 - ν + 3λ3 = 0
# ∂L/∂y = 1 + λ1 - ν + 2λ3 = 0

# Solve the linear system:
# From ∂L/∂x = 0: ν = 2 + 2λ1 + 3λ3
# From ∂L/∂y = 0: ν = 1 + λ1 + 2λ3

# Solve to get x, y from:
# constraint: x + y = 4
# and e.g., 2x + y = 8 (active protein constraint)
# and 3x + 2y ≥ 10 (check complementarity later)

A = np.array([[1, 1],
              [2, 1]])
b = np.array([4, 8])
x_opt = np.linalg.solve(A, b)
x_val, y_val = x_opt

print("\n=== Primal Recovery via KKT ===")
print(f"Recovered x = {x_val:.4f}, y = {y_val:.4f}")
cost = 2*x_val + y_val
print(f"Total cost = {cost:.4f}")
print(f"Protein: {2*x_val + y_val:.4f} (≥8)")
print(f"Fat: {x_val + y_val:.4f} (=4)")
print(f"Carbs: {3*x_val + 2*y_val:.4f} (≥10)")

# === Complementary Slackness Check ===
print("\n=== Complementary Slackness ===")
def f1(x, y): return -2*x - y + 8  # Should be 0 if λ1 > 0
def f3(x, y): return -3*x - 2*y + 10  # Should be 0 if λ3 > 0

print(f"λ1 * f1(x): {λ1_opt * f1(x_val, y_val):.6f}")
print(f"λ3 * f3(x): {λ3_opt * f3(x_val, y_val):.6f}")
