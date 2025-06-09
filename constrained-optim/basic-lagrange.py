# Basic constrained optimization, fewest solvers as possible
import numpy
from scipy.optimize import fsolve # outsourced "systems of equations solving"

"""
Example: maximize area of rectangle subject to perimeter
Perimeter constrained to be 100
Sides of rectangle denoted as x, y
"""

# The lagrangian is L(x, y, λ) = xy - λ(2x + 2y - 100)
def equations(vars):
    x, y, lam = vars
    eq1 = y - 2 * lam # from gradient of lagrangian of objective & constraint
    eq2 = x - 2 * lam
    eq3 = 2 * x + 2 * y - 100 # derivative wrt λ
    return [eq1, eq2, eq3]

initial_guess = [10, 10, 1]

solution = fsolve(equations, initial_guess)
x_opt, y_opt, lam_opt = solution

area = x_opt * y_opt

print(f"Optimal x: {x_opt:.2f}")
print(f"Optimal y: {y_opt:.2f}")
print(f"Lagrange multiplier λ: {lam_opt:.2f}")
print(f"Maximum area: {area:.2f}")