# Linear Programming
import numpy as np

# Simplex algorithm via pivotting & tableau

def simplex(c, A, b, max_iter=100):
    m, n = A.shape

    # Add slack variables by sticking an I matrix with A
    A_slack = np.hstack([A, np.eye(m)])
    c_slack = np.hstack([c, np.zeros(m)])

    # Variable of solution
    B = list(range(n, n+m))
    N = list(range(n))

    # Init m+1 by n+m+1 tableau matrix
    tableau = np.zeros((m+1, n+m+1))

    tableau[:-1, :-1] = A_slack # place A in all but last row, column
    tableau[:-1, -1] = b # placing b on last column except last element
    tableau[-1, :-1] = -c_slack # last row except its last element

    for _ in range(max_iter):
        # Optimality condition
        if np.all(tableau[-1, :-1] >= 0):
            break

        pivot_col = np.argmin(tableau[-1, :-1])

        if np.all(tableau[:-1, pivot_col] <= 0):
            raise ValueError("Linear Program Unbounded")
    
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[tableau[:-1, pivot_col] <= 0] = np.inf
        pivot_row = np.argmin(ratios)

        # Pivot operation formula
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]

        for i in range(m+1):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]

        B[pivot_row] = pivot_col
    
    x = np.zeros(n + m)
    for i in range(m):
        x[B[i]] = tableau[i, -1]
    return x[:n], tableau[-1, -1] 

if __name__ == "__main__":
    # Maximize: 3x + 2y
    # Subject to:
    """
    x + 2y <= 4
    3x + y <= 5
    x, y >= 0
    """

    c = np.array([3, 2]) # objective function

    # Constraint of convex polyhedron
    A = np.array([[1, 2],
                  [3, 1]])
    
    b = np.array([4, 5])

    x_opt, max_val = simplex(c, A, b)
    print("Optimal x:", x_opt)
    print(f"Maximum value: {max_val:.4f}")
