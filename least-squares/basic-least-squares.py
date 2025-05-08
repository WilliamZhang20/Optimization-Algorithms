import numpy as np

A = np.array([[1, 1], [1, 2], [2, 2], [2, 3]], dtype=float)
b = np.array([6, 8, 9, 11], dtype=float)

def normal_equation(A, b):
    """
    Using the normal equation (A^T A)^(-1) A^T b
    """
    A_T_A = np.dot(A.T, A)
    A_T_b = np.dot(A.T, b) # a vector result

    # computes the first matrix inverse times the second vector
    x = np.linalg.solve(A_T_A, A_T_b)
    
    return x

def moore_penrose_inverse(A):
    # Using SVD

    U, Sigma, VT = np.linalg.svd(A, full_matrices=False)

    Sigma_plus = np.zeros_like(Sigma)

    # compute PseudoInverse of Sigma
    non_zero_indices = Sigma > 1e-10  # Avoid division by zero
    Sigma_plus[non_zero_indices] = 1 / Sigma[non_zero_indices]

    # Construct Moore-Penrose Inverse
    A_plus = VT.T @ np.diag(Sigma_plus) @ U.T

    return A_plus

A_plus = moore_penrose_inverse(A)
x = A_plus @ b

print("Least Squares Solution via Moore-Penrose:", x)

x = normal_equation(A, b)

print("Least Squares Solution via Normal Equation:", x)