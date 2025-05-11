# The levenberg-marquadrt Algorithm applicable to nonlinear least squares
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
x_data = np.linspace(0, 2, 50)
true_a, true_b = 2.5, -1.3
y_data = true_a * np.exp(true_b * x_data) + 0.1 * np.random.randn(*x_data.shape)

# Model function and Jacobian
def model(x, params):
    a, b = params
    return a * np.exp(b * x)

def jacobian(x, params):
    a, b = params
    J = np.zeros((x.size, 2))
    exp_bx = np.exp(b * x)
    J[:, 0] = exp_bx
    J[:, 1] = a * x * exp_bx
    return J

# Levenberg-Marquardt algorithm
def levenberg_marquardt(x, y, initial_params, max_iters=100, tol=1e-6, lambda_init=1e-3):
    params = np.array(initial_params, dtype=np.float64)
    lam = lambda_init

    for i in range(max_iters):
        y_pred = model(x, params)
        residuals = y - y_pred
        J = jacobian(x, params)
        JTJ = J.T @ J
        JTr = J.T @ residuals

        # Levenberg-Marquardt update (JTJ + λI) Δ = J^T r
        A = JTJ + lam * np.eye(JTJ.shape[0])
        delta = np.linalg.solve(A, JTr)
        new_params = params + delta
        new_residuals = y - model(x, new_params)

        # Check if the update improved the cost
        if np.sum(new_residuals**2) < np.sum(residuals**2):
            params = new_params
            lam *= 0.7  # decrease lambda (move toward Gauss-Newton)
        else:
            lam *= 2.0  # increase lambda (move toward gradient descent)

        if np.linalg.norm(delta) < tol:
            print(f"Converged at iteration {i}")
            break

    return params

# Run optimization
initial_guess = [1.0, -0.5]
estimated_params = levenberg_marquardt(x_data, y_data, initial_guess)

print("Estimated parameters (LM):", estimated_params)

# Plot the result
plt.scatter(x_data, y_data, label="Noisy Data")
plt.plot(x_data, model(x_data, estimated_params), label="Levenberg-Marquardt Fit", color='red')
plt.plot(x_data, model(x_data, [true_a, true_b]), label="True Model", linestyle='--')
plt.legend()
plt.title("Levenberg-Marquardt Fit to Exponential Model")
plt.show()