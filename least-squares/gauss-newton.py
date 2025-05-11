# Gauss-Newton for nonlinear least squares
import numpy as np
import matplotlib.pyplot as plt

# Function to fit: y = a*e^(bx)

np.random.seed(0)
x_data = np.linspace(0, 2, 50) # generate sample sequence for indep var
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
    J[:, 0] = exp_bx           # ∂f/∂a = e^(bx)
    J[:, 1] = a * x * exp_bx   # ∂f/∂b = a·x·e^(bx)
    return J

# Gauss-Newton algorithm
def gauss_newton(x, y, initial_params, max_iters=10, tol=1e-6):
    params = np.array(initial_params, dtype=np.float64)

    for i in range(max_iters):
        residuals = y - model(x, params)
        J = jacobian(x, params)
        
        # Solve normal equations J^T J Δ = J^T r
        delta = np.linalg.lstsq(J, residuals, rcond=None)[0]
        params += delta

        if np.linalg.norm(delta) < tol:
            print(f"Converged at iteration {i}")
            break

    return params

# Run optimization
initial_guess = [1.0, -0.5]
estimated_params = gauss_newton(x_data, y_data, initial_guess)

print("Estimated parameters:", estimated_params)

# Plot the fit
plt.scatter(x_data, y_data, label="Noisy Data")
plt.plot(x_data, model(x_data, estimated_params), label="Fitted Model", color='red')
plt.plot(x_data, model(x_data, [true_a, true_b]), label="True Model", linestyle='--')
plt.legend()
plt.title("Gauss-Newton Fit to Exponential Model")
plt.show()