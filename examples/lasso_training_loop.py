#!/usr/bin/env python3
"""LASSO training loop with CVXJAX differentiation.

This example demonstrates how to use CVXJAX for differentiable optimization
in a machine learning context. We implement LASSO regression and show how
to compute gradients of the optimal objective value w.r.t. the regularization
parameter.

LASSO Problem:
    minimize    (1/2) ||Ax - b||^2 + lambda ||x||_1
    
We approximate the L1 penalty with a smooth quadratic penalty for 
differentiability:
    minimize    (1/2) ||Ax - b||^2 + lambda (1/2) ||x||^2
    
This becomes a quadratic program that we can differentiate through.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import cvxjax as cx


def generate_lasso_data(n_samples: int = 50, n_features: int = 20, noise_std: float = 0.1, seed: int = 42):
    """Generate synthetic data for LASSO regression.
    
    Args:
        n_samples: Number of data samples.
        n_features: Number of features.
        noise_std: Standard deviation of noise.
        seed: Random seed.
        
    Returns:
        Tuple of (A, b, x_true) where Ax_true + noise = b.
    """
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Generate design matrix
    A = jax.random.normal(key1, (n_samples, n_features))
    
    # Generate sparse true coefficients
    x_true = jnp.zeros(n_features)
    sparse_indices = jnp.array([2, 5, 8, 12, 15])  # Only 5 non-zero coefficients
    x_true = x_true.at[sparse_indices].set(jax.random.normal(key2, (len(sparse_indices),)))
    
    # Generate noisy observations
    noise = noise_std * jax.random.normal(key3, (n_samples,))
    b = A @ x_true + noise
    
    return A, b, x_true


def solve_lasso_qp(A: jnp.ndarray, b: jnp.ndarray, lam: float) -> cx.Solution:
    """Solve LASSO as a quadratic program.
    
    Args:
        A: Design matrix (n_samples x n_features).
        b: Target vector (n_samples,).
        lam: Regularization parameter.
        
    Returns:
        Solution object.
    """
    n_features = A.shape[1]
    
    # Define variable
    x = cx.Variable(shape=(n_features,), name="coefficients")
    
    # Objective: (1/2) ||Ax - b||^2 + (lambda/2) ||x||^2
    residual = A @ x - b
    data_fit = 0.5 * cx.sum_squares(residual)
    regularization = 0.5 * lam * cx.sum_squares(x)
    objective = cx.Minimize(data_fit + regularization)
    
    # No constraints (unconstrained LASSO)
    constraints = []
    
    # Create and solve problem
    problem = cx.Problem(objective, constraints)
    solution = problem.solve_jit(solver="ipm", tol=1e-8)
    
    return solution


def lasso_objective_value(A: jnp.ndarray, b: jnp.ndarray, lam: float) -> float:
    """Compute LASSO objective value for differentiation.
    
    This function wraps the QP solve to return just the objective value,
    enabling differentiation w.r.t. lambda.
    """
    solution = solve_lasso_qp(A, b, lam)
    return solution.obj_value


def main():
    """Run LASSO training loop example."""
    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)
    
    print("LASSO Training Loop with CVXJAX")
    print("=" * 40)
    
    # Generate synthetic data
    A, b, x_true = generate_lasso_data(n_samples=50, n_features=20, noise_std=0.1)
    
    print(f"Data shape: A = {A.shape}, b = {b.shape}")
    print(f"True coefficients (non-zero): {jnp.sum(x_true != 0)} out of {len(x_true)}")
    print(f"True coefficient norm: ||x_true||_2 = {jnp.linalg.norm(x_true):.4f}")
    print()
    
    # Test LASSO solve for different lambda values
    lambda_values = jnp.logspace(-3, 1, 10)  # From 0.001 to 10
    
    print("Solving LASSO for different regularization parameters...")
    solutions = []
    objective_values = []
    
    for i, lam in enumerate(lambda_values):
        solution = solve_lasso_qp(A, b, lam)
        solutions.append(solution)
        objective_values.append(solution.obj_value)
        
        x_opt = solution.primal[list(solution.primal.keys())[0]]
        sparsity = jnp.sum(jnp.abs(x_opt) > 1e-4)
        
        print(f"  λ = {lam:.4f}: obj = {solution.obj_value:.4f}, "
              f"||x||_2 = {jnp.linalg.norm(x_opt):.4f}, "
              f"sparsity = {sparsity}/{len(x_opt)}")
    
    print()
    
    # Demonstrate differentiation w.r.t. lambda
    print("Computing gradients w.r.t. regularization parameter...")
    
    try:
        # Define objective function for differentiation
        objective_fn = lambda lam: lasso_objective_value(A, b, lam)
        
        # Compute gradient function
        grad_fn = jax.grad(objective_fn)
        
        # Test gradient at a specific lambda
        lam_test = 0.1
        grad_lam = grad_fn(lam_test)
        
        print(f"At λ = {lam_test:.3f}:")
        print(f"  Objective value: {objective_fn(lam_test):.6f}")
        print(f"  Gradient d(obj)/d(λ): {grad_lam:.6f}")
        
        # Verify gradient with finite differences
        eps = 1e-6
        grad_fd = (objective_fn(lam_test + eps) - objective_fn(lam_test - eps)) / (2 * eps)
        print(f"  Finite difference gradient: {grad_fd:.6f}")
        print(f"  Relative error: {abs(grad_lam - grad_fd) / abs(grad_fd):.2e}")
        
    except Exception as e:
        print(f"Differentiation failed (expected if not implemented): {e}")
    
    print()
    
    # Gradient-based hyperparameter optimization
    print("Demonstration: Gradient-based lambda optimization...")
    
    try:
        # Define a validation loss (simplified)
        def validation_loss(lam):
            """Validation loss as function of lambda (simplified)."""
            solution = solve_lasso_qp(A, b, lam)
            x_opt = solution.primal[list(solution.primal.keys())[0]]
            
            # Penalty for deviation from true solution (in practice, use validation set)
            return jnp.linalg.norm(x_opt - x_true) ** 2
        
        # Gradient-based optimization of lambda
        lam_current = 0.5
        step_size = 0.01
        
        print(f"Initial λ: {lam_current:.4f}")
        
        for step in range(3):  # Just a few steps for demonstration
            loss = validation_loss(lam_current)
            grad = jax.grad(validation_loss)(lam_current)
            
            print(f"  Step {step}: λ = {lam_current:.4f}, loss = {loss:.4f}, grad = {grad:.4f}")
            
            # Gradient descent step
            lam_current = lam_current - step_size * grad
            lam_current = jnp.maximum(lam_current, 1e-4)  # Keep positive
        
        print(f"Final λ: {lam_current:.4f}")
        
    except Exception as e:
        print(f"Gradient-based optimization failed: {e}")
    
    print()
    
    # Compare solutions at different lambda values
    print("Solution analysis:")
    
    # Low regularization
    solution_low = solve_lasso_qp(A, b, 0.01)
    x_low = solution_low.primal[list(solution_low.primal.keys())[0]]
    
    # High regularization  
    solution_high = solve_lasso_qp(A, b, 1.0)
    x_high = solution_high.primal[list(solution_high.primal.keys())[0]]
    
    print(f"Low regularization (λ=0.01):")
    print(f"  ||x||_2 = {jnp.linalg.norm(x_low):.4f}")
    print(f"  Distance to true: {jnp.linalg.norm(x_low - x_true):.4f}")
    print(f"  Non-zero coefficients: {jnp.sum(jnp.abs(x_low) > 1e-4)}")
    
    print(f"High regularization (λ=1.0):")
    print(f"  ||x||_2 = {jnp.linalg.norm(x_high):.4f}")
    print(f"  Distance to true: {jnp.linalg.norm(x_high - x_true):.4f}")
    print(f"  Non-zero coefficients: {jnp.sum(jnp.abs(x_high) > 1e-4)}")
    
    print()
    print("LASSO training loop example completed!")
    
    # Optional: Create plots if matplotlib is available
    try:
        create_plots(lambda_values, objective_values, A, b, x_true)
    except Exception:
        print("Plotting skipped (matplotlib may not be available)")


def create_plots(lambda_values, objective_values, A, b, x_true):
    """Create visualization plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Objective value vs lambda
    ax1.semilogx(lambda_values, objective_values, 'bo-')
    ax1.set_xlabel('Regularization parameter λ')
    ax1.set_ylabel('Objective value')
    ax1.set_title('LASSO Objective vs Regularization')
    ax1.grid(True)
    
    # Plot 2: Solution path
    coeffs_path = []
    for lam in lambda_values:
        solution = solve_lasso_qp(A, b, lam)
        x_opt = solution.primal[list(solution.primal.keys())[0]]
        coeffs_path.append(x_opt)
    
    coeffs_path = jnp.array(coeffs_path)
    
    for i in range(min(5, coeffs_path.shape[1])):  # Plot first 5 coefficients
        ax2.semilogx(lambda_values, coeffs_path[:, i], 'o-', label=f'x[{i}]')
    
    ax2.set_xlabel('Regularization parameter λ')
    ax2.set_ylabel('Coefficient value')
    ax2.set_title('LASSO Solution Path')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('lasso_results.png', dpi=150, bbox_inches='tight')
    print("Plots saved to 'lasso_results.png'")


if __name__ == "__main__":
    main()
