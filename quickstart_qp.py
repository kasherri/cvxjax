#!/usr/bin/env python3
"""Quickstart example: Simple quadratic programming with CVXJAX.

This example demonstrates the basic usage of CVXJAX for solving a simple
quadratic program with affine constraints.

Problem:
    minimize    (1/2) x^T Q x + q^T x
    subject to  A x = b
               x >= 0

Where:
    Q = [[2, 1], [1, 2]]  (positive definite)
    q = [1, 1]
    A = [[1, 1]]          (budget constraint)
    b = [1]
    x >= 0                (non-negativity)

Expected solution: x ≈ [0.25, 0.25], obj ≈ 0.75
"""

import jax
import jax.numpy as jnp

import cvxjax as cx


def main():
    """Run the quickstart example."""
    # Enable 64-bit precision for numerical stability
    jax.config.update("jax_enable_x64", True)
    
    print("CVXJAX Quickstart Example")
    print("=" * 40)
    
    # Problem data
    Q = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    q = jnp.array([1.0, 1.0])
    
    print(f"Q matrix:\n{Q}")
    print(f"q vector: {q}")
    print()
    
    # Define optimization variable
    x = cx.Variable(shape=(2,), name="x")
    print(f"Variable x: shape={x.shape}, name='{x.name}'")
    
    # Define quadratic objective: (1/2) x^T Q x + q^T x
    objective = cx.Minimize(0.5 * cx.quad_form(x, Q) + q @ x)
    print("Objective: minimize (1/2) x^T Q x + q^T x")
    
    # Define constraints
    constraints = [
        jnp.ones(2) @ x == 1.0,  # Budget constraint: x1 + x2 = 1
        x >= 0.0,                # Non-negativity: x >= 0
    ]
    print("Constraints:")
    print("  x1 + x2 = 1  (budget constraint)")
    print("  x >= 0       (non-negativity)")
    print()
    
    # Create optimization problem
    problem = cx.Problem(objective, constraints)
    print("Problem created successfully")
    
    # Solve with interior point method
    print("Solving with IPM solver...")
    solution = problem.solve(solver="ipm", tol=1e-8, max_iter=50)
    
    # Display results
    print(f"Status: {solution.status}")
    print(f"Optimal objective value: {solution.obj_value:.6f}")
    print(f"Optimal solution x: {solution.primal[x].flatten()}")
    print(f"Iterations: {solution.info.get('iterations', 'N/A')}")
    print(f"Primal residual: {solution.info.get('primal_residual', 'N/A'):.2e}")
    print(f"Dual residual: {solution.info.get('dual_residual', 'N/A'):.2e}")
    print()
    
    # Verify solution
    x_opt = solution.primal[x].flatten()
    
    print("Solution verification:")
    print(f"  Budget constraint: x1 + x2 = {jnp.sum(x_opt):.6f} (should be 1.0)")
    print(f"  Non-negativity: x >= 0? {jnp.all(x_opt >= -1e-6)}")
    
    # Compute objective manually
    obj_manual = 0.5 * x_opt @ Q @ x_opt + q @ x_opt
    print(f"  Objective (manual): {obj_manual:.6f}")
    print(f"  Objective (solver): {solution.obj_value:.6f}")
    print(f"  Difference: {abs(obj_manual - solution.obj_value):.2e}")
    print()
    
    # Try JIT-compiled solve
    print("Testing JIT compilation...")
    try:
        solution_jit = problem.solve_jit(solver="ipm", tol=1e-8)
        print(f"JIT solve status: {solution_jit.status}")
        print(f"JIT objective: {solution_jit.obj_value:.6f}")
        
        # Compare solutions
        x_diff = jnp.linalg.norm(solution.primal[x] - solution_jit.primal[x])
        print(f"Solution difference (JIT vs normal): {x_diff:.2e}")
    except Exception as e:
        print(f"JIT compilation failed: {e}")
    
    print()
    print("Quickstart example completed!")


if __name__ == "__main__":
    main()
