"""Proper JIT-compatible IPM QP solver without Python control flow."""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import NamedTuple


class IPMSolution(NamedTuple):
    """JIT-compatible IPM solution."""
    x: jnp.ndarray
    obj_value: jnp.ndarray
    iterations: jnp.ndarray
    primal_residual: jnp.ndarray
    dual_residual: jnp.ndarray
    converged: jnp.ndarray


@jax.jit
def solve_qp_ipm_jit_clean(
    Q: jnp.ndarray,
    q: jnp.ndarray,
    A_eq: jnp.ndarray,
    b_eq: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    tol: float = 1e-8,
    max_iter: int = 30,
) -> IPMSolution:
    """Clean JIT-compatible IPM for box-constrained QP.
    
    Solves:
        minimize    (1/2) x^T Q x + q^T x
        subject to  A_eq x = b_eq
                   lb <= x <= ub
    
    Uses interior point method with barrier function.
    """
    n_vars = Q.shape[0]
    n_eq = A_eq.shape[0]
    
    # Initialize at feasible point (center of box)
    x = 0.5 * (lb + ub)
    
    # If there are equality constraints, find feasible starting point
    def project_to_equality(x_init):
        # Solve min ||x - x_init||^2 s.t. A_eq @ x = b_eq
        if n_eq > 0:
            # KKT system: [2I  A_eq^T] [x]   [2*x_init]
            #            [A_eq  0   ] [y] = [b_eq     ]
            I = jnp.eye(n_vars)
            kkt = jnp.block([[2*I, A_eq.T], [A_eq, jnp.zeros((n_eq, n_eq))]])
            rhs = jnp.concatenate([2*x_init, b_eq])
            sol = jnp.linalg.solve(kkt, rhs)
            x_proj = sol[:n_vars]
            # Clamp to bounds
            x_proj = jnp.maximum(lb + 1e-3, jnp.minimum(ub - 1e-3, x_proj))
            return x_proj
        else:
            return x_init
    
    x = project_to_equality(x)
    
    # Barrier parameter - starts large and decreases
    mu = 10.0
    
    def newton_iteration(carry):
        x, mu, iteration = carry
        
        # Barrier objective: f(x) - mu * sum(log(x-lb)) - mu * sum(log(ub-x))
        # Gradient: grad_f - mu * sum(1/(x-lb)) + mu * sum(1/(ub-x))
        grad_f = Q @ x + q
        
        # Barrier gradient terms (with safety epsilon)
        eps = 1e-8
        barrier_grad = -mu / jnp.maximum(x - lb, eps) + mu / jnp.maximum(ub - x, eps)
        total_grad = grad_f + barrier_grad
        
        # Barrier Hessian: Q + mu * diag(1/(x-lb)^2 + 1/(ub-x)^2)
        barrier_hess_diag = mu / jnp.maximum((x - lb)**2, eps**2) + mu / jnp.maximum((ub - x)**2, eps**2)
        barrier_hess = jnp.diag(barrier_hess_diag)
        total_hess = Q + barrier_hess
        
        # Newton system
        if n_eq > 0:
            # With equality constraints
            # [H   A^T] [dx]   [-grad]
            # [A   0  ] [dy] = [-res ]
            eq_residual = A_eq @ x - b_eq
            
            kkt_matrix = jnp.block([
                [total_hess, A_eq.T],
                [A_eq, jnp.zeros((n_eq, n_eq))]
            ])
            rhs = jnp.concatenate([-total_grad, -eq_residual])
            
            # Solve with small regularization for numerical stability
            kkt_reg = kkt_matrix + 1e-10 * jnp.eye(kkt_matrix.shape[0])
            solution = jnp.linalg.solve(kkt_reg, rhs)
            dx = solution[:n_vars]
        else:
            # No equality constraints
            dx = -jnp.linalg.solve(total_hess + 1e-10 * jnp.eye(n_vars), total_grad)
        
        # Line search with fixed step size (simplified for JIT)
        alpha = 0.9
        
        # Backtracking to stay feasible (using lax.fori_loop for JIT compatibility)
        def backtrack_step(i, alpha_val):
            x_test = x + alpha_val * dx
            
            # Check if step violates bounds
            lb_violated = jnp.any(x_test <= lb + eps)
            ub_violated = jnp.any(x_test >= ub - eps)
            violated = jnp.logical_or(lb_violated, ub_violated)
            
            # Reduce step size if violated
            new_alpha = jnp.where(violated, alpha_val * 0.5, alpha_val)
            return new_alpha
        
        # Do 5 backtracking steps max
        alpha_final = jax.lax.fori_loop(0, 5, backtrack_step, alpha)
        
        # Update
        x_new = x + alpha_final * dx
        
        # Ensure strict feasibility
        x_new = jnp.maximum(lb + eps, jnp.minimum(ub - eps, x_new))
        
        # Reduce barrier parameter
        mu_new = mu * 0.1
        
        return x_new, mu_new, iteration + 1
    
    def continue_condition(carry):
        x, mu, iteration = carry
        
        # Check convergence
        grad_f = Q @ x + q
        
        # Dual residual (KKT stationarity)
        dual_res = jnp.linalg.norm(grad_f)
        
        # Primal residual (equality constraints)  
        if n_eq > 0:
            primal_res = jnp.linalg.norm(A_eq @ x - b_eq)
        else:
            primal_res = 0.0
        
        # Convergence criteria
        converged = jnp.logical_and(dual_res < tol, primal_res < tol)
        
        return jnp.logical_and(iteration < max_iter, jnp.logical_not(converged))
    
    # Run Newton iterations
    init_carry = (x, mu, jnp.array(0))
    final_carry = jax.lax.while_loop(continue_condition, newton_iteration, init_carry)
    
    x_opt, mu_final, final_iter = final_carry
    
    # Compute final residuals
    grad_f = Q @ x_opt + q
    dual_res = jnp.linalg.norm(grad_f)
    
    if n_eq > 0:
        primal_res = jnp.linalg.norm(A_eq @ x_opt - b_eq)
    else:
        primal_res = 0.0
    
    converged = jnp.logical_and(dual_res <= tol, primal_res <= tol)
    
    # Objective value
    obj_value = 0.5 * x_opt.T @ Q @ x_opt + q.T @ x_opt
    
    return IPMSolution(
        x=x_opt,
        obj_value=obj_value,
        iterations=final_iter,
        primal_residual=primal_res,
        dual_residual=dual_res,
        converged=converged
    )


def test_clean_ipm():
    """Test the clean JIT-compatible IPM solver."""
    print("🔧 TESTING CLEAN JIT-COMPATIBLE IPM SOLVER")
    print("=" * 50)
    
    # Create test problem
    n = 4
    Q = jnp.array([[2.0, 0.5, 0.0, 0.0],
                   [0.5, 1.0, 0.0, 0.0], 
                   [0.0, 0.0, 1.5, 0.2],
                   [0.0, 0.0, 0.2, 1.0]], dtype=jnp.float32)
    q = jnp.array([1.0, -2.0, 0.5, -1.0], dtype=jnp.float32)
    
    # Equality constraint: x[0] + x[1] = 1.0
    A_eq = jnp.array([[1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
    b_eq = jnp.array([1.0], dtype=jnp.float32)
    
    # Box constraints: 0 <= x <= 5
    lb = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    ub = jnp.array([5.0, 5.0, 5.0, 5.0], dtype=jnp.float32)
    
    print("Test problem:")
    print(f"- Variables: {n}")
    print(f"- Equality constraints: {A_eq.shape[0]}")
    print(f"- Box constraints: 0 <= x <= 5")
    print()
    
    return Q, q, A_eq, b_eq, lb, ub


if __name__ == "__main__":
    # Test the solver
    Q, q, A_eq, b_eq, lb, ub = test_clean_ipm()
    
    print('🚀 Testing clean IPM solver...')
    print()
    
    # Test compilation and first solve
    print('First call (with JIT compilation):')
    start_time = time.time()
    try:
        solution = solve_qp_ipm_jit_clean(Q, q, A_eq, b_eq, lb, ub, tol=1e-6, max_iter=50)
        first_time = time.time() - start_time
        
        print(f'✅ Completed in {first_time:.4f} seconds')
        print(f'Solution: x = {solution.x}')
        print(f'Objective: {solution.obj_value:.6f}')
        print(f'Iterations: {solution.iterations}')
        print(f'Converged: {solution.converged}')
        print(f'Primal residual: {solution.primal_residual:.2e}')
        print(f'Dual residual: {solution.dual_residual:.2e}')
        print()
        
        # Verify constraints
        print('🔍 CONSTRAINT VERIFICATION:')
        print('-' * 30)
        
        x_opt = solution.x
        
        # Equality constraint
        eq_residual = jnp.linalg.norm(A_eq @ x_opt - b_eq)
        print(f'Equality constraint residual: {eq_residual:.2e}')
        
        # Box constraints
        lb_violations = jnp.sum(x_opt < lb - 1e-6)
        ub_violations = jnp.sum(x_opt > ub + 1e-6)
        print(f'Lower bound violations: {lb_violations} (should be 0)')
        print(f'Upper bound violations: {ub_violations} (should be 0)')
        print(f'x values: {x_opt}')
        print(f'Lower bounds: {lb}')
        print(f'Upper bounds: {ub}')
        
        # Test second call (should be much faster)
        print()
        print('Second call (no compilation):')
        start_time = time.time()
        solution2 = solve_qp_ipm_jit_clean(Q, q, A_eq, b_eq, lb, ub, tol=1e-6, max_iter=50)
        second_time = time.time() - start_time
        
        print(f'✅ Completed in {second_time:.4f} seconds')
        speedup = first_time / second_time if second_time > 0 else float('inf')
        print(f'🚀 Speedup: {speedup:.1f}x faster!')
        
    except Exception as e:
        print(f'❌ Failed: {e}')
        import traceback
        traceback.print_exc()