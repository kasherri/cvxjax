"""Simplified JIT-compatible IPM QP solver."""

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
def solve_qp_ipm_simple(
    Q: jnp.ndarray,
    q: jnp.ndarray,
    A_eq: jnp.ndarray,
    b_eq: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    tol: float = 1e-8,
    max_iter: int = 30,
) -> IPMSolution:
    """Simple JIT-compatible IPM for box-constrained QP.
    
    Solves:
        minimize    (1/2) x^T Q x + q^T x
        subject to  A_eq x = b_eq
                   lb <= x <= ub
    
    Uses log-barrier method for bounds.
    """
    n_vars = Q.shape[0]
    n_eq = A_eq.shape[0]
    
    # Initialize at feasible point
    x = jnp.maximum(lb + 0.1, jnp.minimum(ub - 0.1, jnp.ones(n_vars) * 0.5))
    
    # If there are equality constraints, project to feasible subspace
    if n_eq > 0:
        # Solve A_eq @ x = b_eq using least squares
        x_ls = jnp.linalg.lstsq(A_eq, b_eq, rcond=1e-12)[0]
        # Ensure bounds are satisfied
        x = jnp.maximum(lb + 1e-6, jnp.minimum(ub - 1e-6, x_ls))
    
    # Log-barrier parameter
    t = 1.0
    
    def barrier_iteration(carry):
        x, t, iteration = carry
        
        # Barrier objective: t * f(x) + phi(x) where phi is log barrier
        # f(x) = (1/2) x^T Q x + q^T x
        # phi(x) = -sum(log(x - lb)) - sum(log(ub - x))
        
        # Barrier gradient
        grad_f = Q @ x + q
        
        # Barrier terms (avoid log(0) with small epsilon)
        eps = 1e-8
        lb_term = -1.0 / jnp.maximum(x - lb, eps)
        ub_term = 1.0 / jnp.maximum(ub - x, eps)
        barrier_grad = lb_term + ub_term
        
        total_grad = t * grad_f + barrier_grad
        
        # Barrier Hessian
        barrier_hess = jnp.diag(1.0 / jnp.maximum((x - lb)**2, eps**2) + 1.0 / jnp.maximum((ub - x)**2, eps**2))
        total_hess = t * Q + barrier_hess
        
        if n_eq > 0:
            # Newton system with equality constraints
            # [H   A^T] [dx]   [-g]
            # [A   0  ] [dy] = [-r]
            eq_residual = A_eq @ x - b_eq
            
            kkt_matrix = jnp.block([
                [total_hess, A_eq.T],
                [A_eq, jnp.zeros((n_eq, n_eq))]
            ])
            rhs = jnp.concatenate([-total_grad, -eq_residual])
            
            # Solve with regularization if needed
            try:
                solution = jnp.linalg.solve(kkt_matrix, rhs)
            except:
                # Add regularization
                kkt_reg = kkt_matrix + 1e-8 * jnp.eye(kkt_matrix.shape[0])
                solution = jnp.linalg.solve(kkt_reg, rhs)
                
            dx = solution[:n_vars]
        else:
            # Unconstrained Newton step
            dx = -jnp.linalg.solve(total_hess + 1e-8 * jnp.eye(n_vars), total_grad)
        
        # Line search to maintain feasibility
        alpha = 1.0
        
        # Ensure step stays within bounds
        for _ in range(10):  # Max 10 backtracking steps
            x_new = x + alpha * dx
            
            # Check bounds
            feasible = jnp.all(jnp.logical_and(x_new > lb + eps, x_new < ub - eps))
            
            if feasible:
                break
            alpha *= 0.5
        
        x_new = x + alpha * dx
        
        # Increase barrier parameter
        t_new = t * 2.0
        
        return x_new, t_new, iteration + 1
    
    def continue_condition(carry):
        x, t, iteration = carry
        
        # Check convergence
        grad_f = Q @ x + q
        
        if n_eq > 0:
            eq_residual = jnp.linalg.norm(A_eq @ x - b_eq)
        else:
            eq_residual = 0.0
        
        grad_norm = jnp.linalg.norm(grad_f)
        
        # Convergence criteria
        converged = jnp.logical_and(grad_norm < tol, eq_residual < tol)
        
        return jnp.logical_and(iteration < max_iter, jnp.logical_not(converged))
    
    # Run barrier method iterations
    init_carry = (x, t, jnp.array(0))
    final_carry = jax.lax.while_loop(continue_condition, barrier_iteration, init_carry)
    
    x_opt, t_final, final_iter = final_carry
    
    # Compute final residuals
    grad_f = Q @ x_opt + q
    if n_eq > 0:
        eq_residual = jnp.linalg.norm(A_eq @ x_opt - b_eq)
    else:
        eq_residual = 0.0
    
    dual_res = jnp.linalg.norm(grad_f)
    primal_res = eq_residual
    
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


def test_simple_ipm():
    """Test the simple JIT-compatible IPM solver."""
    print("🔧 TESTING SIMPLIFIED JIT-COMPATIBLE IPM SOLVER")
    print("=" * 55)
    
    # Create test problem - simpler without general inequalities
    n = 4
    Q = jnp.array([[2.0, 0.5, 0.0, 0.0],
                   [0.5, 1.0, 0.0, 0.0], 
                   [0.0, 0.0, 1.5, 0.2],
                   [0.0, 0.0, 0.2, 1.0]], dtype=jnp.float32)
    q = jnp.array([1.0, -2.0, 0.5, -1.0], dtype=jnp.float32)
    
    # Equality constraint: x[0] + x[1] = 1.0
    A_eq = jnp.array([[1.0, 1.0, 0.0, 0.0]], dtype=jnp.float32)
    b_eq = jnp.array([1.0], dtype=jnp.float32)
    
    # Box constraints: 0 <= x <= 10
    lb = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    ub = jnp.array([10.0, 10.0, 10.0, 10.0], dtype=jnp.float32)
    
    print("Test problem:")
    print(f"- Variables: {n}")
    print(f"- Equality constraints: {A_eq.shape[0]}")
    print(f"- Box constraints: 0 <= x <= 10")
    print()
    
    return Q, q, A_eq, b_eq, lb, ub


if __name__ == "__main__":
    # Test the solver
    Q, q, A_eq, b_eq, lb, ub = test_simple_ipm()
    
    print('🚀 Testing simplified IPM solver...')
    print()
    
    # Test compilation and first solve
    print('First call (with JIT compilation):')
    start_time = time.time()
    try:
        solution = solve_qp_ipm_simple(Q, q, A_eq, b_eq, lb, ub, tol=1e-6, max_iter=30)
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
        bound_violations = jnp.sum(jnp.logical_or(x_opt < lb - 1e-6, x_opt > ub + 1e-6))
        print(f'Bound violations: {bound_violations} (should be 0)')
        print(f'x values: {x_opt}')
        print(f'Lower bounds: {lb}')
        print(f'Upper bounds: {ub}')
        
        # Test second call (should be much faster)
        print()
        print('Second call (no compilation):')
        start_time = time.time()
        solution2 = solve_qp_ipm_simple(Q, q, A_eq, b_eq, lb, ub, tol=1e-6, max_iter=30)
        second_time = time.time() - start_time
        
        print(f'✅ Completed in {second_time:.4f} seconds')
        speedup = first_time / second_time if second_time > 0 else float('inf')
        print(f'🚀 Speedup: {speedup:.1f}x faster!')
        
    except Exception as e:
        print(f'❌ Failed: {e}')
        import traceback
        traceback.print_exc()