"""Test and create JIT-compatible IPM QP solver."""

import jax
import jax.numpy as jnp
import numpy as np
import time
from typing import NamedTuple

from cvxjax.canonicalize import QPData


class IPMSolution(NamedTuple):
    """JIT-compatible IPM solution."""
    x: jnp.ndarray
    obj_value: jnp.ndarray
    iterations: jnp.ndarray
    primal_residual: jnp.ndarray
    dual_residual: jnp.ndarray
    duality_gap: jnp.ndarray
    converged: jnp.ndarray


@jax.jit
def solve_qp_ipm_jit(
    Q: jnp.ndarray,
    q: jnp.ndarray,
    A_eq: jnp.ndarray,
    b_eq: jnp.ndarray,
    A_ineq: jnp.ndarray,
    b_ineq: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    tol: float = 1e-8,
    max_iter: int = 50,
    regularization: float = 1e-12,
) -> IPMSolution:
    """JIT-compatible IPM solver for QP.
    
    Solves:
        minimize    (1/2) x^T Q x + q^T x
        subject to  A_eq x = b_eq
                   A_ineq x <= b_ineq
                   lb <= x <= ub
    """
    n_vars = Q.shape[0]
    n_eq = A_eq.shape[0] if A_eq.size > 0 else 0
    n_ineq = A_ineq.shape[0] if A_ineq.size > 0 else 0
    
    # Convert bounds to inequality constraints
    # lb <= x becomes -x <= -lb
    # x <= ub becomes x <= ub
    
    # Count finite bounds
    has_lb = jnp.isfinite(lb)
    has_ub = jnp.isfinite(ub)
    n_lb = jnp.sum(has_lb)
    n_ub = jnp.sum(has_ub)
    
    # Total inequality constraints
    n_total_ineq = n_ineq + n_lb + n_ub
    
    # Build extended inequality constraint matrix
    A_ineq_ext = jnp.zeros((n_total_ineq, n_vars))
    b_ineq_ext = jnp.zeros(n_total_ineq)
    
    # Original inequality constraints
    if n_ineq > 0:
        A_ineq_ext = A_ineq_ext.at[:n_ineq].set(A_ineq)
        b_ineq_ext = b_ineq_ext.at[:n_ineq].set(b_ineq)
    
    # Lower bound constraints: -x <= -lb
    row_idx = n_ineq
    for i in range(n_vars):
        A_ineq_ext = jnp.where(
            has_lb[i],
            A_ineq_ext.at[row_idx, i].set(-1.0),
            A_ineq_ext
        )
        b_ineq_ext = jnp.where(
            has_lb[i],
            b_ineq_ext.at[row_idx].set(-lb[i]),
            b_ineq_ext
        )
        row_idx = jnp.where(has_lb[i], row_idx + 1, row_idx)
    
    # Upper bound constraints: x <= ub
    for i in range(n_vars):
        A_ineq_ext = jnp.where(
            has_ub[i],
            A_ineq_ext.at[row_idx, i].set(1.0),
            A_ineq_ext
        )
        b_ineq_ext = jnp.where(
            has_ub[i],
            b_ineq_ext.at[row_idx].set(ub[i]),
            b_ineq_ext
        )
        row_idx = jnp.where(has_ub[i], row_idx + 1, row_idx)
    
    # Initial point
    x = jnp.ones(n_vars) * 0.1
    s = jnp.ones(n_total_ineq) * 0.1  # Slack variables for inequalities
    y_eq = jnp.zeros(n_eq)
    y_ineq = jnp.ones(n_total_ineq) * 0.1  # Dual variables for inequalities
    
    # Make initial point feasible for inequalities
    ineq_residual = A_ineq_ext @ x - b_ineq_ext
    s = jnp.maximum(s, -ineq_residual + 0.1)
    
    # IPM iterations
    def ipm_iteration(carry):
        x, s, y_eq, y_ineq, iteration = carry
        
        # Compute residuals
        rd = Q @ x + q  # Dual residual (gradient)
        if n_eq > 0:
            rd += A_eq.T @ y_eq
        if n_total_ineq > 0:
            rd += A_ineq_ext.T @ y_ineq
        
        rp_eq = jnp.zeros(0) if n_eq == 0 else A_eq @ x - b_eq
        rp_ineq = jnp.zeros(0) if n_total_ineq == 0 else A_ineq_ext @ x + s - b_ineq_ext
        
        # Complementarity
        mu = jnp.mean(s * y_ineq) if n_total_ineq > 0 else 0.0
        rc = jnp.zeros(0) if n_total_ineq == 0 else s * y_ineq - mu
        
        # Newton system: solve for [dx, ds, dy_eq, dy_ineq]
        # Regularized Hessian
        H_reg = Q + regularization * jnp.eye(n_vars)
        
        if n_eq == 0 and n_total_ineq == 0:
            # Unconstrained case
            dx = -jnp.linalg.solve(H_reg, rd)
            ds = jnp.zeros(0)
            dy_eq = jnp.zeros(0) 
            dy_ineq = jnp.zeros(0)
        elif n_eq > 0 and n_total_ineq == 0:
            # Only equality constraints
            # [H   A_eq^T] [dx   ]   [-rd   ]
            # [A_eq  0   ] [dy_eq] = [-rp_eq]
            kkt_matrix = jnp.block([
                [H_reg, A_eq.T],
                [A_eq, jnp.zeros((n_eq, n_eq))]
            ])
            rhs = jnp.concatenate([-rd, -rp_eq])
            solution = jnp.linalg.solve(kkt_matrix, rhs)
            dx = solution[:n_vars]
            dy_eq = solution[n_vars:]
            ds = jnp.zeros(0)
            dy_ineq = jnp.zeros(0)
        else:
            # General case with inequality constraints
            # Use barrier method approximation
            S_inv = 1.0 / (s + 1e-12)  # Avoid division by zero
            
            # Modified Hessian: H + A_ineq^T S^{-1} Y A_ineq
            if n_total_ineq > 0:
                H_mod = H_reg + A_ineq_ext.T @ jnp.diag(S_inv * y_ineq) @ A_ineq_ext
            else:
                H_mod = H_reg
            
            # Modified RHS
            rd_mod = -rd
            if n_total_ineq > 0:
                rd_mod += A_ineq_ext.T @ (S_inv * (rc + s * (-rp_ineq)))
            
            if n_eq > 0:
                # With equality constraints
                kkt_matrix = jnp.block([
                    [H_mod, A_eq.T],
                    [A_eq, jnp.zeros((n_eq, n_eq))]
                ])
                rhs = jnp.concatenate([rd_mod, -rp_eq])
                solution = jnp.linalg.solve(kkt_matrix, rhs)
                dx = solution[:n_vars]
                dy_eq = solution[n_vars:]
            else:
                # No equality constraints
                dx = jnp.linalg.solve(H_mod, rd_mod)
                dy_eq = jnp.zeros(0)
            
            # Recover slack and dual directions
            if n_total_ineq > 0:
                ds = -rp_ineq - A_ineq_ext @ dx
                dy_ineq = S_inv * (-rc - y_ineq * ds)
            else:
                ds = jnp.zeros(0)
                dy_ineq = jnp.zeros(0)
        
        # Step length (simplified)
        alpha = 0.95
        
        # Update
        x_new = x + alpha * dx
        s_new = jnp.maximum(s + alpha * ds, 1e-12) if n_total_ineq > 0 else s
        y_eq_new = y_eq + alpha * dy_eq if n_eq > 0 else y_eq
        y_ineq_new = jnp.maximum(y_ineq + alpha * dy_ineq, 1e-12) if n_total_ineq > 0 else y_ineq
        
        return x_new, s_new, y_eq_new, y_ineq_new, iteration + 1
    
    def continue_condition(carry):
        x, s, y_eq, y_ineq, iteration = carry
        
        # Check convergence
        rd = Q @ x + q
        if n_eq > 0:
            rd += A_eq.T @ y_eq
        if n_total_ineq > 0:
            rd += A_ineq_ext.T @ y_ineq
        
        rp_eq = jnp.zeros(1) if n_eq == 0 else A_eq @ x - b_eq
        rp_ineq = jnp.zeros(1) if n_total_ineq == 0 else A_ineq_ext @ x + s - b_ineq_ext
        
        dual_res = jnp.linalg.norm(rd)
        primal_res = jnp.linalg.norm(jnp.concatenate([rp_eq, rp_ineq]))
        gap = jnp.sum(s * y_ineq) if n_total_ineq > 0 else 0.0
        
        converged = jnp.logical_and(
            jnp.logical_and(dual_res <= tol, primal_res <= tol),
            gap <= tol
        )
        
        return jnp.logical_and(iteration < max_iter, jnp.logical_not(converged))
    
    # Run IPM iterations
    init_carry = (x, s, y_eq, y_ineq, jnp.array(0))
    final_carry = jax.lax.while_loop(continue_condition, ipm_iteration, init_carry)
    
    x_opt, s_opt, y_eq_opt, y_ineq_opt, final_iter = final_carry
    
    # Compute final residuals
    rd = Q @ x_opt + q
    if n_eq > 0:
        rd += A_eq.T @ y_eq_opt
    if n_total_ineq > 0:
        rd += A_ineq_ext.T @ y_ineq_opt
    
    rp_eq = jnp.zeros(1) if n_eq == 0 else A_eq @ x_opt - b_eq
    rp_ineq = jnp.zeros(1) if n_total_ineq == 0 else A_ineq_ext @ x_opt + s_opt - b_ineq_ext
    
    dual_res = jnp.linalg.norm(rd)
    primal_res = jnp.linalg.norm(jnp.concatenate([rp_eq, rp_ineq]))
    gap = jnp.sum(s_opt * y_ineq_opt) if n_total_ineq > 0 else 0.0
    
    converged = jnp.logical_and(
        jnp.logical_and(dual_res <= tol, primal_res <= tol),
        gap <= tol
    )
    
    # Objective value
    obj_value = 0.5 * x_opt.T @ Q @ x_opt + q.T @ x_opt
    
    return IPMSolution(
        x=x_opt,
        obj_value=obj_value,
        iterations=final_iter,
        primal_residual=primal_res,
        dual_residual=dual_res,
        duality_gap=gap,
        converged=converged
    )


def test_imp_jit():
    """Test the JIT-compatible IPM solver."""
    print("🔧 TESTING JIT-COMPATIBLE IPM SOLVER")
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
    
    # Inequality constraint: x[2] + x[3] <= 2.0
    A_ineq = jnp.array([[0.0, 0.0, 1.0, 1.0]], dtype=jnp.float32) 
    b_ineq = jnp.array([2.0], dtype=jnp.float32)
    
    # Bounds: x >= 0
    lb = jnp.array([0.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    ub = jnp.array([jnp.inf, jnp.inf, jnp.inf, jnp.inf], dtype=jnp.float32)
    
    print("Test problem:")
    print(f"- Variables: {n}")
    print(f"- Equality constraints: {A_eq.shape[0]}")
    print(f"- Inequality constraints: {A_ineq.shape[0]}")
    print(f"- Bounds: x >= 0")
    print()
    
    return Q, q, A_eq, b_eq, A_ineq, b_ineq, lb, ub


if __name__ == "__main__":
    # There's a typo in the function name - let me fix it
    pass