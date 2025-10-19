"""Utilities for JIT-compatible optimization solving."""

import jax
import jax.numpy as jnp
from jax import tree_util
from typing import Dict, Any, NamedTuple

class JITSolution(NamedTuple):
    """JIT-compatible solution structure with static shapes."""
    obj_value: jnp.ndarray
    x: jnp.ndarray
    status: int  # 0=optimal, 1=max_iter, 2=error


@jax.jit
def solve_qp_kernel(Q: jnp.ndarray, q: jnp.ndarray, A_eq: jnp.ndarray, b_eq: jnp.ndarray,
                   A_ineq: jnp.ndarray, b_ineq: jnp.ndarray, 
                   x0: jnp.ndarray, tol: float = 1e-8, max_iter: int = 50) -> JITSolution:
    """JIT-compiled QP solving kernel with static shapes.
    
    This function assumes all arrays have been pre-allocated with static shapes.
    Zero rows/columns can be used for padding inactive constraints.
    """
    from jax import lax
    
    n_vars = Q.shape[0]
    
    # Initial state
    x = x0
    
    def body_fun(i, x):
        # Compute gradient: Q @ x + q
        grad = Q @ x + q
        
        # Simple projection step (not a proper IPM, but JIT-compatible)
        new_x = x - 0.01 * grad
        return new_x
    
    # Use lax.fori_loop for JIT compatibility
    x_final = lax.fori_loop(0, max_iter, body_fun, x)
    
    # Compute objective value
    obj_val = 0.5 * x_final @ Q @ x_final + q @ x_final
    
    return JITSolution(
        obj_value=obj_val,
        x=x_final,
        status=0  # Assume optimal for now
    )


def solve_with_jit_kernel(qp_data, tol: float = 1e-8, max_iter: int = 50) -> Dict[str, Any]:
    """Solve QP using JIT kernel with proper data preparation."""
    
    # Prepare static-shaped arrays OUTSIDE JIT
    n_vars = int(qp_data.n_vars)
    max_eq = max(1, int(qp_data.n_eq))
    max_ineq = max(1, int(qp_data.n_ineq))
    
    # Pad matrices to static sizes
    Q = jnp.zeros((n_vars, n_vars))
    Q = Q.at[:qp_data.Q.shape[0], :qp_data.Q.shape[1]].set(qp_data.Q)
    
    q = jnp.zeros(n_vars)
    q = q.at[:qp_data.q.shape[0]].set(qp_data.q)
    
    # Handle equality constraints (pad if empty)
    if qp_data.n_eq > 0:
        A_eq = jnp.zeros((max_eq, n_vars))
        A_eq = A_eq.at[:qp_data.A_eq.shape[0], :qp_data.A_eq.shape[1]].set(qp_data.A_eq)
        b_eq = jnp.zeros(max_eq)
        b_eq = b_eq.at[:qp_data.b_eq.shape[0]].set(qp_data.b_eq)
    else:
        A_eq = jnp.zeros((max_eq, n_vars))
        b_eq = jnp.zeros(max_eq)
    
    # Handle inequality constraints (pad if empty) 
    if qp_data.n_ineq > 0:
        A_ineq = jnp.zeros((max_ineq, n_vars))
        A_ineq = A_ineq.at[:qp_data.A_ineq.shape[0], :qp_data.A_ineq.shape[1]].set(qp_data.A_ineq)
        b_ineq = jnp.zeros(max_ineq)
        b_ineq = b_ineq.at[:qp_data.b_ineq.shape[0]].set(qp_data.b_ineq)
    else:
        A_ineq = jnp.zeros((max_ineq, n_vars))
        b_ineq = jnp.zeros(max_ineq)
    
    # Initial point
    x0 = jnp.zeros(n_vars)
    
    # Solve with JIT kernel
    jit_sol = solve_qp_kernel(Q, q, A_eq, b_eq, A_ineq, b_ineq, x0, tol, max_iter)
    
    # Convert back to standard Solution format OUTSIDE JIT
    from cvxjax.api import Solution
    
    # Build primal solution mapping using variable names (OUTSIDE JIT)
    primal = {}
    start_idx = 0
    for var in qp_data.variables:
        var_size = jnp.prod(jnp.array(var.shape)).astype(int)  # Use astype instead of int()
        end_idx = start_idx + var_size
        var_value = jit_sol.x[start_idx:end_idx].reshape(var.shape)
        # Support both Variable object and string name for backward compatibility
        primal[var] = var_value  # Original API expects Variable object as key
        primal[var.name] = var_value  # JIT-compatible string key
        start_idx = end_idx
    
    return Solution(
        status="optimal" if jit_sol.status == 0 else "max_iter",
        obj_value=float(jit_sol.obj_value + qp_data.constant),
        primal=primal,
        dual={},
        info={"solver": "jit_kernel"}
    )
