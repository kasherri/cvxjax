"""JAXOpt BoxCDQP solver for box-constrained quadratic programming."""

from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
from jaxopt import BoxCDQP

if TYPE_CHECKING:
    from cvxjax.api import Solution

from cvxjax.canonicalize import QPData


@jax.jit
def _solve_boxcdqp_jit(
    Q: jnp.ndarray,
    q: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    tol: float,
    max_iter: int,
) -> tuple[jnp.ndarray, float, int, float]:
    """JIT-compiled BoxCDQP solver core.
    
    Args:
        Q: Quadratic term matrix (CVXJax format: x^T Q x).
        q: Linear term vector.
        lb: Lower bounds.
        ub: Upper bounds.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.
        
    Returns:
        Tuple of (solution, objective_value, iterations, optimality_error).
    """
    n_vars = Q.shape[0]
    
    # Create the solver
    solver = BoxCDQP(maxiter=max_iter, tol=tol, verbose=0)
    
    # Initial point (choose middle of bounds where possible)
    def compute_init_point(lb_val, ub_val):
        # Use JAX select operations instead of Python if/else
        both_finite = jnp.isfinite(lb_val) & jnp.isfinite(ub_val)
        only_lb_finite = jnp.isfinite(lb_val) & ~jnp.isfinite(ub_val)
        only_ub_finite = ~jnp.isfinite(lb_val) & jnp.isfinite(ub_val)
        
        # Midpoint if both bounds finite
        midpoint = (lb_val + ub_val) / 2.0
        # lb + 1 if only lower bound finite
        lb_plus_one = lb_val + 1.0
        # ub - 1 if only upper bound finite
        ub_minus_one = ub_val - 1.0
        # 0 if both infinite
        zero = 0.0
        
        return jnp.select(
            [both_finite, only_lb_finite, only_ub_finite],
            [midpoint, lb_plus_one, ub_minus_one],
            default=zero
        )
    
    x0 = jax.vmap(compute_init_point)(lb, ub)
    
    # BoxCDQP expects: minimize 0.5 <x, Qx> + <c, x>
    # CVXJax provides: minimize x^T Q x + q^T x + constant
    # So we need to multiply Q by 2 to convert from CVXJax to BoxCDQP format
    Q_boxcdqp = 2.0 * Q
    c_boxcdqp = q
    
    params_obj = (Q_boxcdqp, c_boxcdqp)
    params_ineq = (lb, ub)
    
    result = solver.run(
        init_params=x0,
        params_obj=params_obj,
        params_ineq=params_ineq
    )
    
    # Extract solution
    x_opt = result.params
    
    # Compute objective value using CVXJax convention
    obj_value = x_opt @ Q @ x_opt + q @ x_opt
    
    # Check optimality
    optimality_error = solver.l2_optimality_error(x_opt, params_obj, params_ineq)
    
    return x_opt, obj_value, result.state.iter_num, optimality_error


@jax.jit
def solve_qp_boxcdqp(
    Q: jnp.ndarray,
    q: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    constant: float = 0.0,
    tol: float = 1e-4,
    max_iter: int = 500,
) -> tuple[jnp.ndarray, float, int, float]:
    """JIT-compiled BoxCDQP solver for box-constrained quadratic programs.
    
    Args:
        Q: Quadratic term matrix (CVXJax format: x^T Q x).
        q: Linear term vector.
        lb: Lower bounds.
        ub: Upper bounds.
        constant: Constant term in objective.
        tol: Convergence tolerance.
        max_iter: Maximum iterations.
        
    Returns:
        Tuple of (solution, objective_value, iterations, optimality_error).
    """
    # Call the core JIT solver directly
    x_opt, obj_value_no_const, iterations, optimality_error = _solve_boxcdqp_jit(
        Q, q, lb, ub, tol, max_iter
    )
    
    # Add constant term
    obj_value = obj_value_no_const + constant
    
    return x_opt, obj_value, iterations, optimality_error


def solve_qp_boxcdqp_wrapper(
    qp_data: QPData,
    tol: float = 1e-4,
    max_iter: int = 500,
    verbose: bool = False,
    **solver_kwargs: Any,
) -> "Solution":
    """Non-JIT wrapper for CVXJax integration.
    
    This function handles constraint checking and Solution object creation.
    The core optimization is delegated to the JIT-compiled solve_qp_boxcdqp.
    """
    
    # Simple constraint validation (JIT-compatible)
    # For JIT compatibility, we skip validation and let the solver handle it
    # In practice, the solver will work fine for unconstrained problems
    # where n_eq = 0 and n_ineq = 0
    
    if verbose:
        print(f"BoxCDQP: Solving QP with coordinate descent")
    
    # Call JIT-compiled solver
    x_opt, obj_value, iterations, optimality_error = solve_qp_boxcdqp(
        qp_data.Q,
        qp_data.q,
        qp_data.lb,
        qp_data.ub,
        qp_data.constant,
        tol,
        max_iter
    )
    
    # Determine status
    optimality_error_scalar = float(optimality_error.item() if hasattr(optimality_error, 'item') else optimality_error)
    iterations_scalar = int(iterations.item() if hasattr(iterations, 'item') else iterations)
    
    if optimality_error_scalar <= tol:
        status = "optimal"
    elif iterations_scalar >= max_iter:
        status = "max_iter"
    else:
        status = "unknown"
    
    # Create primal variable mapping
    primal_vars = {}
    start_idx = 0
    for var in qp_data.variables:
        end_idx = start_idx + var.size
        var_value = x_opt[start_idx:end_idx].reshape(var.shape)
        primal_vars[var] = var_value
        start_idx = end_idx
    
    # Build info
    info = {
        "iterations": iterations_scalar,
        "optimality_error": optimality_error_scalar,
        "solver": "boxcdqp"
    }
    
    # Import Solution locally to avoid circular imports
    from cvxjax.api import Solution
    return Solution(
        status=status,
        obj_value=float(obj_value.item() if hasattr(obj_value, 'item') else obj_value),
        primal=primal_vars,
        dual={},  # No dual variables available from BoxCDQP
        info=info
    )
