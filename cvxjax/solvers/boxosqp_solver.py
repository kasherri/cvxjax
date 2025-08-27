"""JAXOpt BoxOSQP solver for box-constrained quadratic programming."""

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
from jaxopt import BoxOSQP

if TYPE_CHECKING:
    from cvxjax.api import Solution

from cvxjax.canonicalize import QPData


def solve_qp_boxosqp(
    qp_data: QPData,
    tol: float = 1e-8,
    max_iter: int = 1000,
    verbose: bool = False,
    **solver_kwargs: Any,
) -> "Solution":
    """Solve QP using JAXOpt BoxOSQP solver.
    
    This solver is specifically designed for box-constrained quadratic programs:
        minimize    (1/2) x^T Q x + q^T x
        subject to  lb <= x <= ub
    
    Note: BoxOSQP only supports box constraints. General equality and inequality 
    constraints are not supported by this solver.
    
    Args:
        qp_data: QP problem data.
        tol: Convergence tolerance.
        max_iter: Maximum number of iterations.
        verbose: Print solver output.
        **solver_kwargs: Additional solver-specific arguments.
        
    Returns:
        Solution object with optimal values and solver information.
    """
    
    # Check that we only have box constraints
    if qp_data.n_eq > 0 or qp_data.n_ineq > 0:
        raise ValueError(
            "BoxOSQP solver only supports box constraints (lb <= x <= ub). "
            f"Problem has {qp_data.n_eq} equality and {qp_data.n_ineq} inequality constraints."
        )
    
    n_vars = qp_data.n_vars
    
    if verbose:
        print(f"BoxOSQP: Solving box-constrained QP with {n_vars} variables")
    
    # BoxOSQP interface uses matvec functions instead of matrices
    def matvec_Q(x):
        """Matrix-vector product with Q."""
        return qp_data.Q @ x
    
    # Create the solver  
    solver = BoxOSQP(
        matvec_Q=matvec_Q,
        tol=tol,
        maxiter=max_iter,
        verbose=verbose,
        **solver_kwargs
    )
    
    # Initial point (choose middle of bounds where possible)
    x0 = jnp.zeros(n_vars)
    for i in range(n_vars):
        lb_i = qp_data.lb[i]
        ub_i = qp_data.ub[i]
        
        if jnp.isfinite(lb_i) and jnp.isfinite(ub_i):
            # Both bounds finite: use midpoint
            x0 = x0.at[i].set((lb_i + ub_i) / 2.0)
        elif jnp.isfinite(lb_i):
            # Only lower bound: use lb + 1
            x0 = x0.at[i].set(lb_i + 1.0)
        elif jnp.isfinite(ub_i):
            # Only upper bound: use ub - 1  
            x0 = x0.at[i].set(ub_i - 1.0)
        # else: both infinite, keep x0[i] = 0
    
    try:
        # BoxOSQP run signature: run(init_params, params_obj, params_eq, params_ineq)
        # params_obj = (H, q) where H is None (we use matvec_Q)
        # params_ineq = (lb, ub) for box constraints
        params_obj = (None, qp_data.q)
        params_ineq = (qp_data.lb, qp_data.ub)
        
        result = solver.run(
            init_params=x0,
            params_obj=params_obj,
            params_ineq=params_ineq
        )
        
        # Extract solution
        x_opt = result.params
        
        # Compute objective value
        obj_value = 0.5 * x_opt @ qp_data.Q @ x_opt + qp_data.q @ x_opt + qp_data.constant
        
        # Map solver status to our status
        solver_status = result.state.status
        if solver_status == solver.SOLVED:
            status = "optimal"
        elif solver_status == solver.PRIMAL_INFEASIBLE:
            status = "primal_infeasible"
        elif solver_status == solver.DUAL_INFEASIBLE:
            status = "dual_infeasible"
        else:
            status = "max_iter"
        
        # Create primal variable mapping
        primal_vars = {}
        start_idx = 0
        for var in qp_data.variables:
            end_idx = start_idx + var.size
            var_value = x_opt[start_idx:end_idx].reshape(var.shape)
            primal_vars[var] = var_value
            start_idx = end_idx
        
        # No dual variables available from BoxOSQP
        dual_vars = {}
        
        # Build info
        info = {
            "iterations": result.state.iter_num,
            "status_code": solver_status,
            "solver": "boxosqp"
        }
        
        # Import Solution locally to avoid circular imports
        from cvxjax.api import Solution
        return Solution(
            status=status,
            obj_value=obj_value,
            primal=primal_vars,
            dual=dual_vars,
            info=info
        )
        
    except Exception as e:
        if verbose:
            print(f"BoxOSQP solver error: {e}")
        
        # Return error solution
        primal_vars = {}
        for var in qp_data.variables:
            primal_vars[var] = jnp.full(var.shape, jnp.nan)
        
        from cvxjax.api import Solution
        return Solution(
            status="error",
            obj_value=jnp.nan,
            primal=primal_vars,
            dual={},
            info={"solver": "boxosqp", "error": str(e)}
        )
