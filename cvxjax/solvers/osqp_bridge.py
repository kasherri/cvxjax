"""OSQP solver bridge for quadratic programming via jaxopt."""

from typing import TYPE_CHECKING, Any, Dict

import jax.numpy as jnp
import jaxopt

if TYPE_CHECKING:
    from cvxjax.api import Solution

from cvxjax.canonicalize import QPData
from cvxjax.utils.checking import check_finite_arrays, check_problem_dimensions, validate_solver_inputs


def solve_qp_osqp(
    qp_data: QPData,
    tol: float = 1e-8,
    max_iter: int = 50,
    rho: float = 0.1,
    sigma: float = 1e-6,
    alpha: float = 1.6,
    **osqp_kwargs: Any,
) -> "Solution":
    """Solve QP using OSQP via jaxopt.
    
    This function converts the QP to OSQP format and solves using jaxopt.OSQP.
    
    Args:
        qp_data: QP problem data.
        tol: Convergence tolerance.
        max_iter: Maximum number of iterations.
        rho: OSQP penalty parameter.
        sigma: OSQP regularization parameter.
        alpha: OSQP relaxation parameter.
        **osqp_kwargs: Additional OSQP parameters.
        
    Returns:
        Solution object with optimal values and solver information.
    """
    # Validate inputs
    validate_solver_inputs(tol, max_iter, "osqp")
    check_problem_dimensions(
        qp_data.Q, qp_data.q, qp_data.A_eq, qp_data.b_eq,
        qp_data.A_ineq, qp_data.b_ineq, qp_data.lb, qp_data.ub
    )
    check_finite_arrays(
        qp_data.Q, qp_data.q, qp_data.A_eq, qp_data.b_eq,
        qp_data.A_ineq, qp_data.b_ineq, qp_data.lb, qp_data.ub
    )
    
    # Convert to OSQP format
    osqp_data = _convert_to_osqp_format(qp_data)
    
    # Set up OSQP solver
    solver = jaxopt.OSQP(
        tol=tol,
        maxiter=max_iter,
        rho=rho,
        sigma=sigma,
        alpha=alpha,
        **osqp_kwargs
    )
    
    # Solve
    try:
        result = solver.run(
            params_obj=(osqp_data["P"], osqp_data["q"]),
            params_eq=osqp_data["eq_constraints"],
            params_ineq=osqp_data["ineq_constraints"],
        )
        
        x_opt = result.params
        converged = result.state.converged
        num_iter = result.state.iter_num
        
        # Map OSQP status to our status
        if converged:
            status = "optimal"
        elif num_iter >= max_iter:
            status = "max_iter"
        else:
            status = "error"
            
    except Exception as e:
        # Handle solver failure
        x_opt = jnp.zeros(qp_data.n_vars)
        status = "error"
        num_iter = 0
        
        # Build minimal solution for error case
        primal = {}
        start_idx = 0
        for var in qp_data.variables:
            end_idx = start_idx + var.size
            primal[var] = jnp.zeros(var.shape)
            start_idx = end_idx
        
        from cvxjax.api import Solution
        return Solution(
            status=status,
            obj_value=float('inf'),
            primal=primal,
            dual={},
            info={"error": str(e), "solver": "osqp"},
        )
    
    # Compute objective value
    obj_value = 0.5 * x_opt @ qp_data.Q @ x_opt + qp_data.q @ x_opt
    
    # Build primal solution mapping
    primal = {}
    start_idx = 0
    for var in qp_data.variables:
        end_idx = start_idx + var.size
        var_value = x_opt[start_idx:end_idx].reshape(var.shape)
        primal[var] = var_value
        start_idx = end_idx
    
    # Extract dual variables from OSQP result
    dual = {}
    if hasattr(result.state, 'dual_eq') and qp_data.n_eq > 0:
        dual["eq_constraints"] = result.state.dual_eq
    if hasattr(result.state, 'dual_ineq') and qp_data.n_ineq > 0:
        dual["ineq_constraints"] = result.state.dual_ineq
    
    # Compute residuals for info
    residuals = _compute_osqp_residuals(qp_data, x_opt)
    
    info = {
        "iterations": int(num_iter),
        "primal_residual": residuals["primal"],
        "dual_residual": residuals["dual"],
        "solver": "osqp",
        "rho": rho,
        "sigma": sigma,
        "alpha": alpha,
    }
    
    from cvxjax.api import Solution
    return Solution(
        status=status,
        obj_value=obj_value,
        primal=primal,
        dual=dual,
        info=info,
    )


def _convert_to_osqp_format(qp_data: QPData) -> Dict[str, Any]:
    """Convert QP data to OSQP format.
    
    OSQP solves:
        minimize    (1/2) x^T P x + q^T x
        subject to  l <= A x <= u
    
    We need to convert our format:
        minimize    (1/2) x^T Q x + q^T x
        subject to  A_eq x = b_eq
                   A_ineq x <= b_ineq
                   lb <= x <= ub
    """
    n_vars = qp_data.n_vars
    
    # Objective (P matrix and q vector)
    P = qp_data.Q
    q = qp_data.q
    
    # Build constraint matrix A and bounds l, u
    constraint_rows = []
    lower_bounds = []
    upper_bounds = []
    
    # Equality constraints: A_eq x = b_eq becomes b_eq <= A_eq x <= b_eq
    if qp_data.n_eq > 0:
        constraint_rows.append(qp_data.A_eq)
        lower_bounds.append(qp_data.b_eq)
        upper_bounds.append(qp_data.b_eq)
    
    # Inequality constraints: A_ineq x <= b_ineq becomes -inf <= A_ineq x <= b_ineq
    if qp_data.n_ineq > 0:
        constraint_rows.append(qp_data.A_ineq)
        lower_bounds.append(jnp.full(qp_data.n_ineq, -jnp.inf))
        upper_bounds.append(qp_data.b_ineq)
    
    # Box constraints: lb <= x <= ub becomes lb <= I x <= ub
    has_finite_bounds = jnp.logical_or(jnp.isfinite(qp_data.lb), jnp.isfinite(qp_data.ub))
    if jnp.any(has_finite_bounds):
        constraint_rows.append(jnp.eye(n_vars))
        lower_bounds.append(qp_data.lb)
        upper_bounds.append(qp_data.ub)
    
    # Combine all constraints
    if constraint_rows:
        A = jnp.vstack(constraint_rows)
        l = jnp.concatenate(lower_bounds)
        u = jnp.concatenate(upper_bounds)
    else:
        # Unconstrained problem
        A = jnp.zeros((0, n_vars))
        l = jnp.array([])
        u = jnp.array([])
    
    # Format for jaxopt.OSQP
    eq_constraints = None
    ineq_constraints = None
    
    if qp_data.n_eq > 0:
        eq_constraints = (qp_data.A_eq, qp_data.b_eq)
    
    # Combine inequality and bound constraints
    if qp_data.n_ineq > 0 or jnp.any(has_finite_bounds):
        ineq_A_list = []
        ineq_b_list = []
        
        # Inequality constraints
        if qp_data.n_ineq > 0:
            ineq_A_list.append(qp_data.A_ineq)
            ineq_b_list.append(qp_data.b_ineq)
        
        # Box constraints as inequalities
        if jnp.any(jnp.isfinite(qp_data.ub)):
            # Upper bounds: x <= ub becomes x - ub <= 0
            finite_ub = jnp.isfinite(qp_data.ub)
            if jnp.any(finite_ub):
                ub_A = jnp.eye(n_vars)[finite_ub]
                ub_b = qp_data.ub[finite_ub]
                ineq_A_list.append(ub_A)
                ineq_b_list.append(ub_b)
        
        if jnp.any(jnp.isfinite(qp_data.lb)):
            # Lower bounds: lb <= x becomes -x <= -lb  
            finite_lb = jnp.isfinite(qp_data.lb)
            if jnp.any(finite_lb):
                lb_A = -jnp.eye(n_vars)[finite_lb]
                lb_b = -qp_data.lb[finite_lb]
                ineq_A_list.append(lb_A)
                ineq_b_list.append(lb_b)
        
        if ineq_A_list:
            ineq_A = jnp.vstack(ineq_A_list)
            ineq_b = jnp.concatenate(ineq_b_list)
            ineq_constraints = (ineq_A, ineq_b)
    
    return {
        "P": P,
        "q": q,
        "A": A,
        "l": l,
        "u": u,
        "eq_constraints": eq_constraints,
        "ineq_constraints": ineq_constraints,
    }


def _compute_osqp_residuals(qp_data: QPData, x: jnp.ndarray) -> Dict[str, float]:
    """Compute residuals for OSQP solution."""
    residuals = {}
    
    # Primal residuals
    if qp_data.n_eq > 0:
        eq_residual = qp_data.A_eq @ x - qp_data.b_eq
        residuals["primal_eq"] = float(jnp.linalg.norm(eq_residual))
    else:
        residuals["primal_eq"] = 0.0
    
    if qp_data.n_ineq > 0:
        ineq_residual = jnp.maximum(qp_data.A_ineq @ x - qp_data.b_ineq, 0)
        residuals["primal_ineq"] = float(jnp.linalg.norm(ineq_residual))
    else:
        residuals["primal_ineq"] = 0.0
    
    # Bound violations
    lb_violation = jnp.maximum(qp_data.lb - x, 0)
    ub_violation = jnp.maximum(x - qp_data.ub, 0)
    residuals["bounds"] = float(jnp.linalg.norm(lb_violation) + jnp.linalg.norm(ub_violation))
    
    # Overall primal residual
    residuals["primal"] = max(residuals["primal_eq"], residuals["primal_ineq"], residuals["bounds"])
    
    # Dual residual (stationarity) - simplified
    dual_residual = qp_data.Q @ x + qp_data.q
    residuals["dual"] = float(jnp.linalg.norm(dual_residual))
    
    return residuals


def check_osqp_available() -> bool:
    """Check if OSQP is available via jaxopt.
    
    Returns:
        True if OSQP is available, False otherwise.
    """
    try:
        solver = jaxopt.OSQP()
        return True
    except (ImportError, AttributeError):
        return False
