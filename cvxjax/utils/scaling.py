"""Scaling and preconditioning utilities for optimization problems."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple

import jax.numpy as jnp

if TYPE_CHECKING:
    from cvxjax.canonicalize import QPData


@dataclass(frozen=True)
class ScalingReport:
    """Report of scaling applied to optimization problem.
    
    Args:
        row_scale: Row scaling factors for constraints.
        col_scale: Column scaling factors for variables.
        obj_scale: Objective scaling factor.
        condition_before: Condition number before scaling.
        condition_after: Condition number after scaling.
        iterations: Number of scaling iterations performed.
    """
    row_scale: jnp.ndarray
    col_scale: jnp.ndarray
    obj_scale: float
    condition_before: float
    condition_after: float
    iterations: int


def diagonal_scale_qp(qp_data: "QPData", max_iter: int = 10, tol: float = 1e-2) -> Tuple["QPData", ScalingReport]:
    """Apply diagonal scaling to a quadratic program.
    
    This function uses the Ruiz equilibration algorithm to scale the constraint matrix
    and objective to improve numerical conditioning.
    
    Args:
        qp_data: QP data to scale.
        max_iter: Maximum number of scaling iterations.
        tol: Convergence tolerance for scaling factors.
        
    Returns:
        Tuple of scaled QP data and scaling report.
    """
    from cvxjax.canonicalize import QPData
    # Combine constraint matrices
    if qp_data.n_eq > 0 and qp_data.n_ineq > 0:
        A = jnp.vstack([qp_data.A_eq, qp_data.A_ineq])
    elif qp_data.n_eq > 0:
        A = qp_data.A_eq
    elif qp_data.n_ineq > 0:
        A = qp_data.A_ineq
    else:
        # No constraints to scale
        return qp_data, ScalingReport(
            row_scale=jnp.array([]),
            col_scale=jnp.ones(qp_data.n_vars),
            obj_scale=1.0,
            condition_before=1.0,
            condition_after=1.0,
            iterations=0,
        )
    
    # Compute initial condition number
    try:
        condition_before = float(jnp.linalg.cond(A))
    except:
        condition_before = float('inf')
    
    # Initialize scaling factors
    row_scale = jnp.ones(A.shape[0])
    col_scale = jnp.ones(A.shape[1])
    
    A_scaled = A
    
    # Iterative scaling
    for iteration in range(max_iter):
        # Save previous scaling
        row_scale_prev = row_scale.copy()
        col_scale_prev = col_scale.copy()
        
        # Column scaling (variable scaling)
        col_norms = jnp.linalg.norm(A_scaled, axis=0)
        col_norms = jnp.where(col_norms == 0, 1.0, col_norms)
        col_scale_iter = 1.0 / col_norms
        col_scale = col_scale * col_scale_iter
        A_scaled = A_scaled * col_scale_iter[None, :]
        
        # Row scaling (constraint scaling)
        row_norms = jnp.linalg.norm(A_scaled, axis=1)
        row_norms = jnp.where(row_norms == 0, 1.0, row_norms)
        row_scale_iter = 1.0 / row_norms
        row_scale = row_scale * row_scale_iter
        A_scaled = A_scaled * row_scale_iter[:, None]
        
        # Check convergence
        row_change = jnp.linalg.norm(row_scale - row_scale_prev) / jnp.linalg.norm(row_scale_prev)
        col_change = jnp.linalg.norm(col_scale - col_scale_prev) / jnp.linalg.norm(col_scale_prev)
        
        if max(row_change, col_change) < tol:
            break
    
    # Compute final condition number
    try:
        condition_after = float(jnp.linalg.cond(A_scaled))
    except:
        condition_after = float('inf')
    
    # Apply scaling to QP data
    scaled_qp = _apply_scaling_to_qp(qp_data, row_scale, col_scale)
    
    # Determine objective scaling
    obj_scale = 1.0
    if jnp.linalg.norm(scaled_qp.q) > 1e6:
        obj_scale = 1e6 / jnp.linalg.norm(scaled_qp.q)
        scaled_qp = QPData(
            Q=obj_scale * scaled_qp.Q,
            q=obj_scale * scaled_qp.q,
            A_eq=scaled_qp.A_eq,
            b_eq=scaled_qp.b_eq,
            A_ineq=scaled_qp.A_ineq,
            b_ineq=scaled_qp.b_ineq,
            lb=scaled_qp.lb,
            ub=scaled_qp.ub,
            variables=scaled_qp.variables,
            n_vars=scaled_qp.n_vars,
            n_eq=scaled_qp.n_eq,
            n_ineq=scaled_qp.n_ineq,
        )
    
    report = ScalingReport(
        row_scale=row_scale,
        col_scale=col_scale,
        obj_scale=obj_scale,
        condition_before=condition_before,
        condition_after=condition_after,
        iterations=iteration + 1,
    )
    
    return scaled_qp, report


def _apply_scaling_to_qp(qp_data: "QPData", row_scale: jnp.ndarray, col_scale: jnp.ndarray) -> "QPData":
    """Apply row and column scaling to QP data."""
    from cvxjax.canonicalize import QPData
    
    # Scale quadratic term: (Dx)^T Q (Dx) = x^T (D^T Q D) x
    D = jnp.diag(col_scale)
    Q_scaled = D.T @ qp_data.Q @ D
    
    # Scale linear term: q^T (Dx) = (D^T q)^T x  
    q_scaled = D.T @ qp_data.q
    
    # Scale equality constraints: A_eq (Dx) = b_eq becomes (A_eq D) x = b_eq
    if qp_data.n_eq > 0:
        A_eq_scaled = qp_data.A_eq @ D
        # Scale RHS: S_row * A_eq * D * x = S_row * b_eq
        row_scale_eq = row_scale[:qp_data.n_eq] if row_scale.shape[0] >= qp_data.n_eq else jnp.ones(qp_data.n_eq)
        A_eq_scaled = jnp.diag(row_scale_eq) @ A_eq_scaled
        b_eq_scaled = row_scale_eq * qp_data.b_eq
    else:
        A_eq_scaled = qp_data.A_eq
        b_eq_scaled = qp_data.b_eq
    
    # Scale inequality constraints
    if qp_data.n_ineq > 0:
        A_ineq_scaled = qp_data.A_ineq @ D
        row_scale_ineq = row_scale[qp_data.n_eq:] if row_scale.shape[0] > qp_data.n_eq else jnp.ones(qp_data.n_ineq)
        A_ineq_scaled = jnp.diag(row_scale_ineq) @ A_ineq_scaled
        b_ineq_scaled = row_scale_ineq * qp_data.b_ineq
    else:
        A_ineq_scaled = qp_data.A_ineq
        b_ineq_scaled = qp_data.b_ineq
    
    # Scale bounds: lb <= Dx <= ub becomes lb/d <= x <= ub/d
    lb_scaled = qp_data.lb / col_scale
    ub_scaled = qp_data.ub / col_scale
    
    return QPData(
        Q=Q_scaled,
        q=q_scaled,
        A_eq=A_eq_scaled,
        b_eq=b_eq_scaled,
        A_ineq=A_ineq_scaled,
        b_ineq=b_ineq_scaled,
        lb=lb_scaled,
        ub=ub_scaled,
        variables=qp_data.variables,
        n_vars=qp_data.n_vars,
        n_eq=qp_data.n_eq,
        n_ineq=qp_data.n_ineq,
    )


def unscale_solution(solution_scaled: jnp.ndarray, col_scale: jnp.ndarray) -> jnp.ndarray:
    """Unscale solution vector.
    
    Args:
        solution_scaled: Scaled solution x_scaled.
        col_scale: Column scaling factors.
        
    Returns:
        Original solution x = D * x_scaled.
    """
    return col_scale * solution_scaled


def compute_residual_norms(qp_data: "QPData", x: jnp.ndarray, y_eq: jnp.ndarray | None = None, 
                          y_ineq: jnp.ndarray | None = None) -> dict[str, float]:
    """Compute residual norms for QP solution.
    
    Args:
        qp_data: QP problem data.
        x: Primal solution.
        y_eq: Dual variables for equality constraints.
        y_ineq: Dual variables for inequality constraints.
        
    Returns:
        Dictionary with residual norms.
    """
    residuals = {}
    
    # Primal residuals
    if qp_data.n_eq > 0:
        primal_eq_res = qp_data.A_eq @ x - qp_data.b_eq
        residuals["primal_eq"] = float(jnp.linalg.norm(primal_eq_res))
    else:
        residuals["primal_eq"] = 0.0
    
    if qp_data.n_ineq > 0:
        primal_ineq_res = qp_data.A_ineq @ x - qp_data.b_ineq
        # Only consider violations (positive residuals)
        primal_ineq_viol = jnp.maximum(primal_ineq_res, 0)
        residuals["primal_ineq"] = float(jnp.linalg.norm(primal_ineq_viol))
    else:
        residuals["primal_ineq"] = 0.0
    
    # Bound violations
    lb_viol = jnp.maximum(qp_data.lb - x, 0)
    ub_viol = jnp.maximum(x - qp_data.ub, 0)
    residuals["bounds"] = float(jnp.linalg.norm(lb_viol) + jnp.linalg.norm(ub_viol))
    
    # Dual residual (stationarity)
    if y_eq is not None or y_ineq is not None:
        dual_res = qp_data.Q @ x + qp_data.q
        
        if qp_data.n_eq > 0 and y_eq is not None:
            dual_res += qp_data.A_eq.T @ y_eq
            
        if qp_data.n_ineq > 0 and y_ineq is not None:
            dual_res += qp_data.A_ineq.T @ y_ineq
        
        residuals["dual"] = float(jnp.linalg.norm(dual_res))
    else:
        residuals["dual"] = float('inf')
    
    # Overall residual
    residuals["overall"] = max(residuals["primal_eq"], residuals["primal_ineq"], 
                              residuals["bounds"], residuals.get("dual", 0))
    
    return residuals
