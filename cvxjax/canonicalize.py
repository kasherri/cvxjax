"""Canonicalization of optimization problems to standard forms.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import jax.numpy as jnp
from jax import tree_util, lax, jit

if TYPE_CHECKING:
    from cvxjax.api import Variable

from cvxjax.constraints import BoxConstraint, Constraint, EqualityConstraint, InequalityConstraint
from cvxjax.expressions import AffineExpression, Expression, QuadraticExpression


@dataclass(frozen=True)
class CanonicalData:
    """Vectorized data for canonical form with static shapes.
    
    This structure holds all constraint and objective data in vectorized form
    to enable JIT compilation without Python control flow.
    """
    # Variable metadata (static)
    var_sizes: jnp.ndarray  # Size of each variable
    var_offsets: jnp.ndarray  # Offset indices for each variable
    n_vars_total: jnp.ndarray  # Total number of scalar variables (as array)
    
    # Objective data
    Q: jnp.ndarray  # Quadratic term matrix
    q: jnp.ndarray  # Linear term vector
    constant: float  # Constant term
    
    # Constraint data (vectorized)
    constraint_types: jnp.ndarray  # 0=eq, 1=ineq_leq, 2=ineq_geq, 3=box
    constraint_A: jnp.ndarray  # Constraint matrix rows (max_constraints x n_vars)
    constraint_b: jnp.ndarray  # Constraint RHS values
    constraint_mask: jnp.ndarray  # Boolean mask for active constraints
    
    # Box constraint bounds
    lb: jnp.ndarray  # Lower bounds
    ub: jnp.ndarray  # Upper bounds
    
    # Sizes (as JAX arrays to avoid concretization)
    n_constraints: jnp.ndarray  # Number of active constraints
    n_eq: jnp.ndarray  # Number of equality constraints  
    n_ineq: jnp.ndarray  # Number of inequality constraints


# Register as JAX pytree
tree_util.register_pytree_node(
    CanonicalData,
    lambda cd: (
        (cd.var_sizes, cd.var_offsets, cd.n_vars_total, cd.Q, cd.q, cd.constraint_types, 
         cd.constraint_A, cd.constraint_b, cd.constraint_mask, cd.lb, cd.ub,
         cd.n_constraints, cd.n_eq, cd.n_ineq),
        {"constant": cd.constant}
    ),
    lambda aux, children: CanonicalData(
        var_sizes=children[0], var_offsets=children[1], n_vars_total=children[2],
        Q=children[3], q=children[4], constant=aux["constant"],
        constraint_types=children[5], constraint_A=children[6], constraint_b=children[7],
        constraint_mask=children[8], lb=children[9], ub=children[10],
        n_constraints=children[11], n_eq=children[12], n_ineq=children[13]
    ),
)


@dataclass(frozen=True)
class QPData:
    """Standard quadratic program format.
    
    Represents the problem:
        minimize    (1/2) x^T Q x + q^T x + constant
        subject to  A_eq x = b_eq
                   A_ineq x <= b_ineq  
                   lb <= x <= ub
    
    Args:
        Q: Quadratic cost matrix (n x n).
        q: Linear cost vector (n,).
        constant: Constant term in objective.
        A_eq: Equality constraint matrix (m_eq x n).
        b_eq: Equality constraint vector (m_eq,).
        A_ineq: Inequality constraint matrix (m_ineq x n).
        b_ineq: Inequality constraint vector (m_ineq,).
        lb: Lower bounds (n,), -inf for unbounded.
        ub: Upper bounds (n,), +inf for unbounded.
        variables: List of variables in order.
        n_vars: Number of variables (JAX array for JIT compatibility).
        n_eq: Number of equality constraints (JAX array for JIT compatibility).
        n_ineq: Number of inequality constraints (JAX array for JIT compatibility).
    """
    Q: jnp.ndarray
    q: jnp.ndarray
    constant: float
    A_eq: jnp.ndarray
    b_eq: jnp.ndarray
    A_ineq: jnp.ndarray  
    b_ineq: jnp.ndarray
    lb: jnp.ndarray
    ub: jnp.ndarray
    variables: List[Variable]
    n_vars: jnp.ndarray  # Changed to JAX array for JIT compatibility
    n_eq: jnp.ndarray    # Changed to JAX array for JIT compatibility
    n_ineq: jnp.ndarray  # Changed to JAX array for JIT compatibility


# Register QPData as JAX pytree
tree_util.register_pytree_node(
    QPData,
    lambda qp: (
        (qp.Q, qp.q, qp.A_eq, qp.b_eq, qp.A_ineq, qp.b_ineq, qp.lb, qp.ub, qp.n_vars, qp.n_eq, qp.n_ineq),
        {"variables": qp.variables, "constant": qp.constant}
    ),
    lambda aux, children: QPData(
        Q=children[0], q=children[1], constant=aux["constant"],
        A_eq=children[2], b_eq=children[3], A_ineq=children[4], b_ineq=children[5],
        lb=children[6], ub=children[7], variables=aux["variables"],
        n_vars=children[8], n_eq=children[9], n_ineq=children[10]
    ),
)


@dataclass(frozen=True)
class LPData:
    """Standard linear program format.
    
    Represents the problem:
        minimize    c^T x
        subject to  A_eq x = b_eq
                   A_ineq x <= b_ineq
                   lb <= x <= ub
    """
    c: jnp.ndarray
    A_eq: jnp.ndarray
    b_eq: jnp.ndarray
    A_ineq: jnp.ndarray
    b_ineq: jnp.ndarray
    lb: jnp.ndarray
    ub: jnp.ndarray
    variables: List[Variable]
    n_vars: int
    n_eq: int
    n_ineq: int


# Register LPData as JAX pytree
tree_util.register_pytree_node(
    LPData,
    lambda lp: (
        (lp.c, lp.A_eq, lp.b_eq, lp.A_ineq, lp.b_ineq, lp.lb, lp.ub),
        {"variables": lp.variables, "n_vars": lp.n_vars, "n_eq": lp.n_eq, "n_ineq": lp.n_ineq}
    ),
    lambda aux, children: LPData(*children, **aux),
)








def build_qp(objective_expr: Expression, constraints: List[Constraint]) -> QPData:
    """JIT-compatible QP builder. Converts objective and constraints to vectorized form and returns QPData.
    Args:
        objective_expr: Objective expression to minimize.
        constraints: List of constraints.
    Returns:
        QPData representing the standard form QP (JIT-compatible).
    Raises:
        ValueError: If problem is not a valid QP.
    """
    # Extract variables and build mapping
    variables = _extract_variables(objective_expr, constraints)
    n_vars = sum(var.size for var in variables)
    
    # Get preprocessed data from prepare_problem_data
    objective_data, constraint_data, _ = prepare_problem_data(objective_expr, constraints)
    
    # Extract matrices directly from prepared data
    Q = objective_data['Q']
    q = objective_data['q'] 
    constant = objective_data['constant']
    
    constraint_types = constraint_data['types']
    constraint_A = constraint_data['A']
    constraint_b = constraint_data['b']
    constraint_mask = constraint_data['mask']
    
    # Initialize bounds from variable bounds arrays
    lb = constraint_data.get('var_lower', jnp.full(n_vars, -jnp.inf))
    ub = constraint_data.get('var_upper', jnp.full(n_vars, jnp.inf))
    
    # Apply additional box constraints if any (from explicit BoxConstraint objects)
    box_mask = constraint_types == 3
    box_lower = constraint_data['box_lower']
    box_upper = constraint_data['box_upper']
    
    # Use JAX where to conditionally update bounds from box constraints
    lb = jnp.where(
        box_mask[:n_vars] & (box_lower[:n_vars] > -jnp.inf),
        box_lower[:n_vars], 
        lb
    )
    ub = jnp.where(
        box_mask[:n_vars] & (box_upper[:n_vars] < jnp.inf),
        box_upper[:n_vars], 
        ub
    )
    
    # Build constraint matrices using extract_qp_matrices_static logic directly
    eq_mask = constraint_mask & (constraint_types == 0)
    ineq_leq_mask = constraint_mask & (constraint_types == 1) 
    ineq_geq_mask = constraint_mask & (constraint_types == 2)
    
    # Build equality constraint matrix with masking
    A_eq_full = jnp.where(eq_mask[:, None], constraint_A, 0.0)
    b_eq_full = jnp.where(eq_mask, constraint_b, 0.0)
    
    # Build inequality constraint matrices with masking
    # For <= constraints
    A_ineq_leq = jnp.where(ineq_leq_mask[:, None], constraint_A, 0.0)
    b_ineq_leq = jnp.where(ineq_leq_mask, constraint_b, 0.0)
    
    # For >= constraints (convert to <= by negating)
    A_ineq_geq = jnp.where(ineq_geq_mask[:, None], -constraint_A, 0.0)
    b_ineq_geq = jnp.where(ineq_geq_mask, -constraint_b, 0.0)
    
    # Combine inequality constraints
    A_ineq_full = A_ineq_leq + A_ineq_geq
    b_ineq_full = b_ineq_leq + b_ineq_geq
    ineq_mask_full = ineq_leq_mask | ineq_geq_mask
    
    # Zero out inactive inequality constraints
    A_ineq_full = jnp.where(ineq_mask_full[:, None], A_ineq_full, 0.0)
    b_ineq_full = jnp.where(ineq_mask_full, b_ineq_full, 0.0)
    
    # Count active constraints
    n_eq_actual = jnp.sum(eq_mask)
    n_ineq_actual = jnp.sum(ineq_mask_full)
    
    # Use the maximum possible number of constraints for static sizing
    max_constraints = len(constraints)
    
    # Extract only active constraints to avoid shape mismatches
    # For equality constraints
    eq_indices = jnp.where(eq_mask, size=max_constraints, fill_value=0)[0]
    A_eq_active = A_eq_full[eq_indices[:n_eq_actual]]
    b_eq_active = b_eq_full[eq_indices[:n_eq_actual]]
    
    # For inequality constraints  
    ineq_indices = jnp.where(ineq_mask_full, size=max_constraints, fill_value=0)[0]
    A_ineq_active = A_ineq_full[ineq_indices[:n_ineq_actual]]
    b_ineq_active = b_ineq_full[ineq_indices[:n_ineq_actual]]
    
    return QPData(
        Q=Q,
        q=q,
        constant=constant,
        A_eq=A_eq_active,
        b_eq=b_eq_active,
        A_ineq=A_ineq_active,
        b_ineq=b_ineq_active,
        lb=lb,
        ub=ub,
        variables=variables,
        n_vars=jnp.array(n_vars),
        n_eq=n_eq_actual,
        n_ineq=n_ineq_actual
    )




# JIT-compatible constraint processing functions
@jit
def process_constraints_vectorized(
    constraint_types: jnp.ndarray,
    constraint_A: jnp.ndarray, 
    constraint_b: jnp.ndarray,
    constraint_mask: jnp.ndarray,
    n_vars: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Process constraints in vectorized form for JIT compilation.
    
    Args:
        constraint_types: Array of constraint type codes (0=eq, 1=ineq_leq, 2=ineq_geq, 3=box)
        constraint_A: Constraint coefficient matrix
        constraint_b: Constraint RHS values
        constraint_mask: Boolean mask for active constraints
        n_vars: Number of variables
        
    Returns:
        Tuple of (A_eq, b_eq, A_ineq, b_ineq)
    """
    # Extract equality constraints
    eq_mask = constraint_mask & (constraint_types == 0)
    A_eq = jnp.where(
        eq_mask[:, None], 
        constraint_A, 
        jnp.zeros_like(constraint_A)
    )
    b_eq = jnp.where(eq_mask, constraint_b, 0.0)
    
    # Filter to only active equality constraints
    active_eq = jnp.sum(eq_mask)
    A_eq = A_eq[:active_eq]
    b_eq = b_eq[:active_eq]
    
    # Extract inequality constraints
    ineq_leq_mask = constraint_mask & (constraint_types == 1)
    ineq_geq_mask = constraint_mask & (constraint_types == 2)
    
    # Process <= constraints
    A_ineq_leq = jnp.where(
        ineq_leq_mask[:, None],
        constraint_A,
        jnp.zeros_like(constraint_A)
    )
    b_ineq_leq = jnp.where(ineq_leq_mask, constraint_b, 0.0)
    
    # Process >= constraints (convert to <= by negating)
    A_ineq_geq = jnp.where(
        ineq_geq_mask[:, None],
        -constraint_A,  # Negate to convert >= to <=
        jnp.zeros_like(constraint_A)
    )
    b_ineq_geq = jnp.where(ineq_geq_mask, -constraint_b, 0.0)  # Negate RHS too
    
    # Combine all inequality constraints
    A_ineq = jnp.concatenate([A_ineq_leq, A_ineq_geq], axis=0)
    b_ineq = jnp.concatenate([b_ineq_leq, b_ineq_geq], axis=0)
    
    # Filter to only active inequality constraints  
    active_ineq = jnp.sum(ineq_leq_mask) + jnp.sum(ineq_geq_mask)
    A_ineq = A_ineq[:active_ineq]
    b_ineq = b_ineq[:active_ineq]
    
    return A_eq, b_eq, A_ineq, b_ineq


@jit
def update_bounds_vectorized(
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    constraint_types: jnp.ndarray,
    constraint_mask: jnp.ndarray,
    box_lower: jnp.ndarray,
    box_upper: jnp.ndarray,
    var_indices: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Update variable bounds from box constraints in vectorized form.
    
    Args:
        lb: Current lower bounds
        ub: Current upper bounds  
        constraint_types: Constraint type codes
        constraint_mask: Active constraint mask
        box_lower: Lower bound values for box constraints
        box_upper: Upper bound values for box constraints
        var_indices: Variable index mapping
        
    Returns:
        Updated (lb, ub) bounds
    """
    # Identify box constraints
    box_mask = constraint_mask & (constraint_types == 3)
    
    # Update bounds using vectorized operations
    lb = jnp.where(
        box_mask & (box_lower > -jnp.inf),
        jnp.maximum(lb, box_lower),
        lb
    )
    
    ub = jnp.where(
        box_mask & (box_upper < jnp.inf),
        jnp.minimum(ub, box_upper),
        ub
    )
    
    return lb, ub


@jit
def build_objective_matrices_vectorized(
    Q_data: jnp.ndarray,
    q_data: jnp.ndarray,
    Q_indices: jnp.ndarray,
    q_indices: jnp.ndarray,
    n_vars: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build objective matrices Q and q in vectorized form.
    
    Args:
        Q_data: Quadratic coefficient values
        q_data: Linear coefficient values  
        Q_indices: Indices for Q matrix placement
        q_indices: Indices for q vector placement
        n_vars: Number of variables
        
    Returns:
        Objective matrices (Q, q)
    """
    # Initialize matrices
    Q = jnp.zeros((n_vars, n_vars))
    q = jnp.zeros(n_vars)
    
    # Use scatter operations to place coefficients
    Q = Q.at[Q_indices[:, 0], Q_indices[:, 1]].add(Q_data)
    q = q.at[q_indices].add(q_data)
    
    return Q, q


@jit
def canonicalize_problem(
    Q: jnp.ndarray,
    q: jnp.ndarray,
    constant: float,
    constraint_types: jnp.ndarray,
    constraint_A: jnp.ndarray,
    constraint_b: jnp.ndarray,
    constraint_mask: jnp.ndarray,
    box_lower: jnp.ndarray,
    box_upper: jnp.ndarray
) -> QPData:
    """Main JIT-compatible canonicalization function.
    
    Args:
        Q: Quadratic cost matrix
        q: Linear cost vector
        constant: Constant term in objective
        constraint_types: Array of constraint type codes
        constraint_A: Constraint coefficient matrix
        constraint_b: Constraint RHS values
        constraint_mask: Boolean mask for active constraints
        box_lower: Lower bound values for box constraints
        box_upper: Upper bound values for box constraints
        
    Returns:
        Canonicalized QPData
    """
    # This function is now redundant since build_qp handles everything directly
    # Keeping for backward compatibility but redirecting to build_qp logic
    n_vars = Q.shape[0]
    
    # Initialize bounds
    lb = jnp.full(n_vars, -jnp.inf)
    ub = jnp.full(n_vars, jnp.inf)
    
    # Apply box constraints
    box_mask = constraint_types == 3
    lb = jnp.where(
        box_mask[:n_vars] & (box_lower[:n_vars] > -jnp.inf),
        box_lower[:n_vars], 
        lb
    )
    ub = jnp.where(
        box_mask[:n_vars] & (box_upper[:n_vars] < jnp.inf),
        box_upper[:n_vars], 
        ub
    )
    
    # Build constraint matrices
    eq_mask = constraint_mask & (constraint_types == 0)
    ineq_leq_mask = constraint_mask & (constraint_types == 1) 
    ineq_geq_mask = constraint_mask & (constraint_types == 2)
    
    A_eq_full = jnp.where(eq_mask[:, None], constraint_A, 0.0)
    b_eq_full = jnp.where(eq_mask, constraint_b, 0.0)
    
    A_ineq_leq = jnp.where(ineq_leq_mask[:, None], constraint_A, 0.0)
    b_ineq_leq = jnp.where(ineq_leq_mask, constraint_b, 0.0)
    
    A_ineq_geq = jnp.where(ineq_geq_mask[:, None], -constraint_A, 0.0)
    b_ineq_geq = jnp.where(ineq_geq_mask, -constraint_b, 0.0)
    
    A_ineq_full = A_ineq_leq + A_ineq_geq
    b_ineq_full = b_ineq_leq + b_ineq_geq
    ineq_mask_full = ineq_leq_mask | ineq_geq_mask
    
    A_ineq_full = jnp.where(ineq_mask_full[:, None], A_ineq_full, 0.0)
    b_ineq_full = jnp.where(ineq_mask_full, b_ineq_full, 0.0)
    
    n_eq_actual = jnp.sum(eq_mask)
    n_ineq_actual = jnp.sum(ineq_mask_full)
    
    # Extract only active constraints to avoid shape mismatches
    # For equality constraints
    eq_indices = jnp.where(eq_mask, size=max_constraints, fill_value=0)[0]
    A_eq_active = A_eq_full[eq_indices[:n_eq_actual]]
    b_eq_active = b_eq_full[eq_indices[:n_eq_actual]]
    
    # For inequality constraints  
    ineq_indices = jnp.where(ineq_mask_full, size=max_constraints, fill_value=0)[0]
    A_ineq_active = A_ineq_full[ineq_indices[:n_ineq_actual]]
    b_ineq_active = b_ineq_full[ineq_indices[:n_ineq_actual]]
    
    return QPData(
        Q=Q,
        q=q,
        constant=constant,
        A_eq=A_eq_active,
        b_eq=b_eq_active,
        A_ineq=A_ineq_active,
        b_ineq=b_ineq_active,
        lb=lb,
        ub=ub,
        variables=[],  # No variables in JIT context
        n_vars=jnp.array(n_vars),
        n_eq=n_eq_actual,
        n_ineq=n_ineq_actual
    )


def prepare_problem_data(
    objective_expr: Expression, 
    constraints: List[Constraint],
    max_constraints: int = 100
) -> tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """Prepare problem data for JIT compilation by pre-processing into vectorized form.
    
    This function handles Python objects and control flow to prepare static data
    that can be used in JIT-compiled functions.
    
    Args:
        objective_expr: Objective expression
        constraints: List of constraints
        max_constraints: Maximum number of constraints (for static sizing)
        
    Returns:
        Tuple of (objective_data, constraint_data, var_metadata)
    """
    # Extract variables and build metadata
    variables = _extract_variables(objective_expr, constraints)
    var_sizes = jnp.array([var.size for var in variables])
    var_offsets = jnp.cumsum(jnp.concatenate([jnp.array([0]), var_sizes[:-1]]))
    n_vars_total = int(jnp.sum(var_sizes))
    
    var_metadata = {
        'sizes': var_sizes,
        'offsets': var_offsets,
        'n_total': n_vars_total
    }
    
    # Build variable index mapping for objective processing
    var_indices = {}
    start_idx = 0
    for i, var in enumerate(variables):
        var_indices[var] = (start_idx, start_idx + var.size)
        start_idx += var.size
    
    # Extract objective terms
    quad_terms, lin_terms, constant = _extract_objective_terms(objective_expr)
    
    # Build objective matrices
    Q = jnp.zeros((n_vars_total, n_vars_total))
    q = jnp.zeros(n_vars_total)
    
    # Process quadratic terms
    for (var1, var2), coeff in quad_terms.items():
        if var1 == var2:  # Diagonal quadratic term
            start1, end1 = var_indices[var1]
            Q = Q.at[start1:end1, start1:end1].set(coeff)
    
    # Process linear terms
    for var, coeff in lin_terms.items():
        start, end = var_indices[var]
        if coeff.ndim == 0:
            q = q.at[start:end].set(coeff)
        else:
            if coeff.ndim == 1:
                q = q.at[start:end].set(coeff)
            else:
                q = q.at[start:end].set(jnp.diag(coeff))
    
    objective_data = {
        'Q': Q,
        'q': q,
        'constant': constant
    }
    
    # Process constraints into vectorized form
    constraint_types = jnp.zeros(max_constraints, dtype=jnp.int32)
    constraint_A = jnp.zeros((max_constraints, n_vars_total))
    constraint_b = jnp.zeros(max_constraints)
    constraint_mask = jnp.zeros(max_constraints, dtype=bool)
    box_lower = jnp.full(max_constraints, -jnp.inf)
    box_upper = jnp.full(max_constraints, jnp.inf)
    
    # Variable bounds arrays (indexed by variable, not constraint)
    var_lower = jnp.full(n_vars_total, -jnp.inf)
    var_upper = jnp.full(n_vars_total, jnp.inf)
    
    constraint_idx = 0
    for constraint in constraints:
        if constraint_idx >= max_constraints:
            break
            
        if isinstance(constraint, EqualityConstraint):
            row, rhs = _build_constraint_row(constraint.expression, var_indices, n_vars_total)
            constraint_types = constraint_types.at[constraint_idx].set(0)  # equality
            constraint_A = constraint_A.at[constraint_idx].set(row)
            constraint_b = constraint_b.at[constraint_idx].set(-rhs)
            constraint_mask = constraint_mask.at[constraint_idx].set(True)
            constraint_idx += 1
            
        elif isinstance(constraint, InequalityConstraint):
            # Check if this is a simple variable bound that should be converted to box constraint
            if (isinstance(constraint.expression, AffineExpression) and 
                len(constraint.expression.coeffs) == 1):
                
                var, coeff = next(iter(constraint.expression.coeffs.items()))
                start, end = var_indices[var]
                
                # Check if coefficient selects exactly one component (unit vector)
                if coeff.ndim == 2 and coeff.shape[0] == 1:
                    coeff_flat = coeff.flatten()
                    if jnp.sum(jnp.abs(coeff_flat)) == 1.0:  # Exactly one non-zero entry with value 1
                        # Find which component this selects
                        component_idx = jnp.argmax(jnp.abs(coeff_flat))
                        var_idx_in_problem = start + component_idx
                        
                        # This is a simple variable bound - handle directly as bound
                        # Constraint is: coeff * x + offset sense 0
                        # For unit coeff: x + offset sense 0 → x sense -offset
                        bound_value = -jnp.sum(constraint.expression.offset)
                        
                        if constraint.sense == ">=":
                            # x + offset >= 0  →  x >= -offset
                            var_lower = var_lower.at[var_idx_in_problem].set(bound_value)
                        else:  # "<="
                            # x + offset <= 0  →  x <= -offset
                            var_upper = var_upper.at[var_idx_in_problem].set(bound_value)
                        
                        # Skip adding this as a constraint since it's now a bound
                        continue
            
            # Regular inequality constraint processing
            row, rhs = _build_constraint_row(constraint.expression, var_indices, n_vars_total)
            if constraint.sense == "<=":
                constraint_types = constraint_types.at[constraint_idx].set(1)  # ineq_leq
                constraint_A = constraint_A.at[constraint_idx].set(row)
                constraint_b = constraint_b.at[constraint_idx].set(-rhs)
            else:  # ">="
                constraint_types = constraint_types.at[constraint_idx].set(2)  # ineq_geq
                constraint_A = constraint_A.at[constraint_idx].set(-row)
                constraint_b = constraint_b.at[constraint_idx].set(rhs)
            constraint_mask = constraint_mask.at[constraint_idx].set(True)
            constraint_idx += 1
            
        elif isinstance(constraint, BoxConstraint):
            # Handle box constraints
            if not isinstance(constraint.expression, AffineExpression):
                continue
                
            if len(constraint.expression.coeffs) == 1:
                var, coeff = next(iter(constraint.expression.coeffs.items()))
                start, end = var_indices[var]
                
                if jnp.allclose(coeff, jnp.eye(var.size)):
                    # Simple variable bounds
                    for i in range(var.size):
                        if constraint_idx >= max_constraints:
                            break
                        constraint_types = constraint_types.at[constraint_idx].set(3)  # box
                        if constraint.lower is not None:
                            box_lower = box_lower.at[constraint_idx].set(constraint.lower)
                        if constraint.upper is not None:
                            box_upper = box_upper.at[constraint_idx].set(constraint.upper)
                        constraint_mask = constraint_mask.at[constraint_idx].set(True)
                        constraint_idx += 1
    
    constraint_data = {
        'types': constraint_types,
        'A': constraint_A,
        'b': constraint_b,
        'mask': constraint_mask,
        'box_lower': box_lower,
        'box_upper': box_upper,
        'var_lower': var_lower,
        'var_upper': var_upper
    }
    
    return objective_data, constraint_data, var_metadata


def build_lp(objective_expr: Expression, constraints: List[Constraint]) -> LPData:
    """JIT-compatible LP builder. Converts objective and constraints to vectorized form and returns LPData.
    
    Args:
        objective_expr: Linear objective expression.
        constraints: List of constraints.
        
    Returns:
        LPData representing the standard form LP (JIT-compatible).
        
    Raises:
        ValueError: If objective is not affine or contains quadratic terms.
    """
    if not objective_expr.is_affine():
        raise ValueError("LP requires affine objective")
    
    # Build QP first then extract linear part
    qp_data = build_qp(objective_expr, constraints)
    
    # Verify Q is zero (linear objective)
    if not jnp.allclose(qp_data.Q, 0):
        raise ValueError("LP objective contains quadratic terms")
    
    return LPData(
        c=qp_data.q,
        A_eq=qp_data.A_eq,
        b_eq=qp_data.b_eq,
        A_ineq=qp_data.A_ineq,
        b_ineq=qp_data.b_ineq,
        lb=qp_data.lb,
        ub=qp_data.ub,
        variables=qp_data.variables,
        n_vars=qp_data.n_vars,
        n_eq=qp_data.n_eq,
        n_ineq=qp_data.n_ineq,
    )


def _extract_variables(objective: Expression, constraints: List[Constraint]) -> List[Variable]:
    """Extract all variables from objective and constraints."""
    variables = set()
    
    # Extract from objective
    _extract_vars_from_expr(objective, variables)
    
    # Extract from constraints
    for constraint in constraints:
        if hasattr(constraint, 'expression'):
            _extract_vars_from_expr(constraint.expression, variables)
    
    return sorted(list(variables), key=lambda v: (v.name or "", v.shape))


def _extract_vars_from_expr(expr: Expression, variables: set) -> None:
    """Recursively extract variables from an expression."""
    if isinstance(expr, AffineExpression):
        variables.update(expr.coeffs.keys())
    elif isinstance(expr, QuadraticExpression):
        for var in expr.lin_coeffs.keys():
            variables.add(var)
        for (var1, var2) in expr.quad_coeffs.keys():
            variables.add(var1)
            variables.add(var2)
    elif hasattr(expr, 'left') and hasattr(expr, 'right'):
        # Binary expressions (AddExpression, MatMulExpression, etc.)
        _extract_vars_from_expr(expr.left, variables)
        _extract_vars_from_expr(expr.right, variables)
    elif hasattr(expr, 'shape') and hasattr(expr, 'name') and hasattr(expr, 'size'):
        # Single Variable
        from cvxjax.api import Variable  # Import to avoid circular dependency
        if isinstance(expr, Variable):
            variables.add(expr)
    # Add other expression types as needed


def _build_constraint_row(expr: Expression, var_indices: Dict[Variable, tuple[int, int]], n_vars: int) -> tuple[jnp.ndarray, float]:
    """Build constraint matrix row from expression.
    
    Returns:
        Tuple of (coefficient_row, constant_term).
    """
    # Convert expression to coefficient representation
    row = jnp.zeros(n_vars)
    constant = 0.0
    
    # Handle different expression types
    if hasattr(expr, 'coeffs') and hasattr(expr, 'offset'):
        # Already an AffineExpression
        for var, coeff in expr.coeffs.items():
            if var in var_indices:
                start, end = var_indices[var]
                # Handle scalar vs vector/matrix variables
                if coeff.ndim == 0:
                    # Scalar coefficient for scalar variable
                    row = row.at[start:end].set(coeff)
                elif coeff.ndim == 2:
                    # 2D coefficient - extract diagonal for vector variable
                    row = row.at[start:end].set(jnp.diag(coeff))
                else:
                    # Multi-dimensional coefficient (e.g., matrix variables)
                    # Flatten the coefficient and select appropriate elements
                    coeff_reshaped = coeff.reshape(-1, var.size)
                    if coeff_reshaped.shape[0] == var.size and jnp.allclose(coeff_reshaped, jnp.eye(var.size)):
                        # This is an identity mapping - use the first row or diagonal
                        row = row.at[start:end].set(jnp.diag(coeff_reshaped))
                    else:
                        # General case - flatten and use first var.size elements
                        coeff_flat = coeff.flatten()[:var.size]
                        row = row.at[start:end].set(coeff_flat)
        constant = jnp.sum(expr.offset)
        
    elif hasattr(expr, 'left') and hasattr(expr, 'right') and type(expr).__name__ == 'MatMulExpression':
        # MatMulExpression: handle specifically before general binary expressions
        # Handle as A @ x where A is constant AffineExpression and x has coefficients
        if (hasattr(expr.left, 'coeffs') and hasattr(expr.left, 'offset') and 
            not expr.left.coeffs and hasattr(expr.right, 'coeffs')):
            # Left is constant AffineExpression, right has variables
            A = expr.left.offset  # Extract the constant vector/matrix
            
            # Process right side coefficients
            for var, coeff in expr.right.coeffs.items():
                if var in var_indices:
                    start, end = var_indices[var]
                    # Apply matrix multiplication: A @ coeff
                    result = A @ coeff
                    if hasattr(result, "ndim") and result.ndim > 1:
                        # Matrix result: flatten or sum to match shape
                        result = result.flatten()
                        if result.size != (end - start):
                            result = result.sum(axis=0)
                    if result.ndim == 0:
                        row = row.at[start:end].set(result)
                    else:
                        row = row.at[start:end].set(result)
            
            # Handle constant terms from both sides
            constant_term = jnp.sum(A @ expr.right.offset)
            constant += constant_term
        # Add more cases as needed
        
    elif hasattr(expr, 'left') and hasattr(expr, 'right'):
        # Other binary expressions (AddExpression, etc.)
        left_row, left_const = _build_constraint_row(expr.left, var_indices, n_vars)
        right_row, right_const = _build_constraint_row(expr.right, var_indices, n_vars)
        row = left_row + right_row
        constant = left_const + right_const
        
    elif hasattr(expr, 'shape') and hasattr(expr, 'name') and hasattr(expr, 'size'):
        # Single variable (check attributes instead of isinstance)
        if expr in var_indices:
            start, end = var_indices[expr]
            row = row.at[start:end].set(1.0)
            
    elif hasattr(expr, 'shape') and hasattr(expr, 'dtype'):
        # JAX array constant (JIT-compatible)
        constant = jnp.sum(expr)
        
    else:
        # Try to extract as scalar constant (JIT-compatible)
        try:
            constant = jnp.asarray(expr, dtype=jnp.float64)
        except:
            # Unknown expression type
            pass
    
    return row, constant


@dataclass(frozen=True)
class PackedQPData:
    """Packed QP data for JIT compilation with static shapes."""
    data: jnp.ndarray  # Flattened problem data
    shapes: Dict[str, tuple[int, ...]]  # Shape information
    n_vars: int
    n_eq: int
    n_ineq: int


def pack_static(qp_data: QPData) -> PackedQPData:
    """Pack QP data into static arrays for JIT compilation.
    
    Args:
        qp_data: QP data to pack.
        
    Returns:
        PackedQPData with flattened arrays and shape information.
    """
    # Flatten all arrays and concatenate
    arrays = [
        qp_data.Q.flatten(),
        qp_data.q.flatten(),
        qp_data.A_eq.flatten(),
        qp_data.b_eq.flatten(),
        qp_data.A_ineq.flatten(),
        qp_data.b_ineq.flatten(),
        qp_data.lb.flatten(),
        qp_data.ub.flatten(),
    ]
    
    data = jnp.concatenate(arrays)
    
    shapes = {
        "Q": qp_data.Q.shape,
        "q": qp_data.q.shape,
        "A_eq": qp_data.A_eq.shape,
        "b_eq": qp_data.b_eq.shape,
        "A_ineq": qp_data.A_ineq.shape,
        "b_ineq": qp_data.b_ineq.shape,
        "lb": qp_data.lb.shape,
        "ub": qp_data.ub.shape,
    }
    
    return PackedQPData(
        data=data,
        shapes=shapes,
        n_vars=qp_data.n_vars,
        n_eq=qp_data.n_eq,
        n_ineq=qp_data.n_ineq,
    )


def unpack_static(packed_data: PackedQPData, variables: List[Variable]) -> QPData:
    """Unpack static data back to QPData.
    
    Args:
        packed_data: Packed QP data.
        variables: Original variable list.
        
    Returns:
        Unpacked QPData.
    """
    data = packed_data.data
    shapes = packed_data.shapes
    
    # Calculate start indices for each array
    start = 0
    arrays = {}
    
    for name in ["Q", "q", "A_eq", "b_eq", "A_ineq", "b_ineq", "lb", "ub"]:
        size = int(jnp.prod(jnp.array(shapes[name])))
        arrays[name] = data[start:start + size].reshape(shapes[name])
        start += size
    
    return QPData(
        Q=arrays["Q"],
        q=arrays["q"],
        constant=0.0,  # TODO: Pack/unpack constant term
        A_eq=arrays["A_eq"],
        b_eq=arrays["b_eq"],
        A_ineq=arrays["A_ineq"],
        b_ineq=arrays["b_ineq"],
        lb=arrays["lb"],
        ub=arrays["ub"],
        variables=variables,
        n_vars=packed_data.n_vars,
        n_eq=packed_data.n_eq,
        n_ineq=packed_data.n_ineq,
    )


def _extract_objective_terms(expr: Expression) -> tuple[Dict, Dict, float]:
    """Extract quadratic terms, linear terms, and constant from objective expression.
    
    Args:
        expr: Objective expression.
        
    Returns:
        Tuple of (quad_terms, lin_terms, constant).
    """
    from cvxjax.expressions import QuadraticExpression, AffineExpression, AddExpression, SubtractExpression, ScalarMultiplyExpression, MatMulExpression
    
    if isinstance(expr, QuadraticExpression):
        return expr.quad_coeffs, expr.lin_coeffs, expr.offset
        
    elif isinstance(expr, AffineExpression):
        return {}, expr.coeffs, expr.offset
        
    elif isinstance(expr, AddExpression):
        # Recursively extract from left and right sides
        left_quad, left_lin, left_const = _extract_objective_terms(expr.left)
        right_quad, right_lin, right_const = _extract_objective_terms(expr.right)
        
        # Merge quadratic terms
        quad_terms = dict(left_quad)
        for key, coeff in right_quad.items():
            if key in quad_terms:
                quad_terms[key] = quad_terms[key] + coeff
            else:
                quad_terms[key] = coeff
        
        # Merge linear terms
        lin_terms = dict(left_lin)
        for var, coeff in right_lin.items():
            if var in lin_terms:
                lin_terms[var] = lin_terms[var] + coeff
            else:
                lin_terms[var] = coeff
                
        return quad_terms, lin_terms, left_const + right_const
        
    elif isinstance(expr, SubtractExpression):
        # Handle subtraction: left - right
        left_quad, left_lin, left_const = _extract_objective_terms(expr.left)
        right_quad, right_lin, right_const = _extract_objective_terms(expr.right)
        
        # Merge quadratic terms (left - right)
        quad_terms = dict(left_quad)
        for key, coeff in right_quad.items():
            if key in quad_terms:
                quad_terms[key] = quad_terms[key] - coeff
            else:
                quad_terms[key] = -coeff
        
        # Merge linear terms (left - right)
        lin_terms = dict(left_lin)
        for var, coeff in right_lin.items():
            if var in lin_terms:
                lin_terms[var] = lin_terms[var] - coeff
            else:
                lin_terms[var] = -coeff
                
        return quad_terms, lin_terms, left_const - right_const
        
    elif isinstance(expr, ScalarMultiplyExpression):
        # Extract from the inner expression and scale
        inner_quad, inner_lin, inner_const = _extract_objective_terms(expr.expr)
        scalar = expr.scalar
        
        # Scale all terms
        quad_terms = {key: scalar * coeff for key, coeff in inner_quad.items()}
        lin_terms = {var: scalar * coeff for var, coeff in inner_lin.items()}
        constant = scalar * inner_const
        
        return quad_terms, lin_terms, constant
        
    elif isinstance(expr, MatMulExpression):
        # Handle matrix multiplication like mu @ x
        # Both sides are likely AffineExpressions
        
        left_quad, left_lin, left_const = _extract_objective_terms(expr.left)
        right_quad, right_lin, right_const = _extract_objective_terms(expr.right)
        
        # Handle the case mu @ x where mu is constant (left_lin is empty) and x is variable
        if not left_quad and not left_lin and right_lin:
            # Left side is constant, right side has variables
            mu = left_const
            
            # Apply mu to each variable's coefficient in right_lin
            lin_terms = {}
            for var, coeff in right_lin.items():
                lin_terms[var] = mu @ coeff
            
            return {}, lin_terms, 0.0
        
        # Handle other MatMul cases - for now return empty
        return {}, {}, 0.0
        
    else:
        # Unknown expression type - return empty
        return {}, {}, 0.0


