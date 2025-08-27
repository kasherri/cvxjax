"""Canonicalization of optimization problems to standard forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import jax.numpy as jnp
from jax import tree_util

if TYPE_CHECKING:
    from cvxjax.api import Variable

from cvxjax.constraints import BoxConstraint, Constraint, EqualityConstraint, InequalityConstraint
from cvxjax.expressions import AffineExpression, Expression, QuadraticExpression


@dataclass(frozen=True)
class QPData:
    """Standard quadratic program format.
    
    Represents the problem:
        minimize    (1/2) x^T Q x + q^T x
        subject to  A_eq x = b_eq
                   A_ineq x <= b_ineq  
                   lb <= x <= ub
    
    Args:
        Q: Quadratic cost matrix (n x n).
        q: Linear cost vector (n,).
        A_eq: Equality constraint matrix (m_eq x n).
        b_eq: Equality constraint vector (m_eq,).
        A_ineq: Inequality constraint matrix (m_ineq x n).
        b_ineq: Inequality constraint vector (m_ineq,).
        lb: Lower bounds (n,), -inf for unbounded.
        ub: Upper bounds (n,), +inf for unbounded.
        variables: List of variables in order.
        n_vars: Number of variables.
        n_eq: Number of equality constraints.
        n_ineq: Number of inequality constraints.
    """
    Q: jnp.ndarray
    q: jnp.ndarray
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


# Register QPData as JAX pytree
tree_util.register_pytree_node(
    QPData,
    lambda qp: (
        (qp.Q, qp.q, qp.A_eq, qp.b_eq, qp.A_ineq, qp.b_ineq, qp.lb, qp.ub),
        {"variables": qp.variables, "n_vars": qp.n_vars, "n_eq": qp.n_eq, "n_ineq": qp.n_ineq}
    ),
    lambda aux, children: QPData(*children, **aux),
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
    """Build QP formulation from objective and constraints.
    
    Args:
        objective_expr: Objective expression to minimize.
        constraints: List of constraints.
        
    Returns:
        QPData representing the standard form QP.
        
    Raises:
        ValueError: If problem is not a valid QP.
    """
    # Extract all variables
    variables = _extract_variables(objective_expr, constraints)
    n_vars = sum(var.size for var in variables)
    
    # Build variable index mapping
    var_indices = {}
    start_idx = 0
    for var in variables:
        var_indices[var] = (start_idx, start_idx + var.size)
        start_idx += var.size
    
    # Handle objective expression (assume minimization - API layer handles max/min)
    obj_expr = objective_expr
    
    # Build Q and q from objective
    Q = jnp.zeros((n_vars, n_vars))
    q = jnp.zeros(n_vars)
    
    # Extract quadratic and linear terms from objective
    quad_terms, lin_terms, constant = _extract_objective_terms(obj_expr)
    
    # Build Q matrix from quadratic terms
    for (var1, var2), coeff in quad_terms.items():
        if var1 == var2:  # Diagonal quadratic term
            start1, end1 = var_indices[var1]
            Q = Q.at[start1:end1, start1:end1].set(coeff)
    
    # Build q vector from linear terms
    for var, coeff in lin_terms.items():
        start, end = var_indices[var]
        if coeff.ndim == 0:
            # Scalar coefficient
            q = q.at[start:end].set(coeff)
        else:
            # Vector coefficient - extract diagonal or flatten
            if coeff.ndim == 1:
                q = q.at[start:end].set(coeff)
            else:
                q = q.at[start:end].set(jnp.diag(coeff))
    
    # Note: constant term is ignored for QP formulation
    
    # Note: maximize/minimize handling is done at the API layer
    
    # Build constraint matrices
    eq_rows = []
    eq_rhs = []
    ineq_rows = []
    ineq_rhs = []
    
    # Initialize bounds
    lb = jnp.full(n_vars, -jnp.inf)
    ub = jnp.full(n_vars, jnp.inf)
    
    for constraint in constraints:
        if isinstance(constraint, EqualityConstraint):
            row, rhs = _build_constraint_row(constraint.expression, var_indices, n_vars)
            eq_rows.append(row)
            eq_rhs.append(-rhs)  # Move constant to RHS
            
        elif isinstance(constraint, InequalityConstraint):
            # Handle vector constraints by expanding them into scalar constraints
            expr = constraint.expression
            if (isinstance(expr, AffineExpression) and 
                len(expr.coeffs) == 1 and 
                jnp.allclose(expr.offset, 0)):
                # Simple variable constraint like x >= 0
                var, coeff = next(iter(expr.coeffs.items()))
                if var in var_indices and jnp.allclose(coeff, jnp.eye(var.size)):
                    # Vector constraint x >= 0 - expand to element-wise constraints
                    start, end = var_indices[var]
                    for i in range(var.size):
                        # Create row for x[i] >= 0 
                        row = jnp.zeros(n_vars)
                        row = row.at[start + i].set(1.0)
                        
                        if constraint.sense == "<=":
                            ineq_rows.append(row)
                            ineq_rhs.append(0.0)
                        else:  # ">="
                            ineq_rows.append(-row)
                            ineq_rhs.append(0.0)
                else:
                    # General affine constraint - handle as before
                    row, rhs = _build_constraint_row(constraint.expression, var_indices, n_vars)
                    if constraint.sense == "<=":
                        ineq_rows.append(row)
                        ineq_rhs.append(-rhs)
                    else:  # ">="
                        ineq_rows.append(-row)
                        ineq_rhs.append(rhs)
            else:
                # General case
                row, rhs = _build_constraint_row(constraint.expression, var_indices, n_vars)
                if constraint.sense == "<=":
                    ineq_rows.append(row)
                    ineq_rhs.append(-rhs)
                else:  # ">="
                    ineq_rows.append(-row)
                    ineq_rhs.append(rhs)
                
        elif isinstance(constraint, BoxConstraint):
            # Handle box constraints by updating bounds
            if not isinstance(constraint.expression, AffineExpression):
                raise ValueError("Box constraints require affine expressions")
            
            # For simple variable bounds
            if len(constraint.expression.coeffs) == 1:
                var, coeff = next(iter(constraint.expression.coeffs.items()))
                start, end = var_indices[var]
                
                # Check if coefficient is identity (simple variable bound)
                if jnp.allclose(coeff, jnp.eye(var.size)):
                    if constraint.lower is not None:
                        lb = lb.at[start:end].set(constraint.lower)
                    if constraint.upper is not None:
                        ub = ub.at[start:end].set(constraint.upper)
                else:
                    # General affine box constraint - convert to inequalities
                    if constraint.lower is not None:
                        row, rhs = _build_constraint_row(constraint.expression, var_indices, n_vars)
                        ineq_rows.append(-row)
                        ineq_rhs.append(constraint.lower + rhs)
                    if constraint.upper is not None:
                        row, rhs = _build_constraint_row(constraint.expression, var_indices, n_vars)
                        ineq_rows.append(row)
                        ineq_rhs.append(constraint.upper - rhs)
        else:
            raise ValueError(f"Unsupported constraint type: {type(constraint)}")
    
    # Convert to arrays
    if eq_rows:
        A_eq = jnp.array(eq_rows)
        b_eq = jnp.array(eq_rhs)
        n_eq = len(eq_rows)
    else:
        A_eq = jnp.zeros((0, n_vars))
        b_eq = jnp.zeros(0)
        n_eq = 0
    
    if ineq_rows:
        A_ineq = jnp.array(ineq_rows)
        b_ineq = jnp.array(ineq_rhs)
        n_ineq = len(ineq_rows)
    else:
        A_ineq = jnp.zeros((0, n_vars))
        b_ineq = jnp.zeros(0)
        n_ineq = 0
    
    return QPData(
        Q=Q, q=q, A_eq=A_eq, b_eq=b_eq, A_ineq=A_ineq, b_ineq=b_ineq,
        lb=lb, ub=ub, variables=variables, n_vars=n_vars, n_eq=n_eq, n_ineq=n_ineq
    )


def build_lp(objective_expr: Expression, constraints: List[Constraint]) -> LPData:
    """Build LP formulation from objective and constraints.
    
    Args:
        objective_expr: Linear objective expression.
        constraints: List of constraints.
        
    Returns:
        LPData representing the standard form LP.
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
                # Handle scalar vs vector variables
                if coeff.ndim == 0:
                    # Scalar coefficient for scalar variable
                    row = row.at[start:end].set(coeff)
                else:
                    # Extract diagonal for vector variable
                    row = row.at[start:end].set(jnp.diag(coeff))
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
                    if result.ndim == 0:
                        # Scalar result
                        row = row.at[start:end].set(result)
                    else:
                        # Vector result
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
        # JAX array constant
        constant = float(jnp.sum(expr))
        
    else:
        # Try to extract as scalar constant
        try:
            constant = float(expr)
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
    from cvxjax.expressions import QuadraticExpression, AffineExpression, AddExpression, ScalarMultiplyExpression, MatMulExpression
    
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
