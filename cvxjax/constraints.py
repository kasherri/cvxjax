"""Constraint classes for optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Union

import jax
import jax.numpy as jnp

from cvxjax.expressions import Expression


class Constraint(ABC):
    """Base class for all constraints."""
    
    @abstractmethod
    def is_valid(self) -> bool:
        """Check if constraint is valid (e.g., DCP compliant)."""
        pass
    
    @abstractmethod
    def to_canonical_form(self) -> dict:
        """Convert constraint to canonical form for processing.
        
        Returns:
            Dictionary with constraint data in canonical form.
        """
        pass


@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    """Equality constraint: expr == 0.
    
    Args:
        expression: Expression that should equal zero.
        
    Example:
        >>> x = Variable(shape=(2,))
        >>> con = EqualityConstraint(jnp.ones(2) @ x - 1)  # sum(x) == 1
    """
    expression: Expression
    
    def is_valid(self) -> bool:
        """Equality constraints require affine expressions."""
        return self.expression.is_affine()
    
    def to_canonical_form(self) -> dict:
        """Convert to canonical form: expr == 0."""
        return {
            "type": "equality",
            "expression": self.expression,
            "sense": "==",
            "rhs": 0.0
        }
    
    def __post_init__(self) -> None:
        # Skip validation for JIT compatibility
        # Constraint validity should be checked at problem construction time
        pass


@dataclass(frozen=True)
class InequalityConstraint(Constraint):
    """Inequality constraint: expr <= 0 or expr >= 0.
    
    Args:
        expression: Expression for the inequality.
        sense: Direction of inequality ("<=" or ">=").
        
    Example:
        >>> x = Variable(shape=(2,))
        >>> con = InequalityConstraint(x, sense=">=")  # x >= 0
    """
    expression: Expression
    sense: Literal["<=", ">="]
    
    def is_valid(self) -> bool:
        """Inequality constraints require convex expressions for <= and concave for >=.
        
        For JIT compatibility, we use JAX operations instead of Python control flow.
        """
        # Get convexity properties
        is_convex = self.expression.is_convex() 
        is_affine = self.expression.is_affine()
        
        # Use JAX select instead of if/else for JIT compatibility
        # For <= constraints: need convex expression
        # For >= constraints: need concave (for now, only affine which is both)
        is_leq = jnp.array(self.sense == "<=")
        is_geq = jnp.array(self.sense == ">=")
        
        # Use jnp.where for conditional logic
        return jnp.where(
            is_leq,
            is_convex,  # <= requires convex
            is_affine   # >= requires affine (concave not fully implemented)
        )
    
    def to_canonical_form(self) -> dict:
        """Convert to canonical form."""
        return {
            "type": "inequality", 
            "expression": self.expression,
            "sense": self.sense,
            "rhs": 0.0
        }
    
    def __post_init__(self) -> None:
        # Skip validation for JIT compatibility
        # Constraint validity should be checked at problem construction time
        pass


@dataclass(frozen=True)  
class BoxConstraint(Constraint):
    """Box constraint: lb <= expr <= ub.
    
    Args:
        expression: Expression to constrain.
        lower: Lower bound (None for no lower bound).
        upper: Upper bound (None for no upper bound).
        
    Example:
        >>> x = Variable(shape=(2,))
        >>> con = BoxConstraint(x, lower=0.0, upper=1.0)  # 0 <= x <= 1
    """
    expression: Expression
    lower: Union[float, jnp.ndarray, None] = None
    upper: Union[float, jnp.ndarray, None] = None
    
    def is_valid(self) -> bool:
        """Box constraints require affine expressions."""
        return self.expression.is_affine()
    
    def to_canonical_form(self) -> dict:
        """Convert to canonical form."""
        # Convert None bounds to appropriate infinity values for JAX arrays
        lower_bound = jnp.array(-jnp.inf) if self.lower is None else jnp.array(self.lower)
        upper_bound = jnp.array(jnp.inf) if self.upper is None else jnp.array(self.upper)
        
        return {
            "type": "box",
            "expression": self.expression,
            "lower": lower_bound,
            "upper": upper_bound
        }
    
    def has_lower_bound(self) -> jnp.ndarray:
        """Check if constraint has a finite lower bound (JIT-compatible)."""
        if self.lower is None:
            return jnp.array(False)
        return jnp.isfinite(jnp.array(self.lower))
    
    def has_upper_bound(self) -> jnp.ndarray:
        """Check if constraint has a finite upper bound (JIT-compatible).""" 
        if self.upper is None:
            return jnp.array(False)
        return jnp.isfinite(jnp.array(self.upper))
    
    def __post_init__(self) -> None:
        # Skip validation for JIT compatibility
        # Constraint validity should be checked at problem construction time
        pass


# JIT-compatible constraint processing functions
def is_simple_variable_bound(constraint: InequalityConstraint) -> jnp.ndarray:
    """Check if inequality constraint is a simple variable bound (JIT-compatible).
    
    Returns True if the constraint is of the form: x_i >= c or x_i <= c
    where x_i is a single variable component.
    """
    from cvxjax.expressions import AffineExpression
    
    if not isinstance(constraint.expression, AffineExpression):
        return jnp.array(False)
    
    # Check if exactly one variable with unit coefficient
    num_vars = len(constraint.expression.coeffs)
    if num_vars != 1:
        return jnp.array(False)
    
    # Get the single variable and coefficient
    var, coeff = next(iter(constraint.expression.coeffs.items()))
    
    # Check if coefficient is a unit vector (selects single component)
    if coeff.ndim != 2 or coeff.shape[0] != 1:
        return jnp.array(False)
    
    coeff_flat = coeff.flatten()
    # Check if exactly one entry is 1.0 and rest are 0
    is_unit = jnp.isclose(jnp.sum(jnp.abs(coeff_flat)), 1.0) & jnp.isclose(jnp.max(jnp.abs(coeff_flat)), 1.0)
    
    return is_unit


def get_variable_bound_info(constraint: InequalityConstraint) -> dict:
    """Extract variable bound information from simple variable bound constraints (JIT-compatible).
    
    Returns:
        Dictionary with bound information or empty dict if not a simple bound.
    """
    from cvxjax.expressions import AffineExpression
    
    if not isinstance(constraint.expression, AffineExpression):
        return {}
    
    if len(constraint.expression.coeffs) != 1:
        return {}
    
    var, coeff = next(iter(constraint.expression.coeffs.items()))
    
    # Find which component this constraint affects
    coeff_flat = coeff.flatten()
    component_idx = jnp.argmax(jnp.abs(coeff_flat))
    
    # Bound value: x + offset sense 0 → x sense -offset
    bound_value = -jnp.sum(constraint.expression.offset)
    
    return {
        "variable": var,
        "component_index": component_idx,
        "bound_value": bound_value,
        "sense": constraint.sense
    }


@jax.jit
def process_constraint_jit(
    constraint_type: int,
    constraint_data: dict,
    constraint_idx: int,
    max_constraints: int,
    n_vars_total: int
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int]:
    """JIT-compiled constraint processing function.
    
    Args:
        constraint_type: Type of constraint (0=eq, 1=ineq_leq, 2=ineq_geq, 3=box)
        constraint_data: Dictionary with constraint matrices and vectors
        constraint_idx: Current constraint index
        max_constraints: Maximum number of constraints
        n_vars_total: Total number of variables
    
    Returns:
        Updated constraint arrays and new constraint index
    """
    # Initialize arrays if not provided
    constraint_types = constraint_data.get('types', jnp.zeros(max_constraints, dtype=jnp.int32))
    constraint_A = constraint_data.get('A', jnp.zeros((max_constraints, n_vars_total)))
    constraint_b = constraint_data.get('b', jnp.zeros(max_constraints))
    constraint_mask = constraint_data.get('mask', jnp.zeros(max_constraints, dtype=bool))
    
    # Update constraint based on type
    constraint_types = constraint_types.at[constraint_idx].set(constraint_type)
    
    if 'row' in constraint_data:
        constraint_A = constraint_A.at[constraint_idx].set(constraint_data['row'])
    if 'rhs' in constraint_data:
        constraint_b = constraint_b.at[constraint_idx].set(constraint_data['rhs'])
    
    constraint_mask = constraint_mask.at[constraint_idx].set(True)
    
    # Increment index only if within bounds
    new_idx = jnp.where(constraint_idx < max_constraints - 1, constraint_idx + 1, constraint_idx)
    
    return constraint_types, constraint_A, constraint_b, constraint_mask, new_idx


def classify_constraint_type(constraint: Constraint) -> int:
    """Classify constraint type for processing (JIT-compatible preparation).
    
    Returns:
        Integer type: 0=equality, 1=inequality_leq, 2=inequality_geq, 3=box
    """
    if isinstance(constraint, EqualityConstraint):
        return 0
    elif isinstance(constraint, InequalityConstraint):
        return 1 if constraint.sense == "<=" else 2
    elif isinstance(constraint, BoxConstraint):
        return 3
    else:
        return -1  # Unknown type


@jax.jit
def update_variable_bounds_jit(
    var_lower: jnp.ndarray,
    var_upper: jnp.ndarray,
    var_idx: int,
    bound_value: float,
    is_lower_bound: bool
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """JIT-compatible function to update variable bounds.
    
    Args:
        var_lower: Current lower bounds array
        var_upper: Current upper bounds array  
        var_idx: Index of variable to update
        bound_value: New bound value
        is_lower_bound: True for lower bound, False for upper bound
    
    Returns:
        Updated (var_lower, var_upper) arrays
    """
    # Use JAX where instead of Python if/else
    new_lower = jnp.where(
        is_lower_bound,
        var_lower.at[var_idx].set(bound_value),
        var_lower
    )
    
    new_upper = jnp.where(
        ~is_lower_bound,
        var_upper.at[var_idx].set(bound_value),
        var_upper  
    )
    
    return new_lower, new_upper


def validate_constraint_dcp(constraint: Constraint) -> bool:
    """Validate constraint follows DCP rules (non-JIT validation function).
    
    This function performs DCP validation which may involve Python control flow.
    It should be called during problem setup, not inside JIT-compiled functions.
    
    Args:
        constraint: Constraint to validate
        
    Returns:
        True if constraint is DCP-compliant
    """
    # This is intentionally non-JIT to allow comprehensive validation
    try:
        if isinstance(constraint, EqualityConstraint):
            return constraint.expression.is_affine()
        elif isinstance(constraint, InequalityConstraint):
            if constraint.sense == "<=":
                return constraint.expression.is_convex()
            else:  # ">="
                # For >= constraints, expression should be concave
                # For now, we only support affine expressions
                return constraint.expression.is_affine()
        elif isinstance(constraint, BoxConstraint):
            return constraint.expression.is_affine()
        else:
            return False
    except Exception:
        return False


def prepare_constraints_for_jit(constraints: list[Constraint]) -> dict:
    """Prepare constraint data for JIT compilation.
    
    This function does the non-JIT preparation work (validation, classification)
    and returns data structures that can be processed by JIT functions.
    
    Args:
        constraints: List of constraints to prepare
        
    Returns:
        Dictionary with constraint data prepared for JIT processing
    """
    # Validate all constraints first (non-JIT)
    for i, constraint in enumerate(constraints):
        if not validate_constraint_dcp(constraint):
            raise ValueError(f"Constraint {i} violates DCP rules: {constraint}")
    
    # Classify constraints and prepare data
    constraint_types = []
    constraint_forms = []
    
    for constraint in constraints:
        constraint_type = classify_constraint_type(constraint)
        constraint_form = constraint.to_canonical_form()
        
        constraint_types.append(constraint_type)
        constraint_forms.append(constraint_form)
    
    return {
        "types": jnp.array(constraint_types),
        "forms": constraint_forms,
        "count": len(constraints)
    }
