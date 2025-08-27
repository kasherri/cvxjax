"""Constraint classes for optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from cvxjax.expressions import Expression


class Constraint(ABC):
    """Base class for all constraints."""
    
    @abstractmethod
    def is_valid(self) -> bool:
        """Check if constraint is valid (e.g., DCP compliant)."""
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
        """Inequality constraints require convex expressions for <= and concave for >=."""
        if self.sense == "<=":
            return self.expression.is_convex()
        else:  # sense == ">="
            # For >= constraints, we need the expression to be concave
            # For now, we only support affine expressions which are both convex and concave
            return self.expression.is_affine()
    
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
    lower: float | None = None
    upper: float | None = None
    
    def is_valid(self) -> bool:
        """Box constraints require affine expressions."""
        return self.expression.is_affine()
    
    def __post_init__(self) -> None:
        # Skip validation for JIT compatibility
        # Constraint validity should be checked at problem construction time
        pass
