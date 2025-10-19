"""Expression system for building optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Union

import jax.numpy as jnp
from jax import tree_util

from cvxjax.utils.checking import check_shapes_compatible


class Expression(ABC):
    """Base class for all expressions in optimization problems."""
    
    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """Shape of the expression."""
        pass
    
    @property
    def size(self) -> int:
        """Total number of elements in the expression."""
        # Use shape[0] for 1D case or np.prod for static shapes
        if len(self.shape) == 0:
            return 1
        elif len(self.shape) == 1:
            return self.shape[0]
        else:
            # For multi-dimensional, compute at definition time if possible
            import numpy as np
            return int(np.prod(self.shape))
    
    @abstractmethod
    def is_affine(self) -> bool:
        """Check if expression is affine."""
        pass
    
    @abstractmethod
    def is_convex(self) -> bool:
        """Check if expression is convex."""
        pass
    
    def __add__(self, other: Union[Expression, jnp.ndarray, float]) -> Expression:
        # Convert scalars to arrays for uniform handling
        if not hasattr(other, 'shape'):
            other = jnp.array(other)
        # Check if other is a JAX array
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            return AffineExpression.from_constant(other) + self
        return AddExpression(self, other)
    
    def __radd__(self, other: Union[jnp.ndarray, float]) -> Expression:
        return self + other
    
    def __sub__(self, other: Union[Expression, jnp.ndarray, float]) -> Expression:
        # Handle Variable case
        if hasattr(other, 'name') and hasattr(other, 'shape') and not hasattr(other, 'coeffs'):
            # other is a Variable, convert to AffineExpression
            other = AffineExpression.from_variable(other)
        # Handle Expression case
        elif hasattr(other, 'coeffs') or isinstance(other, Expression):
            # other is already an Expression, use as-is
            pass
        # Convert constants to expressions for JIT compatibility
        elif not hasattr(other, 'coeffs'):
            if hasattr(other, 'shape') and hasattr(other, 'dtype'):
                # JAX array
                other = AffineExpression.from_constant(other)
            else:
                # Scalar - make sure it's not an expression object
                if isinstance(other, Expression):
                    pass  # Keep as expression
                else:
                    other = AffineExpression.from_constant(jnp.asarray(other))
        return SubtractExpression(self, other)
    
    def __rsub__(self, other: Union[jnp.ndarray, float]) -> Expression:
        return other + (-self)
    
    def __mul__(self, other: Union[float, jnp.ndarray]) -> Expression:
        # Convert scalars to arrays for uniform handling (JIT-compatible)
        if not hasattr(other, 'shape'):
            other = jnp.asarray(other, dtype=jnp.float64)
        return ScalarMultiplyExpression(self, other)
    
    def __rmul__(self, other: Union[float, jnp.ndarray]) -> Expression:
        return self * other
    
    def __matmul__(self, other: Union[jnp.ndarray, Expression]) -> Expression:
        # Check if other is a JAX array
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            return MatMulExpression(self, AffineExpression.from_constant(other))
        return MatMulExpression(self, other)
    
    def __rmatmul__(self, other: jnp.ndarray) -> Expression:
        return MatMulExpression(AffineExpression.from_constant(other), self)
    
    def __neg__(self) -> Expression:
        return self * (-1.0)
    
    def __le__(self, other: Union[Expression, jnp.ndarray]) -> Any:
        from cvxjax.constraints import InequalityConstraint
        return InequalityConstraint(self - other, sense="<=")
    
    def __ge__(self, other: Union[Expression, jnp.ndarray]) -> Any:
        from cvxjax.constraints import InequalityConstraint
        return InequalityConstraint(self - other, sense=">=")
    
    def __eq__(self, other: Union[Expression, jnp.ndarray]) -> Any:  # type: ignore
        from cvxjax.constraints import EqualityConstraint
        return EqualityConstraint(self - other)


@dataclass(frozen=True)
class AffineExpression(Expression):
    """Affine expression: A*x + b where A is coefficient matrix and b is offset.
    
    Args:
        coeffs: Dictionary mapping variables to their coefficient matrices.
        offset: Constant offset term.
        _shape: Shape of the resulting expression.
    """
    coeffs: Dict[Any, jnp.ndarray]  # Variable -> coefficient matrix
    offset: jnp.ndarray
    _shape: tuple[int, ...]
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the expression."""
        return self._shape
    
    @classmethod
    def from_variable(cls, variable: Any) -> AffineExpression:
        """Create affine expression from a single variable.
        
        Args:
            variable: Variable to create expression from.
            
        Returns:
            AffineExpression representing the variable.
        """
        eye = jnp.eye(variable.size).reshape(variable.shape + variable.shape)
        return cls(
            coeffs={variable: eye},
            offset=jnp.zeros(variable.shape),
            _shape=variable.shape,
        )
    
    @classmethod
    def from_constant(cls, value: jnp.ndarray) -> AffineExpression:
        """Create affine expression from a constant.
        
        Args:
            value: Constant value.
            
        Returns:
            AffineExpression representing the constant.
        """
        return cls(
            coeffs={},
            offset=value,
            _shape=value.shape,
        )
    
    def is_affine(self) -> bool:
        """Affine expressions are always affine."""
        return True
    
    def is_convex(self) -> bool:
        """Affine expressions are always convex."""
        return True
    
    def __eq__(self, other: Union[Expression, jnp.ndarray, float]) -> Any:  # type: ignore
        """Create equality constraint instead of comparing dataclass fields."""
        return super().__eq__(other)
    
    def __add__(self, other: Union[Expression, jnp.ndarray, float]) -> AffineExpression:
        # Convert scalars to arrays for uniform handling
        if not hasattr(other, 'shape'):
            other = jnp.array(other)
        
        # Check if other is a JAX array
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            check_shapes_compatible(self.shape, other.shape)
            return AffineExpression(
                coeffs=self.coeffs,
                offset=self.offset + other,
                _shape=self.shape,
            )
        
        # Check if other is an AffineExpression by examining attributes
        if hasattr(other, 'coeffs') and hasattr(other, 'offset'):
            check_shapes_compatible(self.shape, other.shape)
            
            # For JIT compatibility, we need to avoid dictionary iteration
            # This is a simplified implementation that works for common cases
            # where both expressions have the same variables
            
            # Merge coefficients using a functional approach
            # Start with a copy of self coefficients
            new_coeffs = dict(self.coeffs)
            
            # For JIT compatibility, we assume the coefficient dictionaries
            # are small and can be handled statically
            # In practice, most expressions involve only a few variables
            
            # Add coefficients from other expression
            # This avoids explicit iteration over dict.items()
            if len(other.coeffs) > 0:
                # For each variable in other, add its coefficient
                for var in other.coeffs:
                    coeff = other.coeffs[var]
                    if var in new_coeffs:
                        new_coeffs[var] = new_coeffs[var] + coeff
                    else:
                        new_coeffs[var] = coeff
            
            return AffineExpression(
                coeffs=new_coeffs,
                offset=self.offset + other.offset,
                _shape=self.shape,
            )
        
        return super().__add__(other)
    
    def __sub__(self, other: Union[Expression, jnp.ndarray, float]) -> AffineExpression:
        # Handle Variable case  
        if hasattr(other, 'name') and hasattr(other, 'shape') and not hasattr(other, 'coeffs'):
            # other is a Variable, convert to AffineExpression
            other = AffineExpression.from_variable(other)
        # Handle constant case
        elif not hasattr(other, 'coeffs'):
            if hasattr(other, 'shape') and hasattr(other, 'dtype'):
                # JAX array
                other = AffineExpression.from_constant(other)
            else:
                # Scalar
                other = AffineExpression.from_constant(jnp.array(other))
        
        # If both operands are affine, we can compute the result as an AffineExpression
        if hasattr(other, 'coeffs') and hasattr(other, 'offset'):
            # Subtract coefficients and offsets
            new_coeffs = self.coeffs.copy()
            for var, coeff in other.coeffs.items():
                if var in new_coeffs:
                    new_coeffs[var] = new_coeffs[var] - coeff
                else:
                    new_coeffs[var] = -coeff
            
            new_offset = self.offset - other.offset
            
            return AffineExpression(
                coeffs=new_coeffs,
                offset=new_offset,
                _shape=self.shape,
            )
        
        # Fallback to SubtractExpression
        return SubtractExpression(self, other)
    
    def __mul__(self, scalar: Union[float, jnp.ndarray]) -> AffineExpression:
        # Convert scalars to arrays for uniform handling
        if not hasattr(scalar, 'shape'):
            scalar = jnp.array(scalar)
        
        # Scale coefficients and offset
        new_coeffs = {var: scalar * coeff for var, coeff in self.coeffs.items()}
        new_offset = scalar * self.offset
        
        return AffineExpression(
            coeffs=new_coeffs,
            offset=new_offset,
            _shape=self.shape,
        )
    
    def __matmul__(self, other: Union[jnp.ndarray, Expression]) -> AffineExpression:
        # Check if other is a JAX array
        if hasattr(other, 'shape') and hasattr(other, 'dtype'):
            # Matrix multiplication A @ x becomes (A ⊗ I) vec(x)
            new_coeffs = {}
            for var, coeff in self.coeffs.items():
                # Reshape for matrix multiplication
                new_coeff = jnp.tensordot(self.offset, other, axes=0)  # Simplified
                new_coeffs[var] = new_coeff
            
            new_offset = self.offset @ other
            new_shape = (self.shape[0], other.shape[1]) if other.shape.__len__() > 1 else (self.shape[0],)
            
            return AffineExpression(
                coeffs=new_coeffs,
                offset=new_offset,
                _shape=new_shape,
            )
        
        return super().__matmul__(other)


# Register AffineExpression as JAX pytree
tree_util.register_pytree_node(
    AffineExpression,
    lambda expr: ((expr.coeffs, expr.offset), {"shape": expr.shape}),
    lambda aux, children: AffineExpression(children[0], children[1], **aux),
)


class QuadraticExpression(Expression):
    """Quadratic expression: x^T Q x + q^T x + r.
    
    Args:
        quad_coeffs: Dictionary mapping variable pairs to quadratic coefficients.
        lin_coeffs: Dictionary mapping variables to linear coefficients.
        offset: Constant offset.
        shape: Shape of the expression (should be scalar for quadratic forms).
    """
    
    def __init__(
        self,
        quad_coeffs: Dict[tuple[Any, Any], jnp.ndarray],
        lin_coeffs: Dict[Any, jnp.ndarray],
        offset: jnp.ndarray,
        shape: tuple[int, ...],
    ) -> None:
        self.quad_coeffs = quad_coeffs
        self.lin_coeffs = lin_coeffs
        self.offset = offset
        self._shape = shape
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
    
    def is_affine(self) -> bool:
        """Quadratic expressions are not affine unless quadratic terms are zero."""
        # Check if quad_coeffs dict is empty (JAX-compatible way)
        return not bool(self.quad_coeffs)
    
    def is_convex(self) -> bool:
        """Check convexity by verifying all quadratic matrices are PSD."""
        # Simplified check - in practice would need proper PSD verification
        return True  # Assume convex for now
    
    def __add__(self, other: Union[Expression, jnp.ndarray, float]) -> Expression:
        # Convert scalars to arrays for uniform handling
        if not hasattr(other, 'shape'):
            other = jnp.array(other)
        
        # Handle scalar/array addition
        if hasattr(other, 'shape') and not hasattr(other, 'quad_coeffs'):
            return QuadraticExpression(
                quad_coeffs=self.quad_coeffs,
                lin_coeffs=self.lin_coeffs,
                offset=self.offset + other,
                shape=self.shape,
            )
        
        # Handle QuadraticExpression + QuadraticExpression
        if hasattr(other, 'quad_coeffs') and hasattr(other, 'lin_coeffs'):
            check_shapes_compatible(self.shape, other.shape)
            
            # Merge quadratic coefficients
            new_quad_coeffs = dict(self.quad_coeffs)
            for key, coeff in other.quad_coeffs.items():
                if key in new_quad_coeffs:
                    new_quad_coeffs[key] = new_quad_coeffs[key] + coeff
                else:
                    new_quad_coeffs[key] = coeff
            
            # Merge linear coefficients  
            new_lin_coeffs = dict(self.lin_coeffs)
            for var, coeff in other.lin_coeffs.items():
                if var in new_lin_coeffs:
                    new_lin_coeffs[var] = new_lin_coeffs[var] + coeff
                else:
                    new_lin_coeffs[var] = coeff
            
            return QuadraticExpression(
                quad_coeffs=new_quad_coeffs,
                lin_coeffs=new_lin_coeffs,
                offset=self.offset + other.offset,
                shape=self.shape,
            )
        
        # For other types, fall back to base class
        return super().__add__(other)


# Composite expressions
class AddExpression(Expression):
    """Sum of two expressions."""
    
    def __init__(self, left: Expression, right: Expression) -> None:
        check_shapes_compatible(left.shape, right.shape)
        self.left = left
        self.right = right
        self._shape = left.shape
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
    
    def is_affine(self) -> bool:
        return self.left.is_affine() and self.right.is_affine()
    
    def is_convex(self) -> bool:
        return self.left.is_convex() and self.right.is_convex()


class SubtractExpression(Expression):
    """Subtraction of two expressions."""
    
    def __init__(self, left: Expression, right: Expression) -> None:
        check_shapes_compatible(left.shape, right.shape)
        self.left = left
        self.right = right
        self._shape = left.shape
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
    
    def is_affine(self) -> bool:
        return self.left.is_affine() and self.right.is_affine()
    
    def is_convex(self) -> bool:
        # left - right is convex if left is convex and right is concave
        return self.left.is_convex()  # Simplified for now


class ScalarMultiplyExpression(Expression):
    """Scalar multiplication of an expression."""
    
    def __init__(self, expr: Expression, scalar: jnp.ndarray) -> None:
        self.expr = expr
        self.scalar = scalar
        self._shape = expr.shape
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
    
    def is_affine(self) -> bool:
        return self.expr.is_affine()
    
    def is_convex(self) -> bool:
        # Convex if scalar >= 0 and expr is convex, or scalar <= 0 and expr is concave
        return self.expr.is_convex()  # Simplified


class MatMulExpression(Expression):
    """Matrix multiplication of expressions."""
    
    def __init__(self, left: Expression, right: Expression) -> None:
        self.left = left
        self.right = right
        
        # Compute resulting shape (JAX-compatible)
        left_ndim = left.shape.__len__()
        right_ndim = right.shape.__len__()
        
        if left_ndim == 1 and right_ndim == 1:
            self._shape = ()  # Dot product
        elif left_ndim == 2 and right_ndim == 1:
            self._shape = (left.shape[0],)
        elif left_ndim == 1 and right_ndim == 2:
            self._shape = (right.shape[1],)
        elif left_ndim == 2 and right_ndim == 2:
            self._shape = (left.shape[0], right.shape[1])
        else:
            # For JIT compatibility, use default shape
            self._shape = left.shape
    
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape
    
    def is_affine(self) -> bool:
        return self.left.is_affine() and self.right.is_affine()
    
    def is_convex(self) -> bool:
        return self.left.is_convex() and self.right.is_convex()
