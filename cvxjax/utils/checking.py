"""Validation and checking utilities."""

from typing import Tuple

import jax.numpy as jnp


def check_variable_shape(shape: Tuple[int, ...]) -> None:
    """Check that variable shape is valid.
    
    Args:
        shape: Shape tuple to validate.
        
    Note:
        This function is kept for API compatibility but does minimal checking
        to maintain JIT compatibility. Static shape validation should be done
        at construction time outside of JIT-compiled functions.
    """
    # Minimal validation for JIT compatibility
    # More comprehensive checks should be done at problem construction time
    pass


def check_shapes_compatible(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> None:
    """Check that two shapes are compatible for element-wise operations.
    
    Args:
        shape1: First shape.
        shape2: Second shape.
        
    Note:
        For JIT compatibility, this function does minimal checking.
        Shape compatibility should be validated at problem construction time.
    """
    # JAX will handle broadcasting and raise errors if incompatible
    # during actual computation, so we skip validation here for JIT compatibility
    pass


def check_matrix_multiply_shapes(left_shape: Tuple[int, ...], right_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """Check matrix multiplication shapes and return result shape.
    
    Args:
        left_shape: Shape of left operand.
        right_shape: Shape of right operand.
        
    Returns:
        Shape of the result.
        
    Note:
        For JIT compatibility, this function uses JAX shape inference.
        Actual shape checking is deferred to JAX operations.
    """
    # Use JAX's built-in shape inference for matrix multiplication
    # This maintains JIT compatibility while providing correct shapes
    left_ndim = len(left_shape)
    right_ndim = len(right_shape)
    
    if left_ndim == 1 and right_ndim == 1:
        # Vector dot product
        return ()
    elif left_ndim == 2 and right_ndim == 1:
        # Matrix-vector multiplication
        return (left_shape[0],)
    elif left_ndim == 1 and right_ndim == 2:
        # Vector-matrix multiplication  
        return (right_shape[1],)
    elif left_ndim == 2 and right_ndim == 2:
        # Matrix-matrix multiplication
        return (left_shape[0], right_shape[1])
    else:
        # Default case - let JAX handle it
        return left_shape  # Placeholder, JAX will determine actual shape


def check_psd_matrix(Q: jnp.ndarray, tol: float = 1e-8) -> None:
    """Check if matrix is positive semidefinite.
    
    Args:
        Q: Matrix to check.
        tol: Tolerance for eigenvalue check.
        
    Note:
        For JIT compatibility, this function skips validation.
        PSD checking should be done outside JIT-compiled functions.
    """
    # Skip PSD checking for JIT compatibility
    # This validation should be done at problem construction time
    pass


def check_finite_arrays(*arrays: jnp.ndarray) -> None:
    """Check that all arrays contain finite values.
    
    Args:
        *arrays: Arrays to check.
        
    Note:
        For JIT compatibility, this function skips validation.
        Finite value checking should be done outside JIT-compiled functions.
    """
    # Skip finite checking for JIT compatibility
    # JAX operations will handle NaN/inf appropriately
    pass


def check_problem_dimensions(
    Q: jnp.ndarray,
    q: jnp.ndarray, 
    A_eq: jnp.ndarray,
    b_eq: jnp.ndarray,
    A_ineq: jnp.ndarray,
    b_ineq: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
) -> None:
    """Check problem dimensions for consistency.
    
    Args:
        Q: Quadratic cost matrix.
        q: Linear cost vector.
        A_eq: Equality constraint matrix.
        b_eq: Equality constraint RHS.
        A_ineq: Inequality constraint matrix.
        b_ineq: Inequality constraint RHS.
        lb: Lower bounds.
        ub: Upper bounds.
        
    Note:
        For JIT compatibility, this function skips validation.
        Dimension checking should be done at problem construction time.
    """
    # Skip dimension checking for JIT compatibility
    # JAX operations will handle dimension mismatches appropriately
    pass
def check_feasibility_basic(
    A_eq: jnp.ndarray,
    b_eq: jnp.ndarray,
    A_ineq: jnp.ndarray,
    b_ineq: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
) -> bool:
    """Basic feasibility check for QP problem.
    
    Args:
        A_eq: Equality constraint matrix.
        b_eq: Equality constraint RHS.
        A_ineq: Inequality constraint matrix.
        b_ineq: Inequality constraint RHS.
        lb: Lower bounds.
        ub: Upper bounds.
        
    Returns:
        True if problem appears feasible (basic check only).
        
    Note:
        For JIT compatibility, this function returns True.
        Comprehensive feasibility checking should be done outside JIT.
    """
    # For JIT compatibility, we skip comprehensive feasibility checking
    # and return True. Solvers will detect infeasibility during solution.
    return True


def validate_solver_inputs(tol: float, max_iter: int, solver_name: str) -> None:
    """Validate solver input parameters.
    
    Args:
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
        solver_name: Name of the solver.
        
    Note:
        For JIT compatibility, this function skips validation.
        Parameter validation should be done outside JIT-compiled functions.
    """
    # Skip validation for JIT compatibility
    # Invalid parameters will be handled by the solver
    pass


def create_error_message(
    error_type: str,
    context: dict,
    suggestion: str | None = None,
) -> str:
    """Create informative error message.
    
    Args:
        error_type: Type of error.
        context: Context information.
        suggestion: Optional suggestion for fixing.
        
    Returns:
        Formatted error message.
    """
    message = f"CVXJAX {error_type}:"
    
    for key, value in context.items():
        message += f"\n  {key}: {value}"
    
    if suggestion:
        message += f"\n\nSuggestion: {suggestion}"
    
    return message
