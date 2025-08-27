"""Shape validation and manipulation utilities."""

from typing import Tuple

import jax.numpy as jnp


def check_static_shape(shape: Tuple[int, ...]) -> None:
    """Check that shape contains only static (positive integer) dimensions.
    
    Args:
        shape: Shape tuple to validate.
        
    Raises:
        ValueError: If shape contains non-positive or non-integer dimensions.
    """
    if not isinstance(shape, (tuple, list)):
        raise ValueError(f"Shape must be tuple or list, got {type(shape)}")
    
    for i, dim in enumerate(shape):
        if not isinstance(dim, int):
            raise ValueError(f"Shape dimension {i} must be integer, got {type(dim)}")
        if dim <= 0:
            raise ValueError(f"Shape dimension {i} must be positive, got {dim}")


def broadcast_shapes(shape1: Tuple[int, ...], shape2: Tuple[int, ...]) -> Tuple[int, ...]:
    """Compute broadcast shape for two shapes.
    
    Args:
        shape1: First shape.
        shape2: Second shape.
        
    Returns:
        Broadcasted shape.
        
    Raises:
        ValueError: If shapes are not compatible for broadcasting.
    """
    # Reverse shapes for right-to-left processing
    shape1_rev = list(reversed(shape1))
    shape2_rev = list(reversed(shape2))
    
    result = []
    max_len = max(len(shape1_rev), len(shape2_rev))
    
    for i in range(max_len):
        dim1 = shape1_rev[i] if i < len(shape1_rev) else 1
        dim2 = shape2_rev[i] if i < len(shape2_rev) else 1
        
        if dim1 == dim2:
            result.append(dim1)
        elif dim1 == 1:
            result.append(dim2)
        elif dim2 == 1:
            result.append(dim1)
        else:
            raise ValueError(f"Cannot broadcast shapes {shape1} and {shape2}")
    
    return tuple(reversed(result))


def flatten_shape(shape: Tuple[int, ...]) -> int:
    """Compute total number of elements in shape.
    
    Args:
        shape: Shape tuple.
        
    Returns:
        Total number of elements.
    """
    return int(jnp.prod(jnp.array(shape)))


def is_vector_shape(shape: Tuple[int, ...]) -> bool:
    """Check if shape represents a vector (1D array).
    
    Args:
        shape: Shape to check.
        
    Returns:
        True if shape is 1D.
    """
    return len(shape) == 1


def is_matrix_shape(shape: Tuple[int, ...]) -> bool:
    """Check if shape represents a matrix (2D array).
    
    Args:
        shape: Shape to check.
        
    Returns:
        True if shape is 2D.
    """
    return len(shape) == 2


def is_scalar_shape(shape: Tuple[int, ...]) -> bool:
    """Check if shape represents a scalar (0D array).
    
    Args:
        shape: Shape to check.
        
    Returns:
        True if shape is 0D.
    """
    return len(shape) == 0


def pad_shape_left(shape: Tuple[int, ...], target_ndim: int) -> Tuple[int, ...]:
    """Pad shape with 1s on the left to reach target number of dimensions.
    
    Args:
        shape: Original shape.
        target_ndim: Target number of dimensions.
        
    Returns:
        Padded shape.
        
    Raises:
        ValueError: If shape already has more dimensions than target.
    """
    if len(shape) > target_ndim:
        raise ValueError(f"Shape {shape} has more than {target_ndim} dimensions")
    
    return (1,) * (target_ndim - len(shape)) + shape


def squeeze_shape(shape: Tuple[int, ...], axis: int | None = None) -> Tuple[int, ...]:
    """Remove dimensions of size 1 from shape.
    
    Args:
        shape: Original shape.
        axis: Specific axis to squeeze (None for all size-1 dimensions).
        
    Returns:
        Squeezed shape.
        
    Raises:
        ValueError: If specified axis doesn't have size 1.
    """
    if axis is not None:
        if axis < 0:
            axis = len(shape) + axis
        if axis < 0 or axis >= len(shape):
            raise ValueError(f"Axis {axis} out of bounds for shape {shape}")
        if shape[axis] != 1:
            raise ValueError(f"Cannot squeeze axis {axis} with size {shape[axis]}")
        return shape[:axis] + shape[axis + 1:]
    else:
        return tuple(dim for dim in shape if dim != 1)


def expand_dims_shape(shape: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
    """Add dimension of size 1 at specified axis.
    
    Args:
        shape: Original shape.
        axis: Position to insert new dimension.
        
    Returns:
        Expanded shape.
    """
    if axis < 0:
        axis = len(shape) + 1 + axis
    if axis < 0 or axis > len(shape):
        raise ValueError(f"Axis {axis} out of bounds for insertion")
    
    return shape[:axis] + (1,) + shape[axis:]


def transpose_shape(shape: Tuple[int, ...], axes: Tuple[int, ...] | None = None) -> Tuple[int, ...]:
    """Compute shape after transposition.
    
    Args:
        shape: Original shape.
        axes: Permutation of axes (None for reverse order).
        
    Returns:
        Transposed shape.
    """
    if axes is None:
        return tuple(reversed(shape))
    
    if len(axes) != len(shape):
        raise ValueError(f"Axes length {len(axes)} doesn't match shape dimensions {len(shape)}")
    
    # Normalize negative axes
    axes = tuple(ax if ax >= 0 else len(shape) + ax for ax in axes)
    
    # Check axes are valid
    if not all(0 <= ax < len(shape) for ax in axes):
        raise ValueError(f"Invalid axes {axes} for shape {shape}")
    
    if len(set(axes)) != len(axes):
        raise ValueError(f"Repeated axes in {axes}")
    
    return tuple(shape[ax] for ax in axes)


def reshape_compatible(old_shape: Tuple[int, ...], new_shape: Tuple[int, ...]) -> bool:
    """Check if reshape from old_shape to new_shape is valid.
    
    Args:
        old_shape: Original shape.
        new_shape: Target shape.
        
    Returns:
        True if reshape is valid.
    """
    old_size = flatten_shape(old_shape)
    new_size = flatten_shape(new_shape)
    return old_size == new_size
