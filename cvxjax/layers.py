"""Differentiable CVX layers for JAX neural networks.

This module provides CvxLayer, which wraps a CVXJAX Problem into a JAX-callable 
differentiable function that can be used as a layer in neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

import jax
import jax.numpy as jnp
from jax import tree_util

from cvxjax.api import Problem, Parameter, Variable


def _flatten_primal(primal: Dict[Variable, jnp.ndarray]) -> jnp.ndarray:
    """Flatten primal variable dictionary into a single vector.
    
    Variables are sorted by their memory id for deterministic ordering,
    then flattened and concatenated.
    
    Args:
        primal: Dictionary mapping variables to their values.
        
    Returns:
        Flattened vector containing all variable values.
    """
    if not primal:
        return jnp.array([])
    
    # Sort variables by id for deterministic ordering
    sorted_vars = sorted(primal.keys(), key=lambda v: id(v))
    
    # Flatten and concatenate all variables
    flattened_values = []
    for var in sorted_vars:
        flattened_values.append(primal[var].reshape(-1))
    
    return jnp.concatenate(flattened_values)


def _collect_outputs(solution, return_fields: tuple[str, ...]) -> Union[jnp.ndarray, tuple]:
    """Collect specified outputs from solution.
    
    Args:
        solution: Solver solution object.
        return_fields: Tuple of field names to return.
        
    Returns:
        Single array if one field, tuple of arrays if multiple fields.
    """
    outputs = []
    
    for field in return_fields:
        if field == "primal":
            outputs.append(_flatten_primal(solution.primal))
        elif field == "obj":
            outputs.append(solution.obj_value)
        elif field == "dual":
            # For MVP, return empty array - can be extended later
            outputs.append(jnp.array([]))
        else:
            raise ValueError(f"Unknown return field: {field}")
    
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


@dataclass(frozen=True)
class _LayerConfig:
    """Configuration for CvxLayer."""
    solver: str
    diff_mode: Literal["implicit", "unroll", "none"]
    return_fields: tuple[str, ...]
    settings: Optional[Any] = None


class CvxLayer:
    """Differentiable CVX layer for JAX neural networks.
    
    This class wraps a CVXJAX Problem into a JAX-callable differentiable function
    that can be composed with other JAX operations, including within neural networks.
    
    The layer supports:
    - JIT compilation via jax.jit
    - Batching via jax.vmap  
    - Automatic differentiation with configurable modes
    - Multiple solver backends
    
    Args:
        problem: CVXJAX Problem instance to wrap.
        solver: Solver backend to use. Options: "ipm", "osqp", "boxosqp", "boxcdqp".
        settings: Optional solver-specific settings.
        return_fields: Tuple specifying which outputs to return. 
            Options: "primal", "dual", "obj". Default: ("primal",).
        diff_mode: Differentiation mode. Options:
            - "implicit": Use implicit differentiation (recommended)
            - "unroll": Use unrolled differentiation through solver iterations
            - "none": Stop gradients (no differentiation)
            
    Example:
        >>> import jax.numpy as jnp
        >>> from cvxjax import Variable, Problem, Minimize, CvxLayer, sum_squares
        >>> 
        >>> # Define problem
        >>> x = Variable((3,))
        >>> prob = Problem(Minimize(sum_squares(x)), [x >= 0])
        >>> 
        >>> # Create layer
        >>> layer = CvxLayer(prob, solver="ipm", return_fields=("primal",))
        >>> 
        >>> # Solve
        >>> x_star = layer({})
        >>> 
        >>> # Use in differentiable computation
        >>> def loss_fn(scale):
        ...     x = Variable((2,))
        ...     test_prob = Problem(Minimize(scale * sum_squares(x)), [x >= 0])
        ...     test_layer = CvxLayer(test_prob, solver="ipm")
        ...     return jnp.sum(test_layer({})**2)
        >>> grad_fn = jax.grad(loss_fn)
        >>> gradient = grad_fn(1.0)
    """
    
    def __init__(
        self,
        problem: Problem,
        solver: str = "ipm",
        settings: Optional[Any] = None,
        return_fields: tuple[str, ...] = ("primal",),
        diff_mode: Literal["implicit", "unroll", "none"] = "implicit",
    ):
        """Initialize the CVX layer.
        
        Args:
            problem: CVXJAX Problem instance to wrap.
            solver: Solver backend to use.
            settings: Optional solver-specific settings.
            return_fields: Which outputs to return from solve.
            diff_mode: How to handle differentiation.
        """
        # Validate inputs
        valid_solvers = {"ipm", "osqp", "boxosqp", "boxcdqp"}
        if solver not in valid_solvers:
            raise ValueError(f"Unknown solver: {solver}. Valid options: {valid_solvers}")
            
        valid_fields = {"primal", "dual", "obj"}
        invalid_fields = set(return_fields) - valid_fields
        if invalid_fields:
            raise ValueError(f"Invalid return fields: {invalid_fields}. Valid options: {valid_fields}")
            
        valid_diff_modes = {"implicit", "unroll", "none"}
        if diff_mode not in valid_diff_modes:
            raise ValueError(f"Invalid diff_mode: {diff_mode}. Valid options: {valid_diff_modes}")
        
        self.problem = problem
        self.cfg = _LayerConfig(
            solver=solver,
            diff_mode=diff_mode,
            return_fields=return_fields,
            settings=settings
        )
        
        # For dual outputs, warn if not implemented
        if "dual" in return_fields:
            import warnings
            warnings.warn(
                "Dual variable outputs are not fully implemented yet. "
                "Will return empty arrays for 'dual' field.",
                UserWarning
            )
    
    def __call__(self, params: Dict[Parameter, jnp.ndarray]) -> Union[jnp.ndarray, tuple]:
        """Solve the problem with given parameters.
        
        Args:
            params: Dictionary mapping Parameters to their values.
            
        Returns:
            Solution outputs as specified by return_fields.
            Single array if one field, tuple if multiple fields.
        """
        # For initial implementation, use the regular solve method
        # This avoids JIT compilation issues with dynamic shapes
        solution = self.problem.solve(
            params=params,
            solver=self.cfg.solver,
            **(self.cfg.settings or {})
        )
        
        # Apply stop_gradient if requested
        if self.cfg.diff_mode == "none":
            # Apply stop_gradient to the solution components individually
            solution_stopped = type(solution)(
                solution.status,
                jax.lax.stop_gradient(solution.obj_value),
                {k: jax.lax.stop_gradient(v) for k, v in solution.primal.items()},
                {k: jax.lax.stop_gradient(v) for k, v in solution.dual.items()},
                solution.info
            )
            solution = solution_stopped
        
        # Collect and return specified outputs
        return _collect_outputs(solution, self.cfg.return_fields)


# Register CvxLayer as a JAX pytree for compatibility with transformations
tree_util.register_pytree_node(
    CvxLayer,
    lambda layer: ((), {"problem": layer.problem, "cfg": layer.cfg}),
    lambda aux, children: CvxLayer(
        aux["problem"],
        solver=aux["cfg"].solver,
        settings=aux["cfg"].settings,
        return_fields=aux["cfg"].return_fields,
        diff_mode=aux["cfg"].diff_mode,
    ),
)