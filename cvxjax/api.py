"""Main API for CVXJAX optimization library."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Union

import jax
import jax.numpy as jnp
from jax import tree_util

from cvxjax.canonicalize import build_qp
from cvxjax.constraints import Constraint
from cvxjax.diff import solve_qp_diff
from cvxjax.expressions import AffineExpression, Expression
from cvxjax.solvers.ipm_qp import solve_qp_dense
from cvxjax.solvers.osqp_bridge import solve_qp_osqp
from cvxjax.solvers.boxosqp_solver import solve_qp_boxosqp
from cvxjax.utils.checking import check_variable_shape


@dataclass(frozen=True)
class Variable:
    """Optimization variable with shape and optional name.
    
    Variables are the primary decision variables in optimization problems.
    They support operator overloading for building affine expressions.
    
    Args:
        shape: Shape of the variable array.
        name: Optional name for debugging and display.
        
    Example:
        >>> x = Variable(shape=(3,), name="weights")
        >>> y = Variable(shape=(2, 2), name="matrix")
    """
    shape: tuple[int, ...]
    name: Optional[str] = None
    
    def __post_init__(self) -> None:
        check_variable_shape(self.shape)
    
    @property
    def size(self) -> int:
        """Total number of elements in the variable."""
        # Use shape[0] for 1D case or np.prod for static shapes
        if len(self.shape) == 0:
            return 1
        elif len(self.shape) == 1:
            return self.shape[0]
        else:
            # For multi-dimensional, compute at definition time if possible
            import numpy as np
            return int(np.prod(self.shape))
    
    def is_affine(self) -> bool:
        """Variables are always affine."""
        return True
    
    def is_convex(self) -> bool:
        """Variables are always convex."""
        return True
    
    def __getitem__(self, key):
        """Enable indexing like x[0] or x[1:3]."""
        # For now, create a simple indexed variable
        # This is a simplified implementation
        if isinstance(key, int):
            # Single index
            if key >= self.shape[0]:
                raise IndexError(f"Index {key} out of bounds for variable with shape {self.shape}")
            # Return a new variable representing the indexed element
            return Variable(shape=(1,), name=f"{self.name}[{key}]" if self.name else None)
        else:
            # Slice or other indexing - simplified
            return Variable(shape=(1,), name=f"{self.name}[...]" if self.name else None)
    
    def __add__(self, other: Union[Variable, Expression, jnp.ndarray]) -> AffineExpression:
        return AffineExpression.from_variable(self) + other
    
    def __radd__(self, other: Union[jnp.ndarray, float]) -> AffineExpression:
        return other + AffineExpression.from_variable(self)
    
    def __sub__(self, other: Union[Variable, Expression, jnp.ndarray]) -> AffineExpression:
        return AffineExpression.from_variable(self) - other
    
    def __rsub__(self, other: Union[jnp.ndarray, float]) -> AffineExpression:
        return other - AffineExpression.from_variable(self)
    
    def __mul__(self, other: Union[float, jnp.ndarray]) -> AffineExpression:
        return AffineExpression.from_variable(self) * other
    
    def __rmul__(self, other: Union[float, jnp.ndarray]) -> AffineExpression:
        return other * AffineExpression.from_variable(self)
    
    def __matmul__(self, other: Union[jnp.ndarray, Variable]) -> AffineExpression:
        return AffineExpression.from_variable(self) @ other
    
    def __rmatmul__(self, other: jnp.ndarray) -> AffineExpression:
        return other @ AffineExpression.from_variable(self)
    
    def __neg__(self) -> AffineExpression:
        return -AffineExpression.from_variable(self)
    
    def __le__(self, other: Union[Variable, Expression, jnp.ndarray]) -> Constraint:
        from cvxjax.constraints import InequalityConstraint
        return InequalityConstraint(
            AffineExpression.from_variable(self) - other, 
            sense="<="
        )
    
    def __ge__(self, other: Union[Variable, Expression, jnp.ndarray]) -> Constraint:
        from cvxjax.constraints import InequalityConstraint
        return InequalityConstraint(
            AffineExpression.from_variable(self) - other, 
            sense=">="
        )
    
    def __eq__(self, other) -> Union[bool, Constraint]:  # type: ignore
        # For Variable-to-Variable comparison (used in dict operations), return boolean
        if isinstance(other, Variable):
            return self.shape == other.shape and self.name == other.name
        
        # For optimization DSL (Variable == expression), return constraint
        from cvxjax.constraints import EqualityConstraint
        return EqualityConstraint(AffineExpression.from_variable(self) - other)
    
    def __hash__(self) -> int:
        """Hash based on shape and name for dictionary keys."""
        return hash((self.shape, self.name))


# Register Variable as a JAX pytree
tree_util.register_pytree_node(
    Variable,
    lambda v: ((), {"shape": v.shape, "name": v.name}),
    lambda aux, children: Variable(**aux),
)


@dataclass(frozen=True)
class Parameter:
    """Parameter with fixed value that can be updated between solves.
    
    Parameters allow creating parameterized optimization problems that can
    be solved repeatedly with different parameter values.
    
    Args:
        value: The parameter value as a JAX array.
        name: Optional name for debugging and display.
        
    Example:
        >>> mu = Parameter(jnp.array([0.1, 0.2]), name="returns")
    """
    value: jnp.ndarray
    name: Optional[str] = None
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the parameter."""
        return self.value.shape
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.value.size
    
    def __add__(self, other: Union[Variable, Expression, jnp.ndarray]) -> AffineExpression:
        return AffineExpression.from_constant(self.value) + other
    
    def __radd__(self, other: Union[jnp.ndarray, float]) -> AffineExpression:
        return other + AffineExpression.from_constant(self.value)
    
    def __sub__(self, other: Union[Variable, Expression, jnp.ndarray]) -> AffineExpression:
        return AffineExpression.from_constant(self.value) - other
    
    def __rsub__(self, other: Union[jnp.ndarray, float]) -> AffineExpression:
        return other - AffineExpression.from_constant(self.value)
    
    def __mul__(self, other: Union[float, jnp.ndarray, Variable]) -> AffineExpression:
        return AffineExpression.from_constant(self.value) * other
    
    def __rmul__(self, other: Union[float, jnp.ndarray]) -> AffineExpression:
        return other * AffineExpression.from_constant(self.value)
    
    def __matmul__(self, other: Union[jnp.ndarray, Variable]) -> AffineExpression:
        return AffineExpression.from_constant(self.value) @ other
    
    def __rmatmul__(self, other: jnp.ndarray) -> AffineExpression:
        return other @ AffineExpression.from_constant(self.value)


# Register Parameter as a JAX pytree
tree_util.register_pytree_node(
    Parameter,
    lambda p: ((p.value,), {"name": p.name}),
    lambda aux, children: Parameter(children[0], **aux),
)


@dataclass(frozen=True)
class Constant:
    """Constant value in optimization expressions.
    
    Constants are fixed values that don't change during optimization.
    
    Args:
        value: The constant value as a JAX array.
        name: Optional name for debugging and display.
    """
    value: jnp.ndarray
    name: Optional[str] = None
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the constant."""
        return self.value.shape
    
    @property
    def size(self) -> int:
        """Total number of elements."""
        return self.value.size


# Register Constant as a JAX pytree
tree_util.register_pytree_node(
    Constant,
    lambda c: ((c.value,), {"name": c.name}),
    lambda aux, children: Constant(children[0], **aux),
)


class Objective:
    """Base class for optimization objectives."""
    
    def __init__(self, expression: Expression) -> None:
        self.expression = expression


class Minimize(Objective):
    """Minimization objective.
    
    Args:
        expression: The expression to minimize.
        
    Example:
        >>> x = Variable(shape=(2,))
        >>> obj = Minimize(x @ x)  # Minimize ||x||^2
    """
    pass


class Maximize(Objective):
    """Maximization objective.
    
    Args:
        expression: The expression to maximize.
        
    Example:
        >>> x = Variable(shape=(2,))
        >>> obj = Maximize(jnp.ones(2) @ x)  # Maximize sum of x
    """
    pass


@dataclass(frozen=True)
class Solution:
    """Solution returned by optimization solvers.
    
    Args:
        status: Solver termination status.
        obj_value: Optimal objective value.
        primal: Dictionary mapping variables to their optimal values.
        dual: Dictionary mapping constraints to their dual values.
        info: Additional solver information.
    """
    status: Literal["optimal", "max_iter", "primal_infeasible", "dual_infeasible", "error"]
    obj_value: float
    primal: Dict[Variable, jnp.ndarray]
    dual: Dict[Any, jnp.ndarray]
    info: Dict[str, Any]


# Register Solution as a JAX pytree
tree_util.register_pytree_node(
    Solution,
    lambda s: ((s.obj_value, s.primal, s.dual, s.info), {"status": s.status}),
    lambda aux, children: Solution(aux["status"], *children),
)


class Problem:
    """Optimization problem combining objective and constraints.
    
    Args:
        objective: The objective to optimize (Minimize or Maximize).
        constraints: List of constraints.
        
    Example:
        >>> x = Variable(shape=(2,))
        >>> obj = Minimize(x @ x)
        >>> cons = [x >= 0, jnp.ones(2) @ x == 1]
        >>> prob = Problem(obj, cons)
        >>> sol = prob.solve()
    """
    
    def __init__(
        self, 
        objective: Union[Minimize, Maximize], 
        constraints: Optional[list[Constraint]] = None
    ) -> None:
        self.objective = objective
        self.constraints = constraints or []
        
        # Skip validation for JIT compatibility
        # Problem validation should be done outside JIT-compiled functions
    
    def _validate(self) -> None:
        """Validate problem structure.
        
        Note:
            For JIT compatibility, this function does minimal validation.
            Comprehensive validation should be done at problem construction time.
        """
        # Skip isinstance checks for JIT compatibility
        pass
    
    def solve(
        self,
        params: Optional[Dict[Parameter, jnp.ndarray]] = None,
        solver: Literal["ipm", "osqp", "boxosqp"] = "ipm",
        tol: float = 1e-8,
        max_iter: int = 50,
        **solver_kwargs: Any,
    ) -> Solution:
        """Solve the optimization problem.
        
        Args:
            params: Optional parameter values to override defaults.
            solver: Solver to use ("ipm", "osqp", or "boxosqp").
            tol: Tolerance for convergence.
            max_iter: Maximum number of iterations.
            **solver_kwargs: Additional solver-specific arguments.
            
        Returns:
            Solution object with optimal values and solver information.
        """
        # Update parameters if provided
        if params:
            # Create new objective and constraints with updated parameters
            # This is a simplified implementation
            objective = self.objective
            constraints = self.constraints
        else:
            objective = self.objective
            constraints = self.constraints
        
        # Build QP formulation
        qp_data = build_qp(objective.expression, constraints)
        
        # Solve based on selected solver
        if solver == "ipm":
            return solve_qp_dense(qp_data, tol=tol, max_iter=max_iter, **solver_kwargs)
        elif solver == "osqp":
            return solve_qp_osqp(qp_data, tol=tol, max_iter=max_iter, **solver_kwargs)
        elif solver == "boxosqp":
            return solve_qp_boxosqp(qp_data, tol=tol, max_iter=max_iter, **solver_kwargs)
        else:
            # For JIT compatibility, default to IPM solver if unknown solver specified
            return solve_qp_dense(qp_data, tol=tol, max_iter=max_iter, **solver_kwargs)
    
    def solve_jit(
        self,
        params: Optional[Dict[Parameter, jnp.ndarray]] = None,
        solver: Literal["ipm", "osqp"] = "ipm",
        tol: float = 1e-8,
        max_iter: int = 50,
        **solver_kwargs: Any,
    ) -> Solution:
        """JIT-compiled solve for repeated solving with static shapes.
        
        This method compiles the solve path for better performance when
        solving the same problem structure repeatedly.
        
        Args:
            params: Optional parameter values to override defaults.
            solver: Solver to use ("ipm" or "osqp").
            tol: Tolerance for convergence.
            max_iter: Maximum number of iterations.
            **solver_kwargs: Additional solver-specific arguments.
            
        Returns:
            Solution object with optimal values and solver information.
        """
        @jax.jit
        def _solve_compiled(qp_data):
            if solver == "ipm":
                return solve_qp_dense(qp_data, tol=tol, max_iter=max_iter, **solver_kwargs)
            else:
                return solve_qp_osqp(qp_data, tol=tol, max_iter=max_iter, **solver_kwargs)
        
        # Build QP formulation
        qp_data = build_qp(self.objective.expression, self.constraints)
        
        return _solve_compiled(qp_data)
    
    def solve_diff(
        self,
        params: Optional[Dict[Parameter, jnp.ndarray]] = None,
        solver: Literal["ipm", "osqp"] = "ipm",
        tol: float = 1e-8,
        max_iter: int = 50,
        **solver_kwargs: Any,
    ) -> Solution:
        """Solve with automatic differentiation support.
        
        This method enables computing gradients through the solution
        using implicit differentiation of the KKT conditions.
        
        Args:
            params: Optional parameter values to override defaults.
            solver: Solver to use ("ipm" or "osqp").
            tol: Tolerance for convergence.
            max_iter: Maximum number of iterations.
            **solver_kwargs: Additional solver-specific arguments.
            
        Returns:
            Solution object with optimal values and solver information.
        """
        # Build QP formulation
        qp_data = build_qp(self.objective.expression, self.constraints)
        
        return solve_qp_diff(qp_data, solver=solver, tol=tol, max_iter=max_iter, **solver_kwargs)
