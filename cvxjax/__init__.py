"""CVXJAX: JAX-native convex optimization library."""

from cvxjax.api import (
    Constant,
    Maximize,
    Minimize,
    Parameter,
    Problem,
    Solution,
    Variable,
)
from cvxjax.atoms import abs, quad_form, square, sum, sum_squares
from cvxjax.constraints import Constraint

__version__ = "0.1.0"

__all__ = [
    "Variable",
    "Parameter",
    "Constant",
    "Minimize",
    "Maximize",
    "Problem",
    "Solution",
    "Constraint",
    "quad_form",
    "sum",
    "sum_squares",
    "square",
    "abs",
]
