# CVXJAX

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A JAX-native convex optimization library providing an MVP analogue of CVXPY with automatic differentiation through the solution map.

## Features

- **JAX-native modeling API** for linear programs (LP) and quadratic programs (QP)
- **Affine constraints** and simple box/nonnegativity bounds
- **Two solver backends**:
  - Dense primal-dual interior point method in pure JAX
  - Adapter to `jaxopt.OSQP` for large or sparse problems
- **Automatic differentiation** through solution via implicit differentiation of KKT system
- **JIT and vmap friendly** with static shape support
- **Clean, typed API** with comprehensive docstrings

## Installation

```bash
pip install cvxjax
```

For development:
```bash
git clone https://github.com/cvxjax/cvxjax.git
cd cvxjax
pip install -e .[dev]
```

## Quick Start

```python
import jax.numpy as jnp
import cvxjax as cx

# Enable 64-bit precision for numerical stability
import jax
jax.config.update("jax_enable_x64", True)

# Define optimization variables
x = cx.Variable(shape=(2,), name="x")

# Define quadratic objective
Q = jnp.array([[2.0, 0.5], [0.5, 1.0]])
q = jnp.array([1.0, 1.0])
objective = cx.Minimize(0.5 * cx.quad_form(x, Q) + q @ x)

# Add constraints
constraints = [
    x >= 0,  # Nonnegativity
    jnp.ones(2) @ x == 1,  # Budget constraint
]

# Create and solve problem
problem = cx.Problem(objective, constraints)
solution = problem.solve(solver="ipm", tol=1e-8)

print(f"Status: {solution.status}")
print(f"Optimal value: {solution.obj_value:.4f}")
print(f"Optimal x: {solution.primal[x]}")
```

## Differentiable Optimization

```python
import jax

# Define a parameterized problem
def solve_portfolio(risk_aversion):
    mu = jnp.array([0.1, 0.2])  # Expected returns
    Sigma = jnp.array([[0.1, 0.02], [0.02, 0.15]])  # Covariance
    
    x = cx.Variable(shape=(2,), name="weights")
    
    # Risk-return tradeoff
    objective = cx.Minimize(
        risk_aversion * 0.5 * cx.quad_form(x, Sigma) - mu @ x
    )
    
    constraints = [x >= 0, jnp.ones(2) @ x == 1]
    problem = cx.Problem(objective, constraints)
    
    return problem.solve_jit(solver="ipm")

# Compute gradient of optimal value w.r.t. risk aversion
grad_fn = jax.grad(lambda ra: solve_portfolio(ra).obj_value)
gradient = grad_fn(2.0)
print(f"Gradient: {gradient}")
```

## MVP Feature List

### Modeling
- [x] `Variable`, `Parameter`, `Constant` with shape and dtype
- [x] Operator overloading for affine expressions (`+`, `-`, `@`)
- [x] Constraint creation via comparisons (`<=`, `==`, `>=`)
- [x] Basic atoms: `sum_squares`, `quad_form`, `abs`, `square`

### Solvers
- [x] Dense primal-dual interior point method (pure JAX)
- [x] OSQP adapter via `jaxopt.OSQP`
- [x] Unified `Solution` interface with status and dual variables

### Differentiation
- [x] Implicit differentiation through KKT conditions
- [x] `custom_vjp` for efficient reverse-mode AD
- [x] JIT-compiled solve paths

### Utilities
- [x] Shape validation and static shape support
- [x] PSD checking for quadratic forms
- [x] Diagonal scaling and conditioning

## Documentation

- [Quickstart Guide](docs/quickstart.md)
- [Core Concepts](docs/concepts.md)
- [Troubleshooting](docs/troubleshooting.md)

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
