# CVXJAX Documentation

**CVXJAX** is a JAX-native convex optimization library that brings the expressiveness of CVXPY to the JAX ecosystem. It enables high-performance, differentiable optimization with JIT compilation, automatic differentiation, and GPU acceleration.

## Why CVXJAX?

- 🚀 **High Performance**: JIT compilation and GPU acceleration through JAX
- 🎯 **Automatic Differentiation**: Differentiable optimization for machine learning
- 📝 **Familiar API**: Similar to CVXPY for easy adoption
- 🔧 **JAX Integration**: Seamless integration with the JAX ecosystem
- ⚡ **Multiple Solvers**: Interior point, OSQP, and specialized solvers

## Quick Example

```python
import cvxjax as cx
import jax.numpy as jnp

# Portfolio optimization
w = cx.Variable(shape=(n_assets,), name="weights")
objective = cx.Minimize(-returns @ w + gamma * cx.quad_form(w, Sigma))
constraints = [cx.sum(w) == 1, w >= 0]

problem = cx.Problem(objective, constraints)
solution = problem.solve(solver="ipm")

if solution.status == "optimal":
    optimal_weights = solution.primal[w]
```

## Documentation Structure

### � **Learning Resources**
- **[Getting Started](getting_started.md)** - Step-by-step tutorial with examples
- **[Quick Reference](quick_reference.md)** - Concise API overview and common patterns
- **[API Reference](api_reference.md)** - Complete function documentation

### 🎯 **Practical Guides**
- **[Concepts](concepts.md)** - Core optimization concepts and mathematical background
- **[Troubleshooting](troubleshooting.md)** - Common issues and solutions

### 💡 **Examples**
- **[Portfolio Optimization](../examples/)** - Mean-variance optimization, risk models
- **[Machine Learning](../examples/)** - Regression, regularization, SVM
- **[Engineering](../examples/)** - Control, signal processing, network optimization

### 🔧 Multiple Solvers
- **IPM**: Dense interior-point method (pure JAX)
- **OSQP**: Sparse solver via jaxopt bridge
- Extensible solver interface

### 📐 Type Safety
- Full type hints throughout
- Static shape checking
- Clear error messages

## Installation

### From PyPI (when available)
```bash
pip install cvxjax
```

### Development Installation
```bash
git clone https://github.com/your-org/cvxjax.git
cd cvxjax
pip install -e ".[dev]"
```

## Core Concepts

### Variables and Parameters
```python
# Optimization variables (to be solved for)
x = cx.Variable(shape=(10,), name="weights")
y = cx.Variable(shape=(5, 5), name="matrix")

# Parameters (fixed data)
A = cx.Parameter(value=jnp.ones((3, 10)), name="data_matrix")
b = cx.Parameter(value=jnp.zeros(3), name="targets")
```

### Expressions and Atoms
```python
# Linear and quadratic expressions
linear_expr = A @ x + b
quadratic_expr = cx.sum_squares(x)
quad_form_expr = cx.quad_form(x, P)

# Arithmetic operations
combined = 2 * linear_expr + quadratic_expr
```

### Constraints
```python
# Equality and inequality constraints
constraints = [
    A @ x == b,           # Equality
    x >= 0,               # Element-wise inequality  
    cx.sum(x) <= 1,       # Scalar inequality
    cx.Box(x, 0, 1)       # Box constraints
]
```

### Problem and Solution
```python
# Create problem
problem = cx.Problem(
    objective=cx.Minimize(objective_expr),
    constraints=constraints
)

# Solve with different methods
solution = problem.solve()                    # Default solver
solution = problem.solve(solver="ipm")        # Interior-point
solution = problem.solve(solver="osqp")       # OSQP
solution = problem.solve_jit()                # JIT-compiled

# Access results
if solution.status == "optimal":
    optimal_x = solution.primal["x"]
    optimal_value = solution.obj_value
```

## Advanced Usage

### Differentiation Through Solutions
```python
def solve_qp(P, q):
    x = cx.Variable(shape=(n,))
    problem = cx.Problem(cx.Minimize(0.5 * cx.quad_form(x, P) + q @ x))
    solution = problem.solve_jit()
    return solution.obj_value

# Compute gradients w.r.t. problem data
grad_fn = jax.grad(solve_qp, argnums=(0, 1))
dP, dq = grad_fn(P, q)
```

### Batch Processing with vmap
```python
# Solve multiple problems in parallel
batch_solve = jax.vmap(lambda q: solve_qp(P, q))
batch_solutions = batch_solve(q_batch)
```

### Custom Solver Configuration
```python
solution = problem.solve(
    solver="ipm",
    tol=1e-8,
    max_iters=100,
    verbose=True
)
```

## Examples

The `examples/` directory contains complete working examples:

- **quickstart_qp.py**: Basic quadratic programming
- **lasso_training_loop.py**: Differentiable LASSO with hyperparameter optimization
- **portfolio_qp.py**: Portfolio optimization with various constraints

## API Reference

### Core Classes
- `Variable`: Optimization variables
- `Parameter`: Problem parameters  
- `Expression`: Mathematical expressions
- `Problem`: Optimization problems
- `Solution`: Solution objects

### Atoms (Functions)
- `sum_squares(x)`: Sum of squares
- `quad_form(x, P)`: Quadratic form x^T P x
- `sum(x)`: Sum reduction
- `norm(x, p)`: Vector norms (p=1,2,inf)

### Constraints
- `Equality`: Equality constraints
- `Inequality`: Inequality constraints  
- `Box`: Box constraints

## Performance Tips

1. **Use JIT compilation**: `problem.solve_jit()` for repeated solves
2. **Static shapes**: Avoid dynamic shapes when possible
3. **Warm-up compilation**: JIT compile once before timing
4. **Appropriate solver**: Use IPM for dense problems, OSQP for sparse
5. **Batch processing**: Use vmap for multiple similar problems

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/your-org/cvxjax.git
cd cvxjax
make setup  # Install dependencies and pre-commit hooks
make test   # Run tests
make lint   # Check code quality
```

## License

CVXJAX is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

## Citation

If you use CVXJAX in your research, please cite:

```bibtex
@software{cvxjax2024,
  title={CVXJAX: JAX-Native Convex Optimization},
  author={Your Name},
  year={2024},
  url={https://github.com/your-org/cvxjax}
}
```
