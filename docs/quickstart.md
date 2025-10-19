# Quick Start Guide

This guide will get you up and running with CVXJAX in 5 minutes. For detailed documentation, see [API Reference](api_reference.md).

## Installation

### Basic Installation
```bash
pip install cvxjax  # When available on PyPI
```

### Development Installation
```bash
git clone https://github.com/your-org/cvxjax.git
cd cvxjax
pip install -e ".[dev]"
```

## Your First Problem

Let's solve a simple quadratic program:

```python
import jax.numpy as jnp
import cvxjax as cx

# Problem: minimize ||x - [1, 2]||^2 subject to x >= 0, sum(x) <= 3

# Step 1: Define variables
x = cx.Variable(shape=(2,), name="x")

# Step 2: Define objective function  
target = jnp.array([1.0, 2.0])
objective = cx.Minimize(cx.sum_squares(x - target))

# Step 3: Define constraints
constraints = [
    x >= 0,           # Non-negativity
    cx.sum(x) <= 3    # Budget constraint
]

# Step 4: Create and solve problem
problem = cx.Problem(objective, constraints)
solution = problem.solve()

# Step 5: Check results
if solution.status == "optimal":
    print(f"Optimal value: {solution.obj_value:.4f}")
    print(f"Optimal x: {solution.primal['x']}")
else:
    print(f"Solver failed: {solution.status}")
```

Expected output:
```
Optimal value: 0.5000
Optimal x: [1. 2.]
```

## Key Concepts

### 1. Variables vs Parameters

**Variables** are unknowns that the solver optimizes:
```python
x = cx.Variable(shape=(5,), name="portfolio_weights")
y = cx.Variable(shape=(3, 3), name="covariance_matrix")
```

**Parameters** are fixed data:
```python
returns = cx.Parameter(value=jnp.array([0.1, 0.05, 0.08]), name="expected_returns")
```

### 2. Building Expressions

Combine variables with mathematical operations:
```python
# Linear expression
linear_expr = A @ x + b

# Quadratic expression  
quadratic_expr = cx.sum_squares(x)

# Mixed expression
total_expr = 0.5 * quadratic_expr + cx.sum(linear_expr)
```

### 3. Constraint Types

```python
constraints = [
    # Equality constraints
    A @ x == b,
    cx.sum(x) == 1,
    
    # Inequality constraints  
    x >= 0,
    x <= upper_bounds,
    cx.sum(x) <= budget,
    
    # Box constraints (shorthand)
    cx.Box(x, lower=0, upper=1)
]
```

## Common Patterns

### Portfolio Optimization
```python
import cvxjax as cx
import jax.numpy as jnp

# Data
n_assets = 5
expected_returns = jnp.array([0.1, 0.12, 0.08, 0.09, 0.11])
risk_aversion = 1.0

# Covariance matrix (simplified)
Sigma = jnp.eye(n_assets) * 0.01 + jnp.ones((n_assets, n_assets)) * 0.005

# Optimization
w = cx.Variable(shape=(n_assets,), name="weights")

# Objective: maximize return - risk penalty
expected_return = expected_returns @ w
risk = cx.quad_form(w, Sigma)
objective = cx.Minimize(-expected_return + 0.5 * risk_aversion * risk)

# Constraints
constraints = [
    cx.sum(w) == 1,  # Budget
    w >= 0           # Long-only
]

# Solve
problem = cx.Problem(objective, constraints)
solution = problem.solve()

print(f"Optimal weights: {solution.primal['weights']}")
```

### Least Squares with Regularization
```python
# Data
n, p = 100, 20
A = jax.random.normal(jax.random.PRNGKey(0), (n, p))
b = jax.random.normal(jax.random.PRNGKey(1), (n,))
lambda_reg = 0.1

# Optimization  
x = cx.Variable(shape=(p,), name="coefficients")

# Objective: ||Ax - b||^2 + lambda ||x||^2
data_fit = cx.sum_squares(A @ x - b)
regularization = lambda_reg * cx.sum_squares(x)
objective = cx.Minimize(data_fit + regularization)

# Solve (unconstrained)
problem = cx.Problem(objective, constraints=[])
solution = problem.solve()

print(f"Optimal coefficients: {solution.primal['coefficients']}")
```

## Solver Options

### Available Solvers

**IPM (Interior Point Method)**
- Dense QP solver
- Pure JAX implementation
- Good for small-to-medium problems
- Supports JIT compilation

```python
solution = problem.solve(solver="ipm", tol=1e-8, max_iters=100)
```

**OSQP**
- Sparse QP solver  
- Uses jaxopt.OSQP
- Good for large sparse problems
- May have limited JIT support

```python
solution = problem.solve(solver="osqp", eps_abs=1e-6, eps_rel=1e-6)
```

### Performance Options

**JIT Compilation** (recommended for repeated solves):
```python
# Compile once, reuse many times
solution = problem.solve_jit(solver="ipm")
```

**Batch Processing** with vmap:
```python
# Solve multiple similar problems
def solve_single(data):
    # ... define problem with data ...
    return problem.solve_jit()

# Vectorize over first dimension
batch_solve = jax.vmap(solve_single)
solutions = batch_solve(data_batch)
```

## Common Issues and Solutions

### Issue: "Shapes don't match"
**Cause**: Incompatible array shapes in expressions
**Solution**: Check variable and parameter shapes carefully

```python
# ❌ Wrong
x = cx.Variable(shape=(5,))
A = jnp.ones((3, 4))  # Incompatible!
expr = A @ x

# ✅ Correct  
x = cx.Variable(shape=(4,))
A = jnp.ones((3, 4))
expr = A @ x  # Results in shape (3,)
```

### Issue: "Problem is infeasible"
**Cause**: No solution satisfies all constraints
**Solution**: 
1. Check constraint compatibility
2. Relax some constraints
3. Add slack variables

```python
# Add slack for soft constraints
slack = cx.Variable(shape=(n,), name="slack")
constraints = [
    A @ x <= b + slack,
    slack >= 0
]
# Add slack penalty to objective
objective += penalty_weight * cx.sum(slack)
```

### Issue: "Solver failed to converge"
**Cause**: Numerical issues or poor conditioning
**Solutions**:
1. Increase tolerance
2. Scale problem data
3. Add regularization

```python
# Scale data to reasonable range
A_scaled = A / jnp.linalg.norm(A, axis=0, keepdims=True)
b_scaled = b / jnp.linalg.norm(b)

# Add regularization
objective += 1e-6 * cx.sum_squares(x)
```

## Next Steps

1. **Try the examples**: Run `python examples/quickstart_qp.py`
2. **Read the concepts guide**: [Concepts](concepts.md)
3. **Check troubleshooting**: [Troubleshooting](troubleshooting.md)
4. **Explore advanced features**: Differentiation, custom solvers

## Need Help?

- **GitHub Issues**: Report bugs and request features
- **Documentation**: This site has detailed guides
- **Examples**: Check the `examples/` directory
