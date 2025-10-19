# Troubleshooting Guide

This guide helps you diagnose and fix common issues when using CVXJAX.

## Installation Issues

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'cvxjax'`

**Solutions**:
```bash
# Check if cvxjax is installed
pip list | grep cvxjax

# Install in development mode
pip install -e .

# Install with all dependencies
pip install -e ".[dev]"
```

**Error**: `ImportError: cannot import name 'X' from 'jax'`

**Cause**: JAX version incompatibility

**Solutions**:
```bash
# Update JAX
pip install --upgrade jax jaxlib

# Check versions
python -c "import jax; print(jax.__version__)"

# CVXJAX requires JAX >= 0.4.20
```

## Problem Formulation Issues

### Shape Errors

**Error**: `ShapeError: Cannot add expressions with shapes (5,) and (3,)`

**Cause**: Incompatible shapes in mathematical operations

**Debug Steps**:
```python
# Check variable shapes
print(f"x.shape: {x.shape}")
print(f"y.shape: {y.shape}")

# Check expression shapes
expr1 = A @ x
expr2 = B @ y
print(f"expr1.shape: {expr1.shape}")
print(f"expr2.shape: {expr2.shape}")

# Fix: ensure compatible shapes
assert A.shape[1] == x.shape[0], "Matrix-vector multiply shape mismatch"
```

**Common Solutions**:
```python
# ❌ Wrong: mixing scalars and vectors
x = cx.Variable(shape=(5,))
constraint = x >= 1.0  # Broadcasting issue

# ✅ Correct: consistent shapes  
x = cx.Variable(shape=(5,))
constraint = x >= jnp.ones(5)  # Or: x >= 1.0 (JAX handles broadcasting)

# ❌ Wrong: matrix dimension mismatch
A = jnp.ones((3, 4))
x = cx.Variable(shape=(5,))  # Should be (4,)
expr = A @ x

# ✅ Correct: matching dimensions
A = jnp.ones((3, 4))
x = cx.Variable(shape=(4,))
expr = A @ x  # Results in shape (3,)
```

### Variable Definition Issues

**Error**: `ValueError: Variable names must be unique`

**Solution**:
```python
#  Wrong: duplicate names
x1 = cx.Variable(shape=(3,), name="x")
x2 = cx.Variable(shape=(5,), name="x")  # Same name!

#  Correct: unique names
x1 = cx.Variable(shape=(3,), name="weights")
x2 = cx.Variable(shape=(5,), name="portfolio")
```

**Error**: `TypeError: Variable shape must be a tuple of integers`

**Solution**:
```python
#  Wrong: non-integer or non-tuple shapes
x = cx.Variable(shape=5)          # Should be (5,)
y = cx.Variable(shape=(3.0,))     # Should be (3,)
z = cx.Variable(shape=[2, 3])     # Should be (2, 3)

#  Correct: tuple of integers
x = cx.Variable(shape=(5,))
y = cx.Variable(shape=(3,))
z = cx.Variable(shape=(2, 3))
```

### Constraint Issues

**Error**: `ConvexityError: Expression is not convex`

**Cause**: Non-convex expressions in constraints or objective

**Debug Steps**:
```python
# Check if expression is convex
print(f"Expression is convex: {expr.is_convex()}")
print(f"Expression is affine: {expr.is_affine()}")

# Common non-convex operations
x @ x          # x^T x (convex in minimization, concave in maximization)
x * y          # Product of variables (non-convex)
x / y          # Division by variable (non-convex)
```

**Solutions**:
```python
#  Wrong: non-convex constraint
x = cx.Variable(shape=(2,))
constraint = x[0] * x[1] >= 1  # Non-convex

#  Correct: convex reformulation (if possible)
# Use sum_squares or quad_form for quadratic terms
objective = cx.Minimize(cx.sum_squares(x))

#  Wrong: maximizing convex function
objective = cx.Maximize(cx.sum_squares(x))  # Not convex!

# ✅ Correct: minimize convex function
objective = cx.Minimize(cx.sum_squares(x))
```

## Solver Issues

### Convergence Problems

**Error**: `SolverError: Failed to converge within maximum iterations`

**Causes and Solutions**:

1. **Poor conditioning**:
```python
# Add regularization
regularization = 1e-6 * cx.sum_squares(x)
objective += regularization
```

2. **Infeasible problem**:
```python
# Check constraint compatibility
# Add slack variables for soft constraints
slack = cx.Variable(shape=(m,), name="slack")
constraints = [
    A @ x <= b + slack,
    slack >= 0
]
objective += 1000 * cx.sum(slack)  # Large penalty
```

3. **Poor scaling**:
```python
# Scale problem data
A_scaled = A / jnp.linalg.norm(A, axis=0, keepdims=True)
b_scaled = b / jnp.linalg.norm(b)

# Or use solver-specific scaling
solution = problem.solve(solver="ipm", scale_problem=True)
```

4. **Increase solver tolerance**:
```python
solution = problem.solve(
    solver="ipm",
    tol=1e-6,        # Looser tolerance
    max_iters=200    # More iterations
)
```

### Infeasibility Issues

**Error**: `SolverError: Problem is infeasible`

**Debug Steps**:

1. **Check individual constraints**:
```python
# Test each constraint separately
for i, constraint in enumerate(constraints):
    test_problem = cx.Problem(cx.Minimize(0), [constraint])
    test_solution = test_problem.solve()
    print(f"Constraint {i}: {test_solution.status}")
```

2. **Visualize feasible region** (for small problems):
```python
# For 2D problems, plot constraints
import matplotlib.pyplot as plt
import numpy as np

x1_range = np.linspace(-5, 5, 100)
x2_range = np.linspace(-5, 5, 100)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Plot each constraint
for constraint in constraints:
    # Evaluate constraint at each point
    # Plot feasible region
```

3. **Add feasibility check**:
```python
# Solve feasibility problem
slack = cx.Variable(shape=(m,), name="slack")
feas_constraints = [
    A @ x <= b + slack,
    A_eq @ x == b_eq,
    slack >= 0
]
feas_objective = cx.Minimize(cx.sum(slack))
feas_problem = cx.Problem(feas_objective, feas_constraints)
feas_solution = feas_problem.solve()

if feas_solution.obj_value > 1e-6:
    print("Problem is infeasible")
    print(f"Minimum constraint violation: {feas_solution.obj_value}")
```

### Numerical Issues

**Error**: `LinAlgError: Matrix is singular`

**Causes and Solutions**:

1. **Rank-deficient constraint matrix**:
```python
# Check matrix rank
A_rank = jnp.linalg.matrix_rank(A)
print(f"Constraint matrix rank: {A_rank}, expected: {A.shape[0]}")

# Remove redundant constraints
A_clean, indices = remove_redundant_rows(A)
```

2. **Ill-conditioned problem**:
```python
# Check condition number
cond_num = jnp.linalg.cond(P)  # For QP objective matrix
print(f"Condition number: {cond_num}")

if cond_num > 1e12:
    # Add regularization
    P_reg = P + 1e-8 * jnp.eye(P.shape[0])
```

3. **Scale invariant formulation**:
```python
# Use relative tolerances
max_coeff = jnp.max(jnp.abs(A))
scaled_tol = 1e-8 * max_coeff
solution = problem.solve(tol=scaled_tol)
```

## Performance Issues

### Slow Compilation

**Issue**: JIT compilation takes too long

**Solutions**:

1. **Avoid dynamic shapes**:
```python
# Wrong: shape depends on runtime values
n = some_runtime_computation()
x = cx.Variable(shape=(n,))

# Correct: static shapes
N_MAX = 100  # Known at compile time
x = cx.Variable(shape=(N_MAX,))
```

2. **Simplify problem structure**:
```python
# Avoid deeply nested expressions
# Break complex expressions into parts
intermediate = A @ x
final_expr = B @ intermediate + c
```

3. **Pre-compile solve functions**:
```python
# Compile once, reuse many times
@jax.jit
def solve_fixed_problem(data):
    # Problem structure is fixed, only data changes
    problem = create_problem(data)
    return problem.solve()

# Use compiled function
solution = solve_fixed_problem(current_data)
```

### Memory Issues

**Error**: `OutOfMemoryError` or slow performance

**Solutions**:

1. **Use appropriate solver**:
```python
# For large sparse problems
solution = problem.solve(solver="osqp")

# For small dense problems  
solution = problem.solve(solver="ipm")
```

2. **Batch processing**:
```python
# Process in smaller batches
batch_size = 32
for i in range(0, len(data), batch_size):
    batch = data[i:i+batch_size]
    batch_solutions = jax.vmap(solve_single)(batch)
```

3. **Reduce precision if acceptable**:
```python
# Use 32-bit floats instead of 64-bit
jax.config.update("jax_enable_x64", False)
```

## Differentiation Issues

### Non-differentiable Solutions

**Error**: `ValueError: Cannot differentiate through solution`

**Causes and Solutions**:

1. **Active set changes**:
```python
# Problem: optimal active set changes discontinuously
# Solution: Add small regularization
eps = 1e-6
objective += eps * cx.sum_squares(x)
```

2. **Solver failure in differentiation**:
```python
# Use try-catch in differentiated function
def robust_solve(params):
    try:
        problem = create_problem(params)
        solution = problem.solve()
        return solution.obj_value
    except:
        return jnp.inf  # Large penalty for solver failure

grad_fn = jax.grad(robust_solve)
```

3. **Discontinuous solutions**:
```python
# Check if problem has unique solution
# Add strong convexity via regularization
strong_convex_obj = objective + 1e-4 * cx.sum_squares(all_variables)
```

### Gradient Accuracy Issues

**Issue**: Computed gradients don't match finite differences

**Debug Steps**:

1. **Check gradient implementation**:
```python
def check_gradients(func, x, eps=1e-6):
    """Compare analytical and finite difference gradients."""
    
    # Analytical gradient
    grad_analytical = jax.grad(func)(x)
    
    # Finite difference gradient
    grad_fd = jnp.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.at[i].add(eps)
        x_minus = x.at[i].add(-eps)
        grad_fd = grad_fd.at[i].set((func(x_plus) - func(x_minus)) / (2 * eps))
    
    # Compare
    rel_error = jnp.linalg.norm(grad_analytical - grad_fd) / jnp.linalg.norm(grad_fd)
    print(f"Relative gradient error: {rel_error}")
    
    return rel_error < 1e-4  # Acceptable tolerance

# Test your function
is_correct = check_gradients(my_optimization_function, test_params)
```

2. **Adjust tolerances**:
```python
# Tighter solver tolerance for differentiation
solution = problem.solve(solver="ipm", tol=1e-10)
```

3. **Use double precision**:
```python
# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)
```

## Debugging Tips

### Enable Verbose Output

```python
# Get detailed solver information
solution = problem.solve(verbose=True, debug=True)

# Check solution details
print(f"Status: {solution.status}")
print(f"Objective: {solution.obj_value}")
print(f"Solver info: {solution.info}")
```

### Validate Problem Structure

```python
def validate_problem(problem):
    """Check problem for common issues."""
    
    # Check variable shapes
    for var in problem.variables():
        assert len(var.shape) > 0, f"Variable {var.name} has empty shape"
        assert all(s > 0 for s in var.shape), f"Variable {var.name} has zero dimension"
    
    # Check constraint types
    for constraint in problem.constraints:
        assert constraint.is_convex(), f"Non-convex constraint: {constraint}"
    
    # Check objective convexity
    if isinstance(problem.objective, cx.Minimize):
        assert problem.objective.expr.is_convex(), "Objective is not convex"
    else:  # Maximize
        assert problem.objective.expr.is_concave(), "Objective is not concave"
    
    print("Problem validation passed!")

# Use before solving
validate_problem(my_problem)
```

### Profile Performance

```python
import time

# Time different components
start = time.time()
canonical_form = canonicalize_problem(problem)
canonicalize_time = time.time() - start

start = time.time()
solution = solve_canonical(canonical_form)
solve_time = time.time() - start

print(f"Canonicalization: {canonicalize_time:.4f}s")
print(f"Solve: {solve_time:.4f}s")
```

### Create Minimal Examples

When reporting bugs, create minimal examples:

```python
import cvxjax as cx
import jax.numpy as jnp

# Minimal problem that reproduces the issue
x = cx.Variable(shape=(2,), name="x")
objective = cx.Minimize(cx.sum_squares(x))
constraints = [x >= 0]
problem = cx.Problem(objective, constraints)

# Error occurs here
solution = problem.solve()
```

## Getting Help

1. **Check documentation**: This troubleshooting guide and API docs
2. **Search existing issues**: GitHub issues for similar problems
3. **Create minimal example**: Simplify your problem to isolate the issue
4. **Provide details**: Include error messages, versions, and code
5. **Test with latest version**: Update CVXJAX and dependencies

### Reporting Bugs

Include the following information:
- CVXJAX version
- JAX version  
- Python version
- Operating system
- Complete error message
- Minimal reproducing example
- Expected vs actual behavior
