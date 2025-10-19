# Getting Started with CVXJAX

Welcome to CVXJAX! This guide will walk you through the basics of solving optimization problems with CVXJAX, a JAX-native convex optimization library.

## What is CVXJAX?

CVXJAX brings the expressiveness of CVXPY to the JAX ecosystem, enabling:
- **High-performance optimization** with JIT compilation and automatic differentiation
- **GPU acceleration** through JAX's device-agnostic design
- **Differentiable optimization** for machine learning applications
- **Familiar API** similar to CVXPY for easy adoption

## Installation

```bash
pip install cvxjax
```

**Requirements:**
- Python ≥ 3.8
- JAX ≥ 0.4.0
- NumPy ≥ 1.20.0

## Your First Optimization Problem

Let's solve a simple quadratic program step by step:

**Problem:** Minimize `||x - [1, 2]||²` subject to `x ≥ 0`

```python
import cvxjax as cx
import jax.numpy as jnp

# Step 1: Create optimization variables
x = cx.Variable(shape=(2,), name="x")

# Step 2: Define the objective function
target = jnp.array([1.0, 2.0])
objective = cx.Minimize(cx.sum_squares(x - target))

# Step 3: Add constraints
constraints = [x >= 0]  # Non-negativity constraint

# Step 4: Create the optimization problem
problem = cx.Problem(objective, constraints)

# Step 5: Solve the problem
solution = problem.solve(solver="ipm", tol=1e-8)

# Step 6: Extract and display results
if solution.status == "optimal":
    x_optimal = solution.primal[x]
    print(f"Optimal solution: {x_optimal}")
    print(f"Optimal value: {solution.obj_value:.6f}")
    print(f"Solver converged in {solution.info['iterations']} iterations")
else:
    print(f"Solver failed with status: {solution.status}")
```

**Expected Output:**
```
Optimal solution: [1. 2.]
Optimal value: 0.000000
Solver converged in 1 iterations
```

## Core Concepts

### 1. Variables

Variables represent the unknowns in your optimization problem:

```python
# Scalar variable
x = cx.Variable(shape=(), name="x")

# Vector variable (portfolio weights)
w = cx.Variable(shape=(10,), name="weights")

# Matrix variable
X = cx.Variable(shape=(3, 3), name="matrix")
```

**Key points:**
- Always specify the `shape` 
- Use descriptive `name`s for debugging
- Variables support arithmetic operations: `+`, `-`, `*`, `@`

### 2. Expressions

Build mathematical expressions using variables and JAX arrays:

```python
x = cx.Variable(shape=(3,), name="x")
A = jnp.array([[1, 2, 3], [4, 5, 6]])
b = jnp.array([1, 2])

# Linear expression
linear_expr = A @ x + b

# Quadratic expression  
quadratic_expr = cx.sum_squares(x)

# Combined expression
combined = linear_expr + 0.5 * quadratic_expr
```

### 3. Objectives

Wrap expressions in `Minimize` or `Maximize`:

```python
# Minimization (most common)
objective = cx.Minimize(cx.sum_squares(x))

# Maximization (equivalent to minimizing the negative)
objective = cx.Maximize(returns @ weights)
```

### 4. Constraints

Create constraints using comparison operators:

```python
# Equality constraints
cx.sum(weights) == 1      # Budget constraint
A @ x == b               # Linear equality

# Inequality constraints  
x >= 0                   # Non-negativity
x <= upper_bounds        # Upper bounds
A @ x <= b              # Linear inequality
```

### 5. Problem and Solution

Combine everything into a problem and solve:

```python
problem = cx.Problem(objective, constraints)
solution = problem.solve()

# Always check the status
if solution.status == "optimal":
    # Extract variable values using the Variable object as key
    x_value = solution.primal[x]
    # Access solver information
    num_iters = solution.info['iterations']
```

## Example 1: Portfolio Optimization

Let's solve a classic mean-variance portfolio optimization problem:

```python
import cvxjax as cx
import jax.numpy as jnp
import jax

# Generate sample data
key = jax.random.PRNGKey(42)
n_assets = 5

# Expected returns (annualized)
returns = jnp.array([0.10, 0.12, 0.08, 0.15, 0.09])

# Covariance matrix (generate a random PSD matrix)
L = jax.random.normal(key, (n_assets, n_assets))
Sigma = (L @ L.T) / n_assets

print("Expected Returns:", returns)
print("Volatilities:", jnp.sqrt(jnp.diag(Sigma)))

# Define portfolio weights variable
w = cx.Variable(shape=(n_assets,), name="weights")

# Risk aversion parameter
gamma = 1.0

# Objective: maximize return - risk penalty
# Equivalent to: minimize -return + risk penalty
expected_return = returns @ w
portfolio_risk = cx.quad_form(w, Sigma)
objective = cx.Minimize(-expected_return + gamma * portfolio_risk)

# Constraints
constraints = [
    cx.sum(w) == 1,  # Budget constraint: weights sum to 1
    w >= 0           # Long-only: no short selling
]

# Create and solve problem
problem = cx.Problem(objective, constraints)
solution = problem.solve(solver="ipm", tol=1e-8)

# Display results
if solution.status == "optimal":
    weights = solution.primal[w]
    
    # Compute portfolio statistics
    port_return = returns @ weights
    port_variance = weights @ Sigma @ weights
    port_volatility = jnp.sqrt(port_variance)
    sharpe_ratio = port_return / port_volatility
    
    print("\n=== Portfolio Optimization Results ===")
    print(f"Status: {solution.status}")
    print(f"Expected Return: {port_return:.4f}")
    print(f"Volatility: {port_volatility:.4f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Weights: {weights}")
    print(f"Solver iterations: {solution.info['iterations']}")
else:
    print(f"Optimization failed: {solution.status}")
```

## Example 2: Least Squares Regression

Solve the classic least squares problem with L2 regularization:

```python
import cvxjax as cx
import jax.numpy as jnp
import jax

# Generate synthetic regression data
key = jax.random.PRNGKey(123)
n_samples, n_features = 100, 10

X = jax.random.normal(key, (n_samples, n_features))
true_beta = jax.random.normal(key, (n_features,))
noise = 0.1 * jax.random.normal(key, (n_samples,))
y = X @ true_beta + noise

print(f"Data: {n_samples} samples, {n_features} features")
print(f"True coefficients: {true_beta[:3]}...")  # Show first 3

# Define optimization variable
beta = cx.Variable(shape=(n_features,), name="coefficients")

# Regularization parameter
lambda_reg = 0.1

# Objective: minimize ||X*beta - y||^2 + lambda * ||beta||^2
data_fit = cx.sum_squares(X @ beta - y)
regularization = lambda_reg * cx.sum_squares(beta)
objective = cx.Minimize(data_fit + regularization)

# No additional constraints needed for ridge regression
problem = cx.Problem(objective)
solution = problem.solve(solver="ipm")

# Compare with JAX's built-in least squares
if solution.status == "optimal":
    beta_cvxjax = solution.primal[beta]
    
    # JAX implementation for comparison
    XtX_reg = X.T @ X + lambda_reg * jnp.eye(n_features)
    Xty = X.T @ y
    beta_direct = jnp.linalg.solve(XtX_reg, Xty)
    
    print("\n=== Ridge Regression Results ===")
    print(f"CVXJAX solution: {beta_cvxjax[:3]}...")
    print(f"Direct solution: {beta_direct[:3]}...")
    print(f"True coefficients: {true_beta[:3]}...")
    print(f"CVXJAX vs Direct difference: {jnp.linalg.norm(beta_cvxjax - beta_direct):.2e}")
    print(f"Reconstruction error: {jnp.linalg.norm(X @ beta_cvxjax - y):.4f}")
```

## Example 3: Box-Constrained Optimization

Solve a problem with both lower and upper bounds:

```python
import cvxjax as cx
import jax.numpy as jnp

# Problem: minimize ||x - c||^2 subject to lb <= x <= ub
n = 5
c = jnp.array([1.0, -0.5, 2.0, 0.0, -1.0])  # Target point
lb = jnp.array([0.0, -1.0, 0.5, -0.5, -2.0])  # Lower bounds  
ub = jnp.array([0.8, 0.0, 1.5, 0.5, 0.0])   # Upper bounds

print("Target point c:", c)
print("Lower bounds:", lb)
print("Upper bounds:", ub)

# Define variable
x = cx.Variable(shape=(n,), name="x")

# Objective: minimize squared distance to target
objective = cx.Minimize(cx.sum_squares(x - c))

# Box constraints
constraints = [
    x >= lb,  # Lower bounds
    x <= ub   # Upper bounds
]

# Solve
problem = cx.Problem(objective, constraints)
solution = problem.solve(solver="ipm")

if solution.status == "optimal":
    x_opt = solution.primal[x]
    
    print("\n=== Box-Constrained Optimization Results ===")
    print(f"Optimal solution: {x_opt}")
    print(f"Distance to target: {jnp.linalg.norm(x_opt - c):.4f}")
    
    # Check which constraints are active
    lb_active = jnp.abs(x_opt - lb) < 1e-6
    ub_active = jnp.abs(x_opt - ub) < 1e-6
    
    print(f"Lower bounds active: {jnp.where(lb_active)[0]}")
    print(f"Upper bounds active: {jnp.where(ub_active)[0]}")
    print(f"Free variables: {jnp.where(~(lb_active | ub_active))[0]}")
```

## Choosing the Right Solver

CVXJAX provides multiple solvers optimized for different problem types:

### Interior Point Method (IPM) - Default
- **Best for:** General quadratic programs, high accuracy requirements
- **Advantages:** Fast convergence, high accuracy, handles ill-conditioning well
- **Use when:** You need the most accurate solution

```python
solution = problem.solve(solver="ipm", tol=1e-8)
```

### OSQP Solver
- **Best for:** Large sparse problems, embedded applications
- **Advantages:** Good for sparse constraint matrices, robust
- **Use when:** Your problem has many constraints but sparse structure

```python
solution = problem.solve(solver="osqp", tol=1e-6)
```

### BoxOSQP Solver
- **Best for:** Problems with only box constraints (no general constraints)
- **Advantages:** Specialized for bound-constrained problems
- **Limitations:** Cannot handle equality or general inequality constraints

```python
# Only for problems of the form: minimize f(x) subject to lb <= x <= ub
solution = problem.solve(solver="boxosqp", tol=1e-6)
```

## Best Practices

### 1. Problem Scaling
Always scale your problem data to reasonable ranges:

```python
# Good: scaled data
returns = jnp.array([0.10, 0.12, 0.08])  # Reasonable percentages

# Avoid: extreme scales  
bad_returns = jnp.array([1e6, 1.2e6, 0.8e6])  # Too large
```

### 2. Variable Naming
Use descriptive names for debugging:

```python
# Good
weights = cx.Variable(shape=(n_assets,), name="portfolio_weights")
returns = cx.Parameter(shape=(n_assets,), name="expected_returns")

# Avoid
x = cx.Variable(shape=(n_assets,))  # No name
```

### 3. Constraint Formulation
Some constraint formulations work better than others:

```python
# Preferred: direct equality
w == current_weights + trades_buy - trades_sell

# Avoid: can create boolean constraints
w - current_weights == trades_buy - trades_sell
```

### 4. Solution Checking
Always verify the solution status:

```python
solution = problem.solve()

if solution.status == "optimal":
    # Use the solution
    x_opt = solution.primal[x]
elif solution.status == "max_iter":
    print("Solver hit iteration limit - try increasing max_iter")
elif solution.status in ["primal_infeasible", "dual_infeasible"]:
    print("Problem may be infeasible or unbounded")
else:
    print(f"Solver failed: {solution.status}")
```

## JAX Integration

One of CVXJAX's key advantages is seamless JAX integration:

### Automatic Differentiation

```python
def portfolio_objective_value(expected_returns):
    """Solve portfolio optimization and return objective value."""
    w = cx.Variable(shape=len(expected_returns), name="weights")
    objective = cx.Minimize(-expected_returns @ w + cx.quad_form(w, Sigma))
    constraints = [cx.sum(w) == 1, w >= 0]
    
    problem = cx.Problem(objective, constraints)
    solution = problem.solve(solver="ipm")
    return solution.obj_value

# Compute gradient of objective w.r.t. expected returns
grad_fn = jax.grad(portfolio_objective_value)
gradient = grad_fn(returns)
print(f"Sensitivity to returns: {gradient}")
```

### JIT Compilation

```python
# JIT compile for repeated solves
@jax.jit
def solve_parameterized_qp(c):
    x = cx.Variable(shape=len(c), name="x")
    objective = cx.Minimize(c @ x + cx.sum_squares(x))
    constraints = [x >= 0]
    
    problem = cx.Problem(objective, constraints)
    solution = problem.solve_jit(solver="ipm")  # Note: solve_jit
    return solution.primal[x]

# Fast repeated solves
for i in range(100):
    c = jax.random.normal(jax.random.PRNGKey(i), (10,))
    x_opt = solve_parameterized_qp(c)
```

## Next Steps

Now that you understand the basics:

1. **Explore the examples**: Check out `examples/` for real-world applications
2. **Read the API reference**: See `docs/api_reference.md` for complete function documentation  
3. **Try advanced features**: Experiment with parameters, custom solvers, and differentiation
4. **Join the community**: Report issues and contribute on GitHub

## Common Issues & Solutions

**Problem**: `KeyError` when accessing solution
```python
# ❌ Wrong: using string key
x_opt = solution.primal["x"]

# ✅ Correct: using Variable object  
x_opt = solution.primal[x]
```

**Problem**: Solver returns "max_iter"
```python
# Try increasing iterations or relaxing tolerance
solution = problem.solve(solver="ipm", max_iter=200, tol=1e-6)
```

**Problem**: "Unsupported constraint type: bool"
```python
# ❌ Wrong: can evaluate to boolean
constraint = (x - y == z - w)

# ✅ Correct: rearrange
constraint = (x == y + z - w)
```

**Problem**: JIT compilation fails
```python
# Use regular solve() for dynamic problems
solution = problem.solve(solver="ipm")  # Instead of solve_jit()
```

Happy optimizing with CVXJAX! 
