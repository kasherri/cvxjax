# CVXJAX API Reference

**Version:** 0.1.0

CVXJAX is a JAX-native convex optimization library designed for high-performance, differentiable optimization. It provides an intuitive API similar to CVXPY but with full JAX integration for automatic differentiation, JIT compilation, and vectorization.

## Table of Contents

1. [Core Classes](#core-classes)
   - [Variable](#variable)
   - [Parameter](#parameter) 
   - [Constant](#constant)
   - [Problem](#problem)
   - [Solution](#solution)
2. [Objective Functions](#objective-functions)
   - [Minimize](#minimize)
   - [Maximize](#maximize)
3. [Atomic Functions](#atomic-functions)
   - [sum_squares](#sum_squares)
   - [quad_form](#quad_form)
   - [sum](#sum)
   - [square](#square)
   - [abs](#abs)
4. [Constraints](#constraints)
   - [EqualityConstraint](#equalityconstraint)
   - [InequalityConstraint](#inequalityconstraint)
   - [BoxConstraint](#boxconstraint)
5. [Solver Interface](#solver-interface)
6. [Examples](#examples)

---

## Core Classes

### Variable

```python
class Variable:
    def __init__(self, shape: tuple[int, ...], name: Optional[str] = None)
```

**Description:** Optimization variable representing the primary decision variables in optimization problems. Variables support operator overloading for building affine expressions and can be indexed, sliced, and combined mathematically.

**Parameters:**
- `shape` (tuple[int, ...]): Shape of the variable array. Must be a tuple of positive integers.
- `name` (Optional[str]): Optional name for debugging and display purposes.

**Properties:**
- `size` (int): Total number of elements in the variable (computed as `np.prod(shape)`)
- `shape` (tuple): Shape of the variable
- `name` (Optional[str]): Variable name

**Methods:**
- `is_affine() -> bool`: Always returns `True` (variables are affine)
- `is_convex() -> bool`: Always returns `True` (variables are convex)

**Supported Operations:**
- **Arithmetic:** `+`, `-`, `*`, `@` (matrix multiplication)
- **Indexing:** `x[i]`, `x[i:j]` (creates new variables representing indexed elements)
- **Comparisons:** `==`, `<=`, `>=` (creates constraints)

**Examples:**
```python
import cvxjax as cx
import jax.numpy as jnp

# Scalar variable
x = cx.Variable(shape=(), name="x")

# Vector variable  
w = cx.Variable(shape=(10,), name="weights")

# Matrix variable
X = cx.Variable(shape=(3, 3), name="matrix")

# Variable operations
y = 2 * x + 1                    # Affine expression
objective = cx.sum_squares(w)    # Quadratic expression
constraint = jnp.ones(10) @ w == 1  # Equality constraint
```

---

### Parameter

```python
class Parameter:
    def __init__(self, shape: tuple[int, ...], name: Optional[str] = None)
```

**Description:** Problem parameter that can be updated without rebuilding the optimization problem. Parameters allow for efficient re-solving of problems with different data.

**Parameters:**
- `shape` (tuple[int, ...]): Shape of the parameter array
- `name` (Optional[str]): Optional name for debugging

**Usage:**
Parameters are designed for cases where you solve the same problem structure with different data values.

**Example:**
```python
# Portfolio optimization with changeable expected returns
mu = cx.Parameter(shape=(n_assets,), name="expected_returns") 
w = cx.Variable(shape=(n_assets,), name="weights")

# Objective depends on parameter
objective = cx.Minimize(-mu @ w + 0.5 * cx.quad_form(w, Sigma))
```

---

### Constant

```python
class Constant:
    def __init__(self, value: jnp.ndarray, name: Optional[str] = None)
```

**Description:** Constant value in optimization expressions. Useful for explicitly representing fixed data in problems.

**Parameters:**
- `value` (jnp.ndarray): The constant value
- `name` (Optional[str]): Optional name for debugging

**Example:**
```python
# Risk aversion parameter
gamma = cx.Constant(1.0, name="risk_aversion")
```

---

### Problem

```python
class Problem:
    def __init__(self, objective: Union[Minimize, Maximize], constraints: list[Constraint] = None)
```

**Description:** Optimization problem combining an objective function with constraints. The main interface for solving optimization problems.

**Parameters:**
- `objective` (Union[Minimize, Maximize]): Objective function to optimize
- `constraints` (list[Constraint], optional): List of constraints. Defaults to empty list.

**Methods:**

#### solve()
```python
def solve(
    self,
    solver: Literal["ipm", "osqp", "boxosqp"] = "ipm",
    tol: float = 1e-8,
    max_iter: int = 1000,
    verbose: bool = False,
    **solver_kwargs
) -> Solution
```

Solve the optimization problem.

**Parameters:**
- `solver` (str): Solver to use. Options:
  - `"ipm"`: Dense primal-dual interior point method (default)
  - `"osqp"`: Operator Splitting QP solver
  - `"boxosqp"`: Box-constrained OSQP solver
- `tol` (float): Convergence tolerance (default: 1e-8)
- `max_iter` (int): Maximum iterations (default: 1000)
- `verbose` (bool): Print solver output (default: False)
- `**solver_kwargs`: Additional solver-specific parameters

**Returns:** `Solution` object with optimization results

#### solve_jit()
```python
def solve_jit(
    self,
    solver: Literal["ipm", "osqp"] = "ipm", 
    **kwargs
) -> Solution
```

JIT-compiled version of solve for maximum performance.

**Note:** Currently has limitations with dynamic constraint structures.

**Example:**
```python
# Create problem
x = cx.Variable(shape=(2,), name="x")
objective = cx.Minimize(cx.sum_squares(x - jnp.array([1, 2])))
constraints = [x >= 0, x <= 1]

problem = cx.Problem(objective, constraints)

# Solve
solution = problem.solve(solver="ipm", tol=1e-6)
print(f"Status: {solution.status}")
print(f"Optimal value: {solution.obj_value}")
print(f"Solution: {solution.primal[x]}")
```

---

### Solution

```python
@dataclass(frozen=True)
class Solution:
    status: str
    obj_value: float
    primal: dict[Variable, jnp.ndarray]
    dual: dict[str, jnp.ndarray]  
    info: dict[str, Any]
```

**Description:** Results from solving an optimization problem.

**Attributes:**
- `status` (str): Solution status. Common values:
  - `"optimal"`: Problem solved to optimality
  - `"max_iter"`: Maximum iterations reached
  - `"primal_infeasible"`: Problem is infeasible
  - `"dual_infeasible"`: Problem is unbounded
  - `"error"`: Solver error occurred
- `obj_value` (float): Optimal objective value
- `primal` (dict): Mapping from Variable objects to their optimal values
- `dual` (dict): Dual variable values (constraint multipliers)
- `info` (dict): Additional solver information (iterations, residuals, etc.)

**Example:**
```python
solution = problem.solve()

if solution.status == "optimal":
    # Access primal variables
    x_opt = solution.primal[x]  # Use Variable object as key
    
    # Check solver info
    print(f"Iterations: {solution.info['iterations']}")
    print(f"Dual residual: {solution.info['dual_residual']}")
```

---

## Objective Functions

### Minimize

```python
class Minimize:
    def __init__(self, expression: Expression)
```

**Description:** Minimization objective for optimization problems.

**Parameters:**
- `expression` (Expression): Expression to minimize. Must be convex for global optimality.

**Example:**
```python
x = cx.Variable(shape=(3,))
objective = cx.Minimize(cx.sum_squares(x))  # Minimize ||x||^2
```

### Maximize  

```python
class Maximize:
    def __init__(self, expression: Expression)
```

**Description:** Maximization objective for optimization problems.

**Parameters:**
- `expression` (Expression): Expression to maximize. Must be concave for global optimality.

**Example:**
```python
w = cx.Variable(shape=(n_assets,))
mu = jnp.array([...])  # Expected returns
objective = cx.Maximize(mu @ w)  # Maximize expected return
```

---

## Atomic Functions

### sum_squares

```python
def sum_squares(expr: Expression) -> QuadraticExpression
```

**Description:** Sum of squares function: `||expr||_2^2`. Creates a convex quadratic expression.

**Parameters:**
- `expr` (Expression): Expression to compute sum of squares for

**Returns:** QuadraticExpression representing `||expr||^2`

**Examples:**
```python
x = cx.Variable(shape=(3,))

# Minimize ||x||^2
obj1 = cx.Minimize(cx.sum_squares(x))

# Minimize ||Ax - b||^2 (least squares)
A = jnp.random.normal(key, (5, 3))
b = jnp.random.normal(key, (5,))
obj2 = cx.Minimize(cx.sum_squares(A @ x - b))

# Portfolio risk: ||sqrt(Sigma) @ w||^2
Sigma_sqrt = jnp.linalg.cholesky(Sigma)
risk = cx.sum_squares(Sigma_sqrt @ w)
```

### quad_form

```python
def quad_form(expr: Expression, matrix: jnp.ndarray) -> QuadraticExpression
```

**Description:** Quadratic form: `expr^T @ matrix @ expr`. The matrix should be positive semidefinite for convexity.

**Parameters:**
- `expr` (Expression): Vector expression
- `matrix` (jnp.ndarray): Quadratic form matrix (should be PSD)

**Returns:** QuadraticExpression representing the quadratic form

**Examples:**
```python
w = cx.Variable(shape=(n_assets,))
Sigma = jnp.array([...])  # Covariance matrix

# Portfolio variance
risk = cx.quad_form(w, Sigma)

# With risk aversion parameter
gamma = 1.0
objective = cx.Minimize(-mu @ w + gamma * cx.quad_form(w, Sigma))
```

### sum

```python
def sum(expr: Expression) -> Expression
```

**Description:** Sum all elements of an expression.

**Parameters:**
- `expr` (Expression): Expression to sum

**Returns:** Scalar expression representing the sum

**Example:**
```python
x = cx.Variable(shape=(10,))

# Budget constraint: sum(x) == 1
constraint = cx.sum(x) == 1

# L1 regularization: sum(abs(x))
l1_reg = cx.sum(cx.abs(x))
```

### square

```python
def square(expr: Expression) -> QuadraticExpression
```

**Description:** Element-wise square function.

**Parameters:**
- `expr` (Expression): Expression to square

**Returns:** QuadraticExpression with element-wise squares

### abs

```python
def abs(expr: Expression) -> Expression
```

**Description:** Element-wise absolute value function.

**Parameters:**
- `expr` (Expression): Expression to take absolute value of

**Returns:** Expression representing element-wise absolute values

**Note:** This creates auxiliary variables and constraints for the absolute value formulation.

---

## Constraints

### Equality Constraints

Create equality constraints using the `==` operator:

```python
# Budget constraint
jnp.ones(n) @ x == 1

# Matrix constraint  
A @ x == b

# Variable equality
x == y
```

### Inequality Constraints

Create inequality constraints using `<=` and `>=` operators:

```python
# Non-negativity
x >= 0

# Box constraints
x >= lb
x <= ub

# General linear inequality
A @ x <= b
```

### EqualityConstraint

```python
@dataclass(frozen=True)
class EqualityConstraint(Constraint):
    expression: Expression
```

**Description:** Represents equality constraint `expression == 0`.

**Requirements:** Expression must be affine.

### InequalityConstraint

```python
@dataclass(frozen=True)  
class InequalityConstraint(Constraint):
    expression: Expression
    sense: Literal["<=", ">="]
```

**Description:** Represents inequality constraint `expression <= 0` or `expression >= 0`.

**Requirements:** Expression must be convex (for `<=`) or concave (for `>=`).

### BoxConstraint

```python
@dataclass(frozen=True)
class BoxConstraint(Constraint):
    variable: Variable
    lb: jnp.ndarray
    ub: jnp.ndarray
```

**Description:** Box constraints `lb <= variable <= ub`.

---

## Solver Interface

CVXJAX supports multiple solvers optimized for different problem types:

### Interior Point Method (IPM)

**Solver ID:** `"ipm"`

**Best for:** General quadratic programs, high accuracy requirements

**Features:**
- Dense primal-dual interior point method with Mehrotra predictor-corrector
- Handles equality, inequality, and box constraints via slack form conversion
- Excellent convergence for well-conditioned problems
- Supports automatic differentiation

**Parameters:**
- `tol` (float): Convergence tolerance (default: 1e-8)
- `max_iter` (int): Maximum iterations (default: 50)
- `regularization` (float): Regularization parameter (default: 1e-12)
- `fraction_to_boundary` (float): Step size control (default: 0.995)

### OSQP Solver

**Solver ID:** `"osqp"`

**Best for:** Large sparse problems, embedded applications

**Features:**
- Operator Splitting QP solver
- Good for sparse constraint matrices
- Robust to ill-conditioning

### BoxOSQP Solver

**Solver ID:** `"boxosqp"`

**Best for:** Box-constrained quadratic programs

**Features:**
- Specialized JAXOpt-based solver for box constraints
- No general equality/inequality constraints
- Efficient for problems with only bound constraints

**Example:**
```python
# Compare solvers
solvers = ["ipm", "osqp"]
for solver in solvers:
    solution = problem.solve(solver=solver, tol=1e-6)
    print(f"{solver}: {solution.status}, obj={solution.obj_value:.6f}")
```

---

## Examples

### 1. Basic Quadratic Program

```python
import cvxjax as cx
import jax.numpy as jnp

# Minimize ||x - c||^2 subject to Ax = b, x >= 0
n, m = 10, 5
A = jnp.random.normal(jax.random.PRNGKey(0), (m, n))
b = jnp.random.normal(jax.random.PRNGKey(1), (m,))
c = jnp.random.normal(jax.random.PRNGKey(2), (n,))

x = cx.Variable(shape=(n,), name="x")
objective = cx.Minimize(cx.sum_squares(x - c))
constraints = [A @ x == b, x >= 0]

problem = cx.Problem(objective, constraints)
solution = problem.solve(solver="ipm")

print(f"Status: {solution.status}")
print(f"Optimal value: {solution.obj_value:.6f}")
```

### 2. Portfolio Optimization

```python
import cvxjax as cx
import jax.numpy as jnp

# Generate random data
key = jax.random.PRNGKey(42)
n_assets = 8
returns = 0.01 + 0.02 * jax.random.normal(key, (n_assets,))
Sigma = jax.random.normal(key, (n_assets, n_assets))
Sigma = Sigma @ Sigma.T / n_assets  # Make PSD

# Mean-variance optimization
w = cx.Variable(shape=(n_assets,), name="weights")
gamma = 1.0  # Risk aversion

objective = cx.Minimize(-returns @ w + gamma * cx.quad_form(w, Sigma))
constraints = [
    cx.sum(w) == 1,  # Budget constraint
    w >= 0           # Long-only
]

problem = cx.Problem(objective, constraints)
solution = problem.solve(solver="ipm", tol=1e-8)

if solution.status == "optimal":
    weights = solution.primal[w]
    expected_return = returns @ weights
    portfolio_vol = jnp.sqrt(weights @ Sigma @ weights)
    sharpe_ratio = expected_return / portfolio_vol
    
    print(f"Expected return: {expected_return:.4f}")
    print(f"Volatility: {portfolio_vol:.4f}")
    print(f"Sharpe ratio: {sharpe_ratio:.4f}")
```

### 3. Box-Constrained Optimization

```python
# Minimize (x-1)^2 + (y-2)^2 subject to 0 <= x <= 1, 1 <= y <= 3
x = cx.Variable(shape=(1,), name="x")
y = cx.Variable(shape=(1,), name="y")

objective = cx.Minimize(cx.sum_squares(x - 1.0) + cx.sum_squares(y - 2.0))
constraints = [
    x >= 0, x <= 1,  # Box constraints for x
    y >= 1, y <= 3   # Box constraints for y
]

problem = cx.Problem(objective, constraints)
solution = problem.solve(solver="ipm")

print(f"Optimal x: {solution.primal[x]}")  # Should be [1.0]
print(f"Optimal y: {solution.primal[y]}")  # Should be [2.0] 
print(f"Optimal value: {solution.obj_value}")  # Should be 0.0
```

### 4. Differentiable Optimization

```python
import jax

# Parameterized QP that can be differentiated
def solve_qp(c):
    x = cx.Variable(shape=(2,), name="x")
    objective = cx.Minimize(c @ x + cx.sum_squares(x))
    constraints = [x >= 0]
    
    problem = cx.Problem(objective, constraints)
    solution = problem.solve(solver="ipm")
    return solution.primal[x]

# Compute gradient of solution with respect to parameters
grad_fn = jax.grad(lambda c: jnp.sum(solve_qp(c)))
c = jnp.array([1.0, -0.5])
gradient = grad_fn(c)
print(f"Gradient: {gradient}")
```

---

## Best Practices

1. **Solver Selection:**
   - Use `"ipm"` for most problems requiring high accuracy
   - Use `"osqp"` for large sparse problems
   - Use `"boxosqp"` for pure box-constrained problems

2. **Variable Naming:**
   - Always provide meaningful names for debugging: `Variable(shape=(10,), name="weights")`

3. **Constraint Formulation:**
   - Prefer `w == current_weights + t_plus - t_minus` over `w - current_weights == t_plus - t_minus`
   - Group similar constraints together

4. **Performance:**
   - Use `solve_jit()` for repeated solves with same structure
   - Consider constraint elimination for better conditioning

5. **Numerical Stability:**
   - Scale your problem data to reasonable ranges
   - Use appropriate tolerances for your application
   - Check solution status before using results

---

## Error Handling

Common errors and solutions:

- **"Unsupported constraint type"**: Check constraint formulation, avoid boolean constraints
- **"max_iter"**: Increase `max_iter` or check problem conditioning
- **"JIT compilation error"**: Use `solve()` instead of `solve_jit()` for dynamic problems
- **Variable access errors**: Use Variable objects as keys: `solution.primal[variable]`

For more examples and tutorials, see the `examples/` directory and `docs/` folder.
