# Core Concepts

This guide explains the key concepts and design principles behind CVXJAX.

## Design Philosophy

CVXJAX is built around three core principles:

1. **JAX-Native**: Full integration with JAX's functional programming model
2. **Static Shapes**: All shapes known at compile time for efficient execution
3. **Composable**: Easy to embed optimization in larger JAX computations

## Mathematical Foundation

### Convex Optimization
CVXJAX solves convex optimization problems of the form:

```
minimize    f(x)
subject to  g_i(x) ≤ 0  for i = 1, ..., m
            h_j(x) = 0  for j = 1, ..., p
```

Where `f`, `g_i`, and `h_j` are convex functions.

### Standard Forms

CVXJAX internally converts problems to standard forms:

**Quadratic Program (QP)**:
```
minimize    (1/2) x^T P x + q^T x + r
subject to  G x ≤ h
            A x = b
```

**Linear Program (LP)**:
```
minimize    c^T x
subject to  G x ≤ h
            A x = b
```

## Architecture Overview

```
User Problem → Canonicalization → Standard Form → Solver → Solution
     ↓               ↓                ↓            ↓         ↓
   API Layer     Build Matrices    QP/LP Data   IPM/OSQP   Results
```

### 1. API Layer (`cvxjax.api`)

The user-facing API for building optimization problems:

```python
# Variables and parameters
x = cx.Variable(shape=(n,), name="x")
A = cx.Parameter(value=data, name="A")

# Expressions
expr = A @ x + b

# Constraints  
constraints = [x >= 0, cx.sum(x) == 1]

# Problem
problem = cx.Problem(cx.Minimize(expr), constraints)
```

### 2. Expression System (`cvxjax.expressions`)

Represents mathematical expressions as computational graphs:

- **AffineExpression**: Linear combinations of variables
- **QuadraticExpression**: Quadratic forms and sums of squares
- **Atoms**: Pre-defined convex functions

```python
class AffineExpression(Expression):
    """Represents Ax + b where A and b are constants, x is a variable."""
    
    def __init__(self, coeffs: Dict[Variable, jnp.ndarray], constant: jnp.ndarray):
        self.coeffs = coeffs      # Variable coefficients
        self.constant = constant  # Constant term
```

### 3. Canonicalization (`cvxjax.canonicalize`)

Converts user problems to solver-ready standard forms:

```python
def canonicalize_problem(problem: Problem) -> Union[QPData, LPData]:
    """Convert problem to standard QP or LP form."""
    
    # 1. Extract all variables and create index mapping
    variables = extract_variables(problem)
    var_map = create_variable_mapping(variables)
    
    # 2. Build coefficient matrices
    P, q, r = build_objective_matrices(problem.objective, var_map)
    G, h = build_inequality_matrices(problem.constraints, var_map)
    A, b = build_equality_matrices(problem.constraints, var_map)
    
    # 3. Return standard form
    return QPData(P=P, q=q, r=r, G=G, h=h, A=A, b=b, var_map=var_map)
```

### 4. Solvers (`cvxjax.solvers`)

Numerical algorithms for solving standard form problems:

**Interior Point Method (IPM)**:
- Dense primal-dual algorithm
- Mehrotra predictor-corrector
- Pure JAX implementation

**OSQP Bridge**:
- Adapter to jaxopt.OSQP
- Sparse solver capability
- External dependency

### 5. Differentiation (`cvxjax.diff`)

Automatic differentiation through optimization solutions using implicit function theorem:

```python
@jax.custom_vjp
def solve_qp_differentiable(P, q, G, h, A, b):
    """Differentiable QP solve using KKT conditions."""
    
    # Forward pass: solve the QP
    solution = solve_qp(P, q, G, h, A, b)
    
    # VJP: use implicit function theorem on KKT conditions
    def vjp_fn(cotangents):
        return compute_gradients_via_kkt(solution, cotangents)
    
    return solution, vjp_fn
```

## Data Structures

### Variables and Parameters

**Variables** represent optimization unknowns:
```python
class Variable:
    def __init__(self, shape: Tuple[int, ...], name: str):
        self.shape = shape
        self.name = name
        self.id = generate_unique_id()
    
    # Operator overloading for building expressions
    def __add__(self, other): return AffineExpression(...)
    def __mul__(self, other): return AffineExpression(...)
```

**Parameters** represent fixed problem data:
```python
class Parameter:
    def __init__(self, value: jnp.ndarray, name: str):
        self.value = value
        self.name = name
        self.shape = value.shape
```

### Expressions

**Base Expression Class**:
```python
class Expression:
    """Base class for all mathematical expressions."""
    
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of this expression."""
        pass
    
    @abstractmethod
    def is_affine(self) -> bool:
        """Return True if expression is affine."""
        pass
    
    # Operator overloading
    def __add__(self, other): ...
    def __mul__(self, other): ...
```

**Affine Expressions**:
```python
class AffineExpression(Expression):
    """Linear combination: sum_i A_i * x_i + b"""
    
    def __init__(self, coeffs: Dict[Variable, jnp.ndarray], constant: jnp.ndarray):
        self.coeffs = coeffs      # {variable: coefficient_matrix}
        self.constant = constant  # Constant term
```

**Quadratic Expressions**:
```python
class QuadraticExpression(Expression):
    """Quadratic form: x^T P x + q^T x + r"""
    
    def __init__(self, P: jnp.ndarray, q: AffineExpression, r: float):
        self.P = P                # Quadratic term matrix
        self.q = q                # Linear term (affine expression)
        self.r = r                # Constant term
```

### Constraints

**Base Constraint**:
```python
class Constraint:
    """Base class for optimization constraints."""
    
    @abstractmethod
    def canonicalize(self, var_map: Dict[Variable, int]) -> ConstraintData:
        """Convert to standard form matrices."""
        pass
```

**Specific Constraint Types**:
```python
class Equality(Constraint):
    """Equality constraint: expr == 0"""
    
    def __init__(self, expression: Expression):
        self.expression = expression

class Inequality(Constraint):
    """Inequality constraint: expr <= 0"""
    
    def __init__(self, expression: Expression):
        self.expression = expression

class Box(Constraint):
    """Box constraint: lower <= variable <= upper"""
    
    def __init__(self, variable: Variable, lower: float, upper: float):
        self.variable = variable
        self.lower = lower
        self.upper = upper
```

## JAX Integration

### Pytree Registration

All CVXJAX objects are JAX pytrees for seamless integration:

```python
# Register Variable as pytree
def variable_flatten(var):
    return (), (var.shape, var.name, var.id)

def variable_unflatten(aux, children):
    shape, name, var_id = aux
    var = Variable(shape, name)
    var.id = var_id
    return var

jax.tree_util.register_pytree_node(Variable, variable_flatten, variable_unflatten)
```

### JIT Compilation

Problems can be JIT compiled for efficient repeated solving:

```python
def solve_jit(self, **kwargs):
    """JIT-compiled solve method."""
    
    @jax.jit
    def _solve_compiled():
        return self.solve(**kwargs)
    
    return _solve_compiled()
```

### Automatic Differentiation

Solutions are differentiable w.r.t. problem parameters:

```python
def loss_function(data):
    # Build optimization problem with data
    problem = create_problem(data)
    solution = problem.solve_jit()
    return solution.obj_value

# Compute gradients
grad_fn = jax.grad(loss_function)
gradients = grad_fn(data)
```

### Vectorization

Use vmap for batch processing:

```python
# Single problem solver
def solve_single(param):
    problem = create_problem(param)
    return problem.solve_jit()

# Batch solver
batch_solve = jax.vmap(solve_single)
solutions = batch_solve(param_batch)
```

## Shape System

CVXJAX uses static shapes throughout for efficient compilation:

### Shape Propagation
```python
# Variable shapes
x = Variable(shape=(n,))           # Shape: (n,)
y = Variable(shape=(m, k))         # Shape: (m, k)

# Expression shapes  
A @ x                              # Shape: depends on A.shape[0]
x.T @ P @ x                        # Shape: () (scalar)
cx.sum(x)                          # Shape: () (scalar)
```

### Shape Checking
```python
def check_shapes_compatible(expr1, expr2, op_name):
    """Ensure expressions have compatible shapes for operation."""
    shape1, shape2 = expr1.shape, expr2.shape
    
    if op_name in ["add", "sub"]:
        if shape1 != shape2:
            raise ShapeError(f"Cannot {op_name} shapes {shape1} and {shape2}")
    
    elif op_name == "matmul":
        if shape1[-1] != shape2[-2]:
            raise ShapeError(f"Cannot multiply shapes {shape1} and {shape2}")
```

## Extensibility

### Custom Atoms

Add new convex functions:

```python
def log_sum_exp(x):
    """Log-sum-exp function (convex)."""
    # Implementation using JAX operations
    return jax.scipy.special.logsumexp(x)

# Register as atomic function
cx.register_atom("log_sum_exp", log_sum_exp, convex=True)
```

### Custom Solvers

Implement new solution algorithms:

```python
class CustomSolver:
    def solve(self, qp_data: QPData) -> Solution:
        """Solve QP using custom algorithm."""
        # Your algorithm here
        x_opt = custom_qp_solver(qp_data.P, qp_data.q, ...)
        
        return Solution(
            status="optimal",
            obj_value=compute_objective(x_opt, qp_data),
            primal={"x": x_opt},
            dual=None
        )

# Register solver
cx.register_solver("custom", CustomSolver())
```

## Performance Considerations

### Memory Layout
- Use row-major (C-style) arrays
- Avoid unnecessary transposes
- Prefer in-place operations where possible

### Compilation
- JIT compile problem solve methods
- Avoid Python loops in hot paths
- Use static shapes throughout

### Numerical Stability
- Scale problem data appropriately
- Add regularization for ill-conditioned problems
- Use appropriate solver tolerances

### Batching
- Use vmap for similar problems
- Batch matrix operations when possible
- Consider memory constraints for large batches

This conceptual foundation enables efficient, differentiable convex optimization within the JAX ecosystem.
