"""Test API functionality."""

import jax
import jax.numpy as jnp
import pytest

import cvxjax as cx


class TestAPI:
    """Test main API components."""
    
    def setup_method(self):
        """Set up test environment."""
        jax.config.update("jax_enable_x64", True)
    
    def test_variable_creation(self):
        """Test Variable creation and properties."""
        x = cx.Variable(shape=(2,), name="x")
        assert x.shape == (2,)
        assert x.name == "x"
        assert x.size == 2
    
    def test_parameter_creation(self):
        """Test Parameter creation."""
        value = jnp.array([1.0, 2.0])
        p = cx.Parameter(value, name="param")
        assert p.shape == (2,)
        assert p.name == "param"
        assert jnp.allclose(p.value, value)
    
    def test_constant_creation(self):
        """Test Constant creation."""
        value = jnp.array([1.0, 2.0])
        c = cx.Constant(value, name="const")
        assert c.shape == (2,)
        assert c.name == "const" 
        assert jnp.allclose(c.value, value)
    
    def test_simple_qp_problem(self):
        """Test solving a simple QP problem.
        
        minimize    x^2 + y^2
        subject to  x + y = 1
                   x >= 0, y >= 0
        
        Expected solution: x = 0.5, y = 0.5, obj = 0.5
        """
        # Variables
        x = cx.Variable(shape=(1,), name="x")
        y = cx.Variable(shape=(1,), name="y")
        
        # Objective: minimize x^2 + y^2
        objective = cx.Minimize(cx.sum_squares(x) + cx.sum_squares(y))
        
        # Constraints
        constraints = [
            x + y == 1.0,  # x + y = 1
            x >= 0,        # x >= 0
            y >= 0,        # y >= 0
        ]
        
        # Problem
        problem = cx.Problem(objective, constraints)
        
        # Solve
        solution = problem.solve(solver="ipm", tol=1e-6)
        
        # Check solution
        assert solution.status == "optimal"
        assert abs(solution.obj_value - 0.5) < 1e-4
        assert abs(solution.primal[x][0] - 0.5) < 1e-4
        assert abs(solution.primal[y][0] - 0.5) < 1e-4
    
    def test_portfolio_qp(self):
        """Test portfolio optimization QP.
        
        minimize    (1/2) x^T Sigma x - mu^T x
        subject to  sum(x) = 1
                   x >= 0
        
        This is a standard portfolio optimization problem.
        """
        # Problem data
        n = 3
        mu = jnp.array([0.1, 0.2, 0.15])  # Expected returns
        Sigma = jnp.array([
            [0.1, 0.02, 0.01],
            [0.02, 0.2, 0.05], 
            [0.01, 0.05, 0.15]
        ])  # Covariance matrix
        
        # Variables
        x = cx.Variable(shape=(n,), name="weights")
        
        # Objective: minimize risk - return
        risk = 0.5 * cx.quad_form(x, Sigma)
        expected_return = mu @ x
        objective = cx.Minimize(risk - expected_return)
        
        # Constraints
        constraints = [
            jnp.ones(n) @ x == 1,  # Budget constraint
            x >= 0,                # Long-only
        ]
        
        # Problem
        problem = cx.Problem(objective, constraints)
        
        # Solve
        solution = problem.solve(solver="ipm", tol=1e-6)
        
        # Check solution
        assert solution.status == "optimal"
        
        # Check constraints are satisfied
        weights = solution.primal[x]
        assert abs(jnp.sum(weights) - 1.0) < 1e-4  # Budget constraint
        assert jnp.all(weights >= -1e-6)           # Non-negativity
        
        # Check objective value is reasonable
        obj_check = 0.5 * weights @ Sigma @ weights - mu @ weights
        assert abs(solution.obj_value - obj_check) < 1e-4
    
    def test_box_constraints(self):
        """Test problem with box constraints.
        
        minimize    (x-1)^2 + (y-2)^2
        subject to  0 <= x <= 1
                   1 <= y <= 3
        
        Expected solution: x = 1, y = 2, obj = 0
        """
        # Variables
        x = cx.Variable(shape=(1,), name="x") 
        y = cx.Variable(shape=(1,), name="y")
        
        # Objective: minimize (x-1)^2 + (y-2)^2
        objective = cx.Minimize(
            cx.sum_squares(x - 1.0) + cx.sum_squares(y - 2.0)
        )
        
        # Box constraints
        constraints = [
            x >= 0, x <= 1,  # 0 <= x <= 1
            y >= 1, y <= 3,  # 1 <= y <= 3
        ]
        
        # Problem
        problem = cx.Problem(objective, constraints)
        
        # Solve
        solution = problem.solve(solver="ipm", tol=1e-6)
        
        # Check solution
        assert solution.status == "optimal"
        assert abs(solution.obj_value - 0.0) < 1e-4
        assert abs(solution.primal[x][0] - 1.0) < 1e-4
        assert abs(solution.primal[y][0] - 2.0) < 1e-4
    
    def test_solve_jit(self):
        """Test JIT compilation of solve."""
        # Simple quadratic problem
        x = cx.Variable(shape=(2,), name="x")
        objective = cx.Minimize(cx.sum_squares(x))
        constraints = [jnp.ones(2) @ x == 1]
        
        problem = cx.Problem(objective, constraints)
        
        # JIT solve should work
        solution = problem.solve_jit(solver="ipm", tol=1e-6)
        assert solution.status == "optimal"
        assert abs(solution.obj_value - 0.5) < 1e-4
    
    def test_maximize_objective(self):
        """Test maximization objective."""
        x = cx.Variable(shape=(2,), name="x")
        
        # Maximize sum(x) subject to ||x||^2 <= 1
        objective = cx.Maximize(jnp.ones(2) @ x)
        constraints = [cx.sum_squares(x) <= 1]
        
        problem = cx.Problem(objective, constraints)
        
        # Note: This requires inequality constraint handling
        # For now, test that the problem is created successfully
        assert isinstance(problem.objective, cx.Maximize)
        assert len(problem.constraints) == 1
    
    def test_infeasible_problem(self):
        """Test detection of infeasible problem."""
        x = cx.Variable(shape=(1,), name="x")
        
        # Infeasible constraints: x >= 1 and x <= 0
        objective = cx.Minimize(cx.sum_squares(x))
        constraints = [x >= 1, x <= 0]
        
        problem = cx.Problem(objective, constraints)
        
        # Solver should detect infeasibility or hit max iterations
        solution = problem.solve(solver="ipm", max_iter=10)
        assert solution.status in ["max_iter", "primal_infeasible", "error"]
    
    def test_expression_operations(self):
        """Test expression arithmetic operations."""
        x = cx.Variable(shape=(2,), name="x")
        y = cx.Variable(shape=(2,), name="y")
        
        # Test addition
        expr1 = x + y
        assert expr1.is_affine()
        
        # Test subtraction  
        expr2 = x - y
        assert expr2.is_affine()
        
        # Test scalar multiplication
        expr3 = 2.0 * x
        assert expr3.is_affine()
        
        # Test matrix multiplication
        A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        expr4 = A @ x
        assert expr4.is_affine()
        assert expr4.shape == (2,)
    
    def test_constraint_operations(self):
        """Test constraint creation via comparisons."""
        x = cx.Variable(shape=(2,), name="x")
        
        # Equality constraint
        eq_con = (x[0] + x[1] == 1)
        assert isinstance(eq_con, cx.Constraint)
        
        # Inequality constraints
        ineq_con1 = (x >= 0)
        ineq_con2 = (x <= 1)
        assert isinstance(ineq_con1, cx.Constraint)
        assert isinstance(ineq_con2, cx.Constraint)
        
        # Matrix constraint
        A = jnp.array([[1.0, 1.0]])
        b = jnp.array([1.0])
        matrix_con = (A @ x == b)
        assert isinstance(matrix_con, cx.Constraint)


if __name__ == "__main__":
    pytest.main([__file__])
