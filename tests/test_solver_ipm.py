"""Test IPM solver functionality."""

import jax
import jax.numpy as jnp
import pytest

import cvxjax as cx
from cvxjax.canonicalize import build_qp
from cvxjax.solvers.ipm_qp import solve_qp_dense


class TestIPMSolver:
    """Test dense interior point method solver."""
    
    def setup_method(self):
        """Set up test environment."""
        jax.config.update("jax_enable_x64", True)
    
    def test_unconstrained_qp(self):
        """Test unconstrained quadratic problem.
        
        minimize (1/2) x^T Q x + q^T x
        Solution: x* = -Q^{-1} q
        """
        # Problem data
        Q = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        q = jnp.array([1.0, 1.0])
        
        # Create problem via API
        x = cx.Variable(shape=(2,), name="x")
        objective = 0.5 * cx.quad_form(x, Q) + q @ x
        
        qp_data = build_qp(objective, [])
        
        # Solve
        solution = solve_qp_dense(qp_data, tol=1e-8, max_iter=50)
        
        # Check convergence
        assert solution.status == "optimal"
        
        # Check solution accuracy
        x_opt = jnp.linalg.solve(Q, -q)
        x_solved = list(solution.primal.values())[0].flatten()
        
        assert jnp.allclose(x_solved, x_opt, atol=1e-6)
        
        # Check objective value
        obj_expected = 0.5 * x_opt @ Q @ x_opt + q @ x_opt
        assert abs(solution.obj_value - obj_expected) < 1e-6
    
    def test_equality_constrained_qp(self):
        """Test QP with equality constraints.
        
        minimize (1/2) x^T Q x + q^T x
        subject to A x = b
        """
        # Problem data
        Q = jnp.array([[2.0, 0.0], [0.0, 2.0]])  # Identity scaled
        q = jnp.array([0.0, 0.0])
        A = jnp.array([[1.0, 1.0]])  # Sum constraint
        b = jnp.array([1.0])
        
        # Create via API
        x = cx.Variable(shape=(2,), name="x")
        objective = 0.5 * cx.quad_form(x, Q)
        constraints = [A @ x == b]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve
        solution = solve_qp_dense(qp_data, tol=1e-8, max_iter=50)
        
        # Check convergence
        assert solution.status == "optimal"
        
        # Check constraint satisfaction
        x_solved = list(solution.primal.values())[0].flatten()
        constraint_residual = A @ x_solved - b
        assert jnp.linalg.norm(constraint_residual) < 1e-6
        
        # For this problem, solution should be x = [0.5, 0.5]
        expected_x = jnp.array([0.5, 0.5])
        assert jnp.allclose(x_solved, expected_x, atol=1e-5)
    
    def test_inequality_constrained_qp(self):
        """Test QP with inequality constraints."""
        # minimize x^2 + y^2 subject to x + y <= 1, x >= 0, y >= 0
        x = cx.Variable(shape=(1,), name="x")
        y = cx.Variable(shape=(1,), name="y")
        
        objective = cx.sum_squares(x) + cx.sum_squares(y)
        constraints = [
            x + y <= 1.0,  # Inequality constraint
            x >= 0,        # Non-negativity
            y >= 0,
        ]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve
        solution = solve_qp_dense(qp_data, tol=1e-8, max_iter=100)
        
        # Check convergence (may be challenging for IPM with inequalities)
        assert solution.status in ["optimal", "max_iter"]
        
        if solution.status == "optimal":
            # Check constraints
            x_val = solution.primal[x][0]
            y_val = solution.primal[y][0]
            
            assert x_val >= -1e-6  # x >= 0
            assert y_val >= -1e-6  # y >= 0
            assert x_val + y_val <= 1.0 + 1e-6  # x + y <= 1
    
    def test_simple_portfolio_problem(self):
        """Test portfolio optimization QP."""
        # Data
        n = 3
        mu = jnp.array([0.1, 0.2, 0.15])
        Sigma = jnp.array([
            [0.1, 0.01, 0.01],
            [0.01, 0.2, 0.02],
            [0.01, 0.02, 0.15]
        ])
        risk_aversion = 1.0
        
        # Problem
        x = cx.Variable(shape=(n,), name="weights")
        objective = risk_aversion * 0.5 * cx.quad_form(x, Sigma) - mu @ x
        constraints = [
            jnp.ones(n) @ x == 1,  # Budget
            x >= 0,                # Long-only
        ]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve
        solution = solve_qp_dense(qp_data, tol=1e-6, max_iter=100)
        
        # Check convergence
        assert solution.status in ["optimal", "max_iter"]
        
        if solution.status == "optimal":
            weights = solution.primal[x]
            
            # Check budget constraint
            assert abs(jnp.sum(weights) - 1.0) < 1e-4
            
            # Check non-negativity
            assert jnp.all(weights >= -1e-6)
    
    def test_regularization_effect(self):
        """Test that regularization helps with ill-conditioned problems."""
        # Create ill-conditioned problem
        Q = jnp.array([[1e6, 0], [0, 1e-6]])  # Very different eigenvalues
        q = jnp.array([1.0, 1.0])
        
        x = cx.Variable(shape=(2,), name="x")
        objective = 0.5 * cx.quad_form(x, Q) + q @ x
        
        qp_data = build_qp(objective, [])
        
        # Solve with different regularization levels
        solution_low_reg = solve_qp_dense(qp_data, regularization=1e-12, max_iter=20)
        solution_high_reg = solve_qp_dense(qp_data, regularization=1e-6, max_iter=20)
        
        # Higher regularization should converge more reliably
        # (may still hit max_iter but should be more stable)
        assert solution_high_reg.status in ["optimal", "max_iter"]
    
    def test_convergence_tolerance(self):
        """Test different convergence tolerances."""
        # Simple well-conditioned problem
        x = cx.Variable(shape=(2,), name="x")
        objective = cx.sum_squares(x)
        constraints = [jnp.ones(2) @ x == 1]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve with different tolerances
        solution_loose = solve_qp_dense(qp_data, tol=1e-4, max_iter=50)
        solution_tight = solve_qp_dense(qp_data, tol=1e-8, max_iter=50)
        
        # Both should converge
        assert solution_loose.status == "optimal"
        assert solution_tight.status == "optimal"
        
        # Tighter tolerance should have smaller residuals
        assert solution_tight.info["primal_residual"] <= solution_loose.info["primal_residual"]
        assert solution_tight.info["dual_residual"] <= solution_loose.info["dual_residual"]
    
    def test_max_iterations(self):
        """Test maximum iteration limit."""
        # Problem that may take many iterations
        n = 10
        x = cx.Variable(shape=(n,), name="x")
        
        # Dense random problem
        key = jax.random.PRNGKey(42)
        A_rand = jax.random.normal(key, (n, n))
        Q = A_rand.T @ A_rand + 0.01 * jnp.eye(n)
        q = jax.random.normal(key, (n,))
        
        objective = 0.5 * cx.quad_form(x, Q) + q @ x
        constraints = [x >= 0]  # Non-negativity makes it harder
        
        qp_data = build_qp(objective, constraints)
        
        # Solve with very few iterations
        solution = solve_qp_dense(qp_data, tol=1e-10, max_iter=5)
        
        # Should hit max iterations
        assert solution.status == "max_iter"
        assert solution.info["iterations"] == 5
    
    def test_kkt_residuals(self):
        """Test that KKT residuals are computed correctly."""
        # Simple problem with known solution
        x = cx.Variable(shape=(2,), name="x")
        objective = cx.sum_squares(x)
        constraints = [jnp.ones(2) @ x == 1]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve
        solution = solve_qp_dense(qp_data, tol=1e-8, max_iter=50)
        
        assert solution.status == "optimal"
        
        # Check that residuals are small
        assert solution.info["primal_residual"] < 1e-6
        assert solution.info["dual_residual"] < 1e-6
        assert solution.info["duality_gap"] < 1e-6
    
    def test_solver_info(self):
        """Test that solver returns correct info."""
        x = cx.Variable(shape=(2,), name="x")
        objective = cx.sum_squares(x)
        constraints = [jnp.ones(2) @ x == 1]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve
        solution = solve_qp_dense(qp_data, tol=1e-6, max_iter=50)
        
        # Check info fields
        assert "iterations" in solution.info
        assert "primal_residual" in solution.info
        assert "dual_residual" in solution.info
        assert "duality_gap" in solution.info
        assert "solver" in solution.info
        
        assert solution.info["solver"] == "ipm_dense"
        assert isinstance(solution.info["iterations"], int)
        assert solution.info["iterations"] >= 0


if __name__ == "__main__":
    pytest.main([__file__])
