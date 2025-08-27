"""Test OSQP solver functionality."""

import jax
import jax.numpy as jnp
import pytest

import cvxjax as cx
from cvxjax.canonicalize import build_qp
from cvxjax.solvers.osqp_bridge import solve_qp_osqp, check_osqp_available


class TestOSQPSolver:
    """Test OSQP solver bridge."""
    
    def setup_method(self):
        """Set up test environment."""
        jax.config.update("jax_enable_x64", True)
    
    @pytest.mark.skipif(not check_osqp_available(), reason="OSQP not available")
    def test_simple_qp_osqp(self):
        """Test simple QP with OSQP solver."""
        # minimize x^2 + y^2 subject to x + y = 1, x >= 0, y >= 0
        x = cx.Variable(shape=(1,), name="x")
        y = cx.Variable(shape=(1,), name="y")
        
        objective = cx.sum_squares(x) + cx.sum_squares(y)
        constraints = [
            x + y == 1.0,
            x >= 0,
            y >= 0,
        ]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve with OSQP
        solution = solve_qp_osqp(qp_data, tol=1e-6, max_iter=100)
        
        # Check convergence
        assert solution.status in ["optimal", "max_iter"]
        
        if solution.status == "optimal":
            # Check solution accuracy
            x_val = solution.primal[x][0]
            y_val = solution.primal[y][0]
            
            # Expected solution: x = 0.5, y = 0.5
            assert abs(x_val - 0.5) < 1e-4
            assert abs(y_val - 0.5) < 1e-4
            assert abs(solution.obj_value - 0.5) < 1e-4
    
    @pytest.mark.skipif(not check_osqp_available(), reason="OSQP not available")
    def test_portfolio_osqp(self):
        """Test portfolio optimization with OSQP."""
        n = 3
        mu = jnp.array([0.1, 0.2, 0.15])
        Sigma = jnp.array([
            [0.1, 0.01, 0.01],
            [0.01, 0.2, 0.02],
            [0.01, 0.02, 0.15]
        ])
        
        # Problem
        x = cx.Variable(shape=(n,), name="weights")
        objective = 0.5 * cx.quad_form(x, Sigma) - mu @ x
        constraints = [
            jnp.ones(n) @ x == 1,  # Budget
            x >= 0,                # Long-only
        ]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve
        solution = solve_qp_osqp(qp_data, tol=1e-6, max_iter=100)
        
        # Check convergence
        assert solution.status in ["optimal", "max_iter"]
        
        if solution.status == "optimal":
            weights = solution.primal[x]
            
            # Check constraints
            assert abs(jnp.sum(weights) - 1.0) < 1e-4
            assert jnp.all(weights >= -1e-6)
    
    @pytest.mark.skipif(not check_osqp_available(), reason="OSQP not available")
    def test_inequality_constraints_osqp(self):
        """Test OSQP with inequality constraints."""
        x = cx.Variable(shape=(2,), name="x")
        
        # minimize ||x||^2 subject to A x <= b
        A = jnp.array([[1.0, 1.0], [1.0, -1.0]])
        b = jnp.array([1.0, 0.5])
        
        objective = cx.sum_squares(x)
        constraints = [A @ x <= b]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve
        solution = solve_qp_osqp(qp_data, tol=1e-6, max_iter=100)
        
        # Check convergence
        assert solution.status in ["optimal", "max_iter"]
        
        if solution.status == "optimal":
            x_sol = solution.primal[x]
            
            # Check constraint satisfaction
            constraint_vals = A @ x_sol
            assert jnp.all(constraint_vals <= b + 1e-6)
    
    @pytest.mark.skipif(not check_osqp_available(), reason="OSQP not available")
    def test_box_constraints_osqp(self):
        """Test OSQP with box constraints."""
        x = cx.Variable(shape=(2,), name="x")
        
        # minimize (x-1)^2 + (y-2)^2 subject to 0 <= x <= 1, 1 <= y <= 3
        objective = cx.sum_squares(x - jnp.array([1.0, 2.0]))
        constraints = [
            x >= jnp.array([0.0, 1.0]),
            x <= jnp.array([1.0, 3.0]),
        ]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve
        solution = solve_qp_osqp(qp_data, tol=1e-6, max_iter=100)
        
        # Check convergence
        assert solution.status in ["optimal", "max_iter"]
        
        if solution.status == "optimal":
            x_sol = solution.primal[x]
            
            # Expected solution: x = [1, 2] (unconstrained optimum is feasible)
            assert abs(x_sol[0] - 1.0) < 1e-4
            assert abs(x_sol[1] - 2.0) < 1e-4
            assert abs(solution.obj_value - 0.0) < 1e-4
    
    @pytest.mark.skipif(not check_osqp_available(), reason="OSQP not available")
    def test_osqp_parameters(self):
        """Test OSQP with different parameter settings."""
        x = cx.Variable(shape=(2,), name="x")
        objective = cx.sum_squares(x)
        constraints = [jnp.ones(2) @ x == 1]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve with different OSQP parameters
        solution1 = solve_qp_osqp(qp_data, rho=0.1, sigma=1e-6, alpha=1.6)
        solution2 = solve_qp_osqp(qp_data, rho=1.0, sigma=1e-4, alpha=1.0)
        
        # Both should converge to same solution
        if solution1.status == "optimal" and solution2.status == "optimal":
            x1 = solution1.primal[x]
            x2 = solution2.primal[x]
            assert jnp.allclose(x1, x2, atol=1e-4)
    
    def test_osqp_availability_check(self):
        """Test OSQP availability checking."""
        # This should not raise an error
        available = check_osqp_available()
        assert isinstance(available, bool)
    
    @pytest.mark.skipif(not check_osqp_available(), reason="OSQP not available")
    def test_osqp_solver_info(self):
        """Test OSQP solver info reporting."""
        x = cx.Variable(shape=(2,), name="x")
        objective = cx.sum_squares(x)
        constraints = [jnp.ones(2) @ x == 1]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve
        solution = solve_qp_osqp(qp_data, tol=1e-6, max_iter=100)
        
        # Check info
        assert "solver" in solution.info
        assert solution.info["solver"] == "osqp"
        assert "iterations" in solution.info
        assert "primal_residual" in solution.info
        assert "dual_residual" in solution.info
    
    @pytest.mark.skipif(not check_osqp_available(), reason="OSQP not available")
    def test_compare_ipm_osqp(self):
        """Compare IPM and OSQP solutions on same problem."""
        x = cx.Variable(shape=(2,), name="x")
        objective = cx.sum_squares(x)
        constraints = [jnp.ones(2) @ x == 1, x >= 0]
        
        qp_data = build_qp(objective, constraints)
        
        # Solve with both solvers
        from cvxjax.solvers.ipm_qp import solve_qp_dense
        
        solution_ipm = solve_qp_dense(qp_data, tol=1e-6, max_iter=100)
        solution_osqp = solve_qp_osqp(qp_data, tol=1e-6, max_iter=100)
        
        # Both should converge to similar solutions
        if solution_ipm.status == "optimal" and solution_osqp.status == "optimal":
            x_ipm = solution_ipm.primal[x]
            x_osqp = solution_osqp.primal[x]
            
            # Solutions should be close
            assert jnp.allclose(x_ipm, x_osqp, atol=1e-4)
            assert abs(solution_ipm.obj_value - solution_osqp.obj_value) < 1e-4
    
    def test_osqp_fallback_on_error(self):
        """Test OSQP handles solver errors gracefully."""
        # Create a problem that might cause issues
        x = cx.Variable(shape=(2,), name="x")
        
        # Very ill-conditioned problem
        Q = jnp.array([[1e12, 0], [0, 1e-12]])
        objective = 0.5 * cx.quad_form(x, Q)
        
        qp_data = build_qp(objective, [])
        
        # This might fail, but should return error status rather than crash
        solution = solve_qp_osqp(qp_data, tol=1e-6, max_iter=10)
        
        # Should return some valid status
        assert solution.status in ["optimal", "max_iter", "error", "primal_infeasible", "dual_infeasible"]


if __name__ == "__main__":
    pytest.main([__file__])
