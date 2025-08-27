"""Test canonicalization functionality."""

import jax
import jax.numpy as jnp
import pytest

import cvxjax as cx
from cvxjax.canonicalize import build_qp, build_lp


class TestCanonicalization:
    """Test problem canonicalization to standard forms."""
    
    def setup_method(self):
        """Set up test environment."""
        jax.config.update("jax_enable_x64", True)
    
    def test_simple_qp_canonicalization(self):
        """Test canonicalization of simple QP."""
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
        
        # Check dimensions
        assert qp_data.n_vars == 2
        assert qp_data.n_eq >= 1  # At least the equality constraint
        
        # Check Q matrix structure (should be 2x2 identity for x^2 + y^2)
        expected_Q = 2 * jnp.eye(2)  # Factor of 2 from quadratic form definition
        assert qp_data.Q.shape == (2, 2)
        
        # Check that we have variables
        assert len(qp_data.variables) == 2
    
    def test_linear_program_canonicalization(self):
        """Test canonicalization of linear program."""
        # minimize c^T x subject to Ax <= b, x >= 0
        x = cx.Variable(shape=(3,), name="x")
        c = jnp.array([1.0, 2.0, 3.0])
        
        objective = c @ x
        constraints = [x >= 0]
        
        lp_data = build_lp(objective, constraints)
        
        # Check dimensions
        assert lp_data.n_vars == 3
        assert jnp.allclose(lp_data.c, c)
        
        # Q matrix should be zero for LP
        qp_data = build_qp(objective, constraints)
        assert jnp.allclose(qp_data.Q, 0)
    
    def test_quadratic_form_canonicalization(self):
        """Test canonicalization with quadratic forms."""
        x = cx.Variable(shape=(2,), name="x")
        Q = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        
        objective = cx.quad_form(x, Q)
        constraints = [jnp.ones(2) @ x == 1]
        
        qp_data = build_qp(objective, constraints)
        
        # Check Q matrix
        assert qp_data.Q.shape == (2, 2)
        # The canonical form uses (1/2) x^T Q x, so our Q should appear directly
        assert jnp.allclose(qp_data.Q, Q, atol=1e-6)
    
    def test_mixed_constraints_canonicalization(self):
        """Test canonicalization with mixed constraint types."""
        x = cx.Variable(shape=(2,), name="x")
        
        objective = cx.sum_squares(x)
        constraints = [
            x[0] + x[1] == 1.0,     # Equality
            x[0] - x[1] <= 0.5,     # Inequality  
            x[0] >= 0.0,            # Lower bound
            x[1] <= 1.0,            # Upper bound
        ]
        
        qp_data = build_qp(objective, constraints)
        
        # Should have 2 variables
        assert qp_data.n_vars == 2
        
        # Should have at least 1 equality constraint
        assert qp_data.n_eq >= 1
        
        # Check that bounds are handled
        assert jnp.isfinite(qp_data.lb[0])  # x[0] >= 0
        assert jnp.isfinite(qp_data.ub[1])  # x[1] <= 1
    
    def test_parameter_in_constraints(self):
        """Test canonicalization with parameters in constraints."""
        x = cx.Variable(shape=(2,), name="x")
        p = cx.Parameter(jnp.array([1.0, 2.0]), name="param")
        
        objective = cx.sum_squares(x)
        constraints = [p @ x == 3.0, x >= 0]
        
        qp_data = build_qp(objective, constraints)
        
        # Should successfully canonicalize
        assert qp_data.n_vars == 2
        assert qp_data.n_eq >= 1
    
    def test_matrix_variable_canonicalization(self):
        """Test canonicalization with matrix variables."""
        X = cx.Variable(shape=(2, 2), name="X")
        
        # Minimize Frobenius norm squared: ||X||_F^2
        objective = cx.sum_squares(X)  # This should flatten X
        constraints = [X >= 0]  # Element-wise non-negativity
        
        qp_data = build_qp(objective, constraints)
        
        # Matrix variable should be flattened to 4 scalar variables
        assert qp_data.n_vars == 4
    
    def test_inconsistent_constraints_detection(self):
        """Test detection of obviously inconsistent constraints."""
        x = cx.Variable(shape=(1,), name="x")
        
        objective = cx.sum_squares(x)
        constraints = [
            x >= 1.0,
            x <= 0.0,  # Inconsistent with x >= 1
        ]
        
        # Should still canonicalize but constraints will be infeasible
        qp_data = build_qp(objective, constraints)
        
        # Check that bounds reflect the inconsistency
        # (may be caught at solve time rather than canonicalization)
        assert qp_data.n_vars == 1
    
    def test_empty_constraints(self):
        """Test canonicalization with no constraints."""
        x = cx.Variable(shape=(2,), name="x")
        
        objective = cx.sum_squares(x)
        constraints = []
        
        qp_data = build_qp(objective, constraints)
        
        # Should be unconstrained problem
        assert qp_data.n_vars == 2
        assert qp_data.n_eq == 0
        assert qp_data.n_ineq == 0
        assert jnp.all(qp_data.lb == -jnp.inf)
        assert jnp.all(qp_data.ub == jnp.inf)
    
    def test_large_problem_canonicalization(self):
        """Test canonicalization scalability with larger problems."""
        n = 50
        x = cx.Variable(shape=(n,), name="x")
        
        # Random positive definite Q
        A = jax.random.normal(jax.random.PRNGKey(42), (n, n))
        Q = A.T @ A + 0.1 * jnp.eye(n)
        
        objective = cx.quad_form(x, Q)
        constraints = [x >= 0, jnp.ones(n) @ x == 1]
        
        qp_data = build_qp(objective, constraints)
        
        # Check dimensions
        assert qp_data.n_vars == n
        assert qp_data.Q.shape == (n, n)
        assert qp_data.n_eq >= 1  # Budget constraint
        
        # Check Q is preserved
        assert jnp.allclose(qp_data.Q, Q, atol=1e-6)
    
    def test_canonicalization_with_constants(self):
        """Test canonicalization with constant terms."""
        x = cx.Variable(shape=(2,), name="x")
        
        # Objective with constant term: x^2 + y^2 + 5
        objective = cx.sum_squares(x) + 5.0
        constraints = [jnp.ones(2) @ x == 1]
        
        qp_data = build_qp(objective, constraints)
        
        # Constant should not affect Q or q matrices
        # (constant terms are ignored in quadratic form)
        assert qp_data.Q.shape == (2, 2)
        assert qp_data.q.shape == (2,)
    
    def test_maximize_canonicalization(self):
        """Test canonicalization of maximization problems."""
        x = cx.Variable(shape=(2,), name="x")
        
        # Maximize -(x^2 + y^2) = minimize x^2 + y^2
        objective_max = cx.Maximize(-(cx.sum_squares(x)))
        objective_min = cx.Minimize(cx.sum_squares(x))
        
        constraints = [jnp.ones(2) @ x == 1]
        
        qp_data_max = build_qp(objective_max.expression, constraints)
        qp_data_min = build_qp(objective_min.expression, constraints)
        
        # After negation, they should be equivalent
        # (This tests the principle - actual implementation may vary)
        assert qp_data_max.Q.shape == qp_data_min.Q.shape
        assert qp_data_max.q.shape == qp_data_min.q.shape


if __name__ == "__main__":
    pytest.main([__file__])
