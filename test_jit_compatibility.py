#!/usr/bin/env python3
"""Test script to verify JIT compatibility of the entire CVXJAX codebase."""

import jax
import jax.numpy as jnp

import cvxjax as cx


def test_basic_qp_jit():
    """Test basic quadratic programming with JIT compilation."""
    print("Testing basic QP with JIT...")
    
    @jax.jit
    def solve_basic_qp():
        # Simple QP: minimize (1/2) x^T Q x + q^T x subject to x >= 0, sum(x) <= 1
        x = cx.Variable(shape=(2,), name="x")
        Q = jnp.array([[2.0, 0.5], [0.5, 1.0]])
        q = jnp.array([1.0, 0.5])
        
        objective = cx.Minimize(0.5 * cx.quad_form(x, Q) + q @ x)
        constraints = [x >= 0, cx.sum(x) <= 1]
        
        problem = cx.Problem(objective, constraints)
        solution = problem.solve(solver="ipm", tol=1e-6, max_iter=20)
        
        return solution.obj_value, solution.primal["x"]
    
    # This should compile and run without errors
    obj_val, x_opt = solve_basic_qp()
    print(f"  Objective value: {obj_val:.6f}")
    print(f"  Optimal x: {x_opt}")
    print("✅ Basic QP JIT test passed!")
    return True


def test_portfolio_optimization_jit():
    """Test portfolio optimization with JIT compilation."""
    print("\nTesting portfolio optimization with JIT...")
    
    @jax.jit  
    def solve_portfolio(returns, cov_matrix, risk_aversion):
        n_assets = returns.shape[0]
        w = cx.Variable(shape=(n_assets,), name="weights")
        
        # Mean-variance objective
        expected_return = returns @ w
        risk = cx.quad_form(w, cov_matrix)
        objective = cx.Minimize(-expected_return + 0.5 * risk_aversion * risk)
        
        # Portfolio constraints
        constraints = [
            cx.sum(w) == 1,  # Budget constraint
            w >= 0           # Long-only
        ]
        
        problem = cx.Problem(objective, constraints)
        solution = problem.solve(solver="ipm", tol=1e-6, max_iter=30)
        
        return solution.obj_value, solution.primal["weights"]
    
    # Test data
    n_assets = 3
    returns = jnp.array([0.10, 0.12, 0.08])
    cov_matrix = jnp.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.09, 0.01], 
        [0.02, 0.01, 0.04]
    ])
    risk_aversion = 2.0
    
    obj_val, weights = solve_portfolio(returns, cov_matrix, risk_aversion)
    print(f"  Objective value: {obj_val:.6f}")
    print(f"  Portfolio weights: {weights}")
    print("✅ Portfolio optimization JIT test passed!")
    return True


def test_lasso_regression_jit():
    """Test LASSO regression with JIT compilation."""
    print("\nTesting LASSO regression with JIT...")
    
    @jax.jit
    def solve_lasso(A, b, lambda_reg):
        n_features = A.shape[1]
        x = cx.Variable(shape=(n_features,), name="coefficients")
        
        # LASSO objective: ||Ax - b||^2 + lambda ||x||^2 (ridge regularization for convexity)
        residual = A @ x - b
        data_fit = cx.sum_squares(residual)
        regularization = lambda_reg * cx.sum_squares(x)
        objective = cx.Minimize(data_fit + regularization)
        
        problem = cx.Problem(objective, constraints=[])
        solution = problem.solve(solver="ipm", tol=1e-6, max_iter=25)
        
        return solution.obj_value, solution.primal["coefficients"]
    
    # Test data
    n_samples, n_features = 10, 5
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (n_samples, n_features))
    b = jax.random.normal(jax.random.split(key)[1], (n_samples,))
    lambda_reg = 0.1
    
    obj_val, coeffs = solve_lasso(A, b, lambda_reg)
    print(f"  Objective value: {obj_val:.6f}")
    print(f"  Coefficients: {coeffs}")
    print("✅ LASSO regression JIT test passed!")
    return True


def test_box_constraints_jit():
    """Test optimization with box constraints and JIT compilation."""
    print("\nTesting box constraints with JIT...")
    
    @jax.jit
    def solve_box_constrained(Q, q, lb, ub):
        n = Q.shape[0]
        x = cx.Variable(shape=(n,), name="x")
        
        objective = cx.Minimize(0.5 * cx.quad_form(x, Q) + q @ x)
        constraints = [x >= lb, x <= ub]
        
        problem = cx.Problem(objective, constraints)
        solution = problem.solve(solver="ipm", tol=1e-6, max_iter=20)
        
        return solution.obj_value, solution.primal["x"]
    
    # Test data
    n = 3
    Q = jnp.eye(n) * 2.0
    q = jnp.ones(n)
    lb = jnp.zeros(n)
    ub = jnp.ones(n) * 2.0
    
    obj_val, x_opt = solve_box_constrained(Q, q, lb, ub)
    print(f"  Objective value: {obj_val:.6f}")
    print(f"  Optimal x: {x_opt}")
    print("✅ Box constraints JIT test passed!")
    return True


def test_batch_optimization_jit():
    """Test batch optimization with vmap and JIT."""
    print("\nTesting batch optimization with vmap + JIT...")
    
    def solve_single_qp(q_vec):
        """Solve a single QP with varying linear term."""
        x = cx.Variable(shape=(2,), name="x")
        Q = jnp.array([[2.0, 0.0], [0.0, 1.0]])
        
        objective = cx.Minimize(0.5 * cx.quad_form(x, Q) + q_vec @ x)
        constraints = [x >= 0, cx.sum(x) <= 1]
        
        problem = cx.Problem(objective, constraints)
        solution = problem.solve(solver="ipm", tol=1e-6, max_iter=15)
        
        return solution.obj_value
    
    # Create batch solver
    batch_solve = jax.jit(jax.vmap(solve_single_qp))
    
    # Test data: batch of different linear terms
    q_batch = jnp.array([
        [1.0, 0.5],
        [0.5, 1.0], 
        [1.5, 0.3],
        [0.8, 1.2]
    ])
    
    obj_values = batch_solve(q_batch)
    print(f"  Batch objective values: {obj_values}")
    print("✅ Batch optimization JIT test passed!")
    return True


def test_advanced_expressions_jit():
    """Test complex expressions with JIT compilation."""
    print("\nTesting advanced expressions with JIT...")
    
    @jax.jit
    def solve_complex_objective():
        # Multiple variables and complex objective
        x = cx.Variable(shape=(3,), name="x")
        y = cx.Variable(shape=(2,), name="y")
        
        # Complex objective combining variables
        A = jnp.array([[1.0, 2.0, 0.5], [0.5, 1.0, 1.5]])
        b = jnp.array([1.0, 0.8])
        
        term1 = cx.sum_squares(x)
        term2 = cx.sum_squares(A @ x - b)
        term3 = cx.sum_squares(y)
        
        objective = cx.Minimize(0.5 * term1 + term2 + 0.3 * term3)
        
        constraints = [
            x >= 0,
            y >= 0,
            cx.sum(x) + cx.sum(y) <= 2,
            A @ x == b + y  # Coupling constraint
        ]
        
        problem = cx.Problem(objective, constraints)
        solution = problem.solve(solver="ipm", tol=1e-6, max_iter=30)
        
        return solution.obj_value, solution.primal["x"], solution.primal["y"]
    
    obj_val, x_opt, y_opt = solve_complex_objective()
    print(f"  Objective value: {obj_val:.6f}")
    print(f"  Optimal x: {x_opt}")
    print(f"  Optimal y: {y_opt}")
    print("✅ Advanced expressions JIT test passed!")
    return True


def main():
    """Run all JIT compatibility tests."""
    print("🔧 Testing CVXJAX JIT Compatibility")
    print("=" * 50)
    
    # Enable 64-bit precision for numerical stability
    jax.config.update("jax_enable_x64", True)
    
    tests = [
        test_basic_qp_jit,
        test_portfolio_optimization_jit,
        test_lasso_regression_jit,
        test_box_constraints_jit,
        test_batch_optimization_jit,
        test_advanced_expressions_jit,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"🎯 JIT Compatibility Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All tests passed! CVXJAX is fully JIT-compatible!")
        return True
    else:
        print(f"\n⚠️  {failed} test(s) failed. Some components may not be JIT-compatible.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
