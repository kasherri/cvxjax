#!/usr/bin/env python3
"""Test BoxCDQP solver implementation."""

import cvxjax as cx
import jax.numpy as jnp


def test_boxcdqp_unconstrained():
    """Test unconstrained quadratic: minimize x^2 - 4*x."""
    x = cx.Variable(shape=(1,), name='x_unconstrained')
    objective = cx.sum_squares(x) - 4 * x
    problem = cx.Problem(cx.Minimize(objective))
    solution = problem.solve(solver='boxcdqp')
    
    x_val = float(solution.primal[x][0])
    obj_val = float(solution.obj_value)
    
    assert abs(x_val - 2.0) < 1e-5, f"Expected x=2.0, got x={x_val}"
    assert abs(obj_val + 4.0) < 1e-5, f"Expected obj=-4.0, got obj={obj_val}"
    assert solution.status == "optimal"
    
    print("✓ Test 1 (unconstrained) passed")


def test_boxcdqp_simple_box():
    """Test box-constrained: minimize x^2 + y^2 subject to 0 <= x,y <= 5."""
    x = cx.Variable(shape=(1,), name='x_simple')
    y = cx.Variable(shape=(1,), name='y_simple')
    objective = cx.sum_squares(x) + cx.sum_squares(y)
    constraints = [x >= 0, y >= 0, x <= 5, y <= 5]
    problem = cx.Problem(cx.Minimize(objective), constraints)
    solution = problem.solve(solver='boxcdqp')
    
    x_val = float(solution.primal[x][0])
    y_val = float(solution.primal[y][0])
    obj_val = float(solution.obj_value)
    
    assert abs(x_val) < 1e-5, f"Expected x=0.0, got x={x_val}"
    assert abs(y_val) < 1e-5, f"Expected y=0.0, got y={y_val}"
    assert abs(obj_val) < 1e-5, f"Expected obj=0.0, got obj={obj_val}"
    assert solution.status == "optimal"
    
    print("✓ Test 2 (simple box constraints) passed")


def test_boxcdqp_interior_optimum():
    """Test box-constrained with optimum in interior: minimize (x-3)^2 + (y-2)^2."""
    x = cx.Variable(shape=(1,), name='x_interior')
    y = cx.Variable(shape=(1,), name='y_interior')
    objective = cx.sum_squares(x - 3) + cx.sum_squares(y - 2)
    constraints = [x >= 0, y >= 0, x <= 5, y <= 5]
    problem = cx.Problem(cx.Minimize(objective), constraints)
    solution = problem.solve(solver='boxcdqp')
    
    x_val = float(solution.primal[x][0])
    y_val = float(solution.primal[y][0])
    obj_val = float(solution.obj_value)
    
    assert abs(x_val - 3.0) < 1e-5, f"Expected x=3.0, got x={x_val}"
    assert abs(y_val - 2.0) < 1e-5, f"Expected y=2.0, got y={y_val}"
    assert abs(obj_val) < 1e-5, f"Expected obj=0.0, got obj={obj_val}"
    assert solution.status == "optimal"
    
    print("✓ Test 3 (interior optimum) passed")


def test_boxcdqp_active_constraints():
    """Test box-constrained with active constraints: minimize (x-10)^2 + y^2."""
    x = cx.Variable(shape=(1,), name='x_active')
    y = cx.Variable(shape=(1,), name='y_active')
    objective = cx.sum_squares(x - 10) + cx.sum_squares(y)
    constraints = [x >= 0, y >= 0, x <= 5, y <= 5]
    problem = cx.Problem(cx.Minimize(objective), constraints)
    solution = problem.solve(solver='boxcdqp')
    
    x_val = float(solution.primal[x][0])
    y_val = float(solution.primal[y][0])
    obj_val = float(solution.obj_value)
    
    assert abs(x_val - 5.0) < 1e-5, f"Expected x=5.0, got x={x_val}"
    assert abs(y_val) < 1e-5, f"Expected y=0.0, got y={y_val}"
    assert abs(obj_val - 25.0) < 1e-5, f"Expected obj=25.0, got obj={obj_val}"
    assert solution.status == "optimal"
    
    print("✓ Test 4 (active constraints) passed")


def test_boxcdqp_error_cases():
    """Test that BoxCDQP properly rejects non-box-constrained problems."""
    x = cx.Variable(shape=(1,), name='x_error')
    y = cx.Variable(shape=(1,), name='y_error')
    
    # Test equality constraint rejection
    try:
        objective = cx.sum_squares(x) + cx.sum_squares(y)
        constraints = [x + y == 1]  # Equality constraint
        problem = cx.Problem(cx.Minimize(objective), constraints)
        solution = problem.solve(solver='boxcdqp')
        assert False, "Should have raised ValueError for equality constraints"
    except ValueError as e:
        assert "equality constraints" in str(e).lower()
        print("✓ Test 5a (equality constraint rejection) passed")
    
    # Test general inequality constraint rejection  
    try:
        objective = cx.sum_squares(x) + cx.sum_squares(y)
        constraints = [x + y <= 1]  # General inequality constraint
        problem = cx.Problem(cx.Minimize(objective), constraints)
        solution = problem.solve(solver='boxcdqp')
        assert False, "Should have raised ValueError for general inequality constraints"
    except ValueError as e:
        assert "non-box inequality" in str(e).lower()
        print("✓ Test 5b (general inequality rejection) passed")


if __name__ == "__main__":
    print("=== Testing BoxCDQP Solver Implementation ===")
    print()
    
    test_boxcdqp_unconstrained()
    test_boxcdqp_simple_box()
    test_boxcdqp_interior_optimum()
    test_boxcdqp_active_constraints()
    test_boxcdqp_error_cases()
    
    print()
    print("🎉 All BoxCDQP solver tests passed!")
    print()
    print("Summary:")
    print("  ✓ Unconstrained quadratic problems")
    print("  ✓ Simple box-constrained problems")
    print("  ✓ Interior optimal solutions")
    print("  ✓ Boundary optimal solutions (active constraints)")
    print("  ✓ Proper error handling for unsupported constraints")
    print()
    print("The BoxCDQP solver is successfully integrated into CVXJax!")