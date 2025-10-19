"""
Improved JIT-compatible IPM solver for CVXJax integration.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple
from cvxjax.canonicalize import QPData


class IPMSolutionJIT(NamedTuple):
    """JIT-compatible IPM solution structure."""
    x: jnp.ndarray  # Primal solution
    obj_value: jnp.ndarray  # Objective value (scalar)
    iterations: jnp.ndarray  # Number of iterations (scalar)
    primal_residual: jnp.ndarray  # Primal residual norm
    dual_residual: jnp.ndarray  # Dual residual norm
    converged: jnp.ndarray  # Convergence flag (boolean)


@jax.jit 
def solve_qp_ipm_core(
    Q: jnp.ndarray,
    q: jnp.ndarray, 
    A_eq: jnp.ndarray,
    b_eq: jnp.ndarray,
    lb: jnp.ndarray,
    ub: jnp.ndarray,
    tol: float = 1e-8,
    max_iter: int = 50,
    regularization: float = 1e-12
) -> IPMSolutionJIT:
    """JIT-compatible IPM core solver.
    
    Solves box-constrained QP with equality constraints:
        minimize    (1/2) x^T Q x + q^T x
        subject to  A_eq x = b_eq
                   lb <= x <= ub
    
    Uses log-barrier interior point method.
    """
    n_vars = Q.shape[0]
    n_eq = A_eq.shape[0]
    
    # Initialize at feasible point
    eps = 1e-6
    x = jnp.maximum(lb + eps, jnp.minimum(ub - eps, 0.5 * (lb + ub)))
    
    # Project onto equality constraints if they exist
    if n_eq > 0:
        # Find projection: minimize ||x - x0||^2 subject to A_eq @ x = b_eq
        # KKT: [2I  A_eq^T] [x] = [2*x0]
        #      [A_eq  0   ] [y]   [b_eq]
        I = jnp.eye(n_vars)
        kkt_proj = jnp.block([[2*I, A_eq.T], [A_eq, jnp.zeros((n_eq, n_eq))]])
        rhs_proj = jnp.concatenate([2*x, b_eq])
        
        # Solve with regularization for numerical stability
        kkt_reg = kkt_proj + regularization * jnp.eye(kkt_proj.shape[0])
        try:
            sol_proj = jnp.linalg.solve(kkt_reg, rhs_proj)
            x_proj = sol_proj[:n_vars]
            # Ensure feasibility
            x = jnp.maximum(lb + eps, jnp.minimum(ub - eps, x_proj))
        except:
            # Fallback: use original point
            pass
    
    # Barrier parameter
    mu = 1.0
    mu_decay = 0.2
    
    def barrier_step(carry):
        x, mu, iteration = carry
        
        # Objective: f(x) - mu * (sum log(x-lb) + sum log(ub-x))
        f_grad = Q @ x + q
        
        # Barrier gradient
        barrier_grad = -mu / jnp.maximum(x - lb, eps) + mu / jnp.maximum(ub - x, eps)
        total_grad = f_grad + barrier_grad
        
        # Barrier Hessian  
        barrier_hess_diag = mu / jnp.maximum((x - lb)**2, eps**2) + mu / jnp.maximum((ub - x)**2, eps**2)
        total_hess = Q + jnp.diag(barrier_hess_diag) + regularization * jnp.eye(n_vars)
        
        # Newton step
        if n_eq > 0:
            # Equality constrained Newton step
            eq_residual = A_eq @ x - b_eq
            
            # KKT system: [H  A_eq^T] [dx] = [-grad]
            #             [A_eq  0  ] [dy]   [-res ]
            kkt_newton = jnp.block([
                [total_hess, A_eq.T],
                [A_eq, jnp.zeros((n_eq, n_eq))]
            ])
            rhs_newton = jnp.concatenate([-total_grad, -eq_residual])
            
            # Solve Newton system
            kkt_newton_reg = kkt_newton + regularization * jnp.eye(kkt_newton.shape[0])
            sol_newton = jnp.linalg.solve(kkt_newton_reg, rhs_newton)
            dx = sol_newton[:n_vars]
        else:
            # Unconstrained Newton step
            dx = -jnp.linalg.solve(total_hess, total_grad)
        
        # Line search to maintain feasibility
        # Find maximum step that keeps x within bounds
        alpha_max = 1.0
        
        # For each variable, find max step that keeps it in bounds
        for i in range(n_vars):
            if dx[i] > 0:  # Moving towards upper bound
                step_limit = (ub[i] - x[i] - eps) / dx[i]
                alpha_max = jnp.minimum(alpha_max, step_limit)
            elif dx[i] < 0:  # Moving towards lower bound  
                step_limit = (lb[i] - x[i] + eps) / dx[i]
                alpha_max = jnp.minimum(alpha_max, step_limit)
        
        # Use fraction of maximum step
        alpha = 0.95 * jnp.maximum(alpha_max, 0.0)
        alpha = jnp.minimum(alpha, 1.0)
        
        # Update
        x_new = x + alpha * dx
        x_new = jnp.maximum(lb + eps, jnp.minimum(ub - eps, x_new))
        
        # Update barrier parameter
        mu_new = mu * mu_decay
        
        return x_new, mu_new, iteration + 1
    
    def continue_condition(carry):
        x, mu, iteration = carry
        
        # Check convergence based on KKT conditions
        f_grad = Q @ x + q
        
        # For barrier method, optimality at x* means:
        # Q x* + q + lambda = 0  (stationarity)
        # A_eq x* = b_eq        (primal feasibility)
        # where lambda are the barrier-induced multipliers
        
        # Approximate KKT stationarity
        dual_residual = jnp.linalg.norm(f_grad)
        
        # Primal feasibility
        if n_eq > 0:
            primal_residual = jnp.linalg.norm(A_eq @ x - b_eq)
        else:
            primal_residual = 0.0
        
        # Barrier term indicates how close to boundary we are
        barrier_term = mu * (jnp.sum(1.0 / jnp.maximum(x - lb, eps)) + 
                           jnp.sum(1.0 / jnp.maximum(ub - x, eps)))
        
        # Convergence: small residuals and small barrier parameter
        converged = jnp.logical_and(
            jnp.logical_and(dual_residual < tol, primal_residual < tol),
            mu < tol
        )
        
        return jnp.logical_and(iteration < max_iter, jnp.logical_not(converged))
    
    # Run barrier method iterations
    init_carry = (x, mu, jnp.array(0))
    final_carry = jax.lax.while_loop(continue_condition, barrier_step, init_carry)
    
    x_final, mu_final, iterations = final_carry
    
    # Compute final residuals
    f_grad = Q @ x_final + q
    dual_residual = jnp.linalg.norm(f_grad)
    
    if n_eq > 0:
        primal_residual = jnp.linalg.norm(A_eq @ x_final - b_eq)
    else:
        primal_residual = 0.0
    
    # Final convergence check
    converged = jnp.logical_and(
        jnp.logical_and(dual_residual <= tol, primal_residual <= tol),
        mu_final <= tol
    )
    
    # Objective value
    obj_value = 0.5 * x_final.T @ Q @ x_final + q.T @ x_final
    
    return IPMSolutionJIT(
        x=x_final,
        obj_value=obj_value,
        iterations=iterations,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        converged=converged
    )


def solve_qp_ipm_wrapper(qp_data: QPData, **kwargs) -> "Solution":
    """Wrapper to integrate JIT IPM with CVXJax Solution format."""
    # Extract problem data
    Q = qp_data.Q
    q = qp_data.q
    A_eq = qp_data.A_eq if qp_data.n_eq > 0 else jnp.zeros((0, qp_data.n_vars))
    b_eq = qp_data.b_eq if qp_data.n_eq > 0 else jnp.zeros(0)
    lb = qp_data.lb
    ub = qp_data.ub
    
    # Set default parameters
    tol = kwargs.get('tol', 1e-8)
    max_iter = kwargs.get('max_iter', 50)
    regularization = kwargs.get('regularization', 1e-12)
    
    # Solve with JIT core
    jit_solution = solve_qp_ipm_core(Q, q, A_eq, b_eq, lb, ub, tol, max_iter, regularization)
    
    # Build CVXJax Solution object
    x_opt = jit_solution.x
    obj_value = jit_solution.obj_value
    
    # Build primal solution mapping
    primal = {}
    start_idx = 0
    for var in qp_data.variables:
        var_size = int(jnp.prod(jnp.array(var.shape)))
        end_idx = start_idx + var_size
        var_value = x_opt[start_idx:end_idx].reshape(var.shape)
        primal[var] = var_value
        primal[var.name] = var_value
        start_idx = end_idx
    
    # Build dual solution (simplified for box constraints)
    dual = {}
    
    # Compute dual variables for bounds (simplified)
    eps = 1e-8
    lb_active = x_opt <= lb + eps
    ub_active = x_opt >= ub - eps
    
    dual_lb = jnp.where(lb_active, jnp.maximum(0.0, -(Q @ x_opt + q)), 0.0)
    dual_ub = jnp.where(ub_active, jnp.maximum(0.0, Q @ x_opt + q), 0.0)
    
    if qp_data.n_eq > 0:
        # Dual for equality constraints (from KKT system)
        dual["eq_constraints"] = jnp.zeros(qp_data.n_eq)  # Simplified
    
    # Info dictionary
    info = {
        "iterations": int(jit_solution.iterations),
        "primal_residual": float(jit_solution.primal_residual),
        "dual_residual": float(jit_solution.dual_residual),
        "converged": bool(jit_solution.converged),
        "solver": "ipm_jit",
    }
    
    # Determine status
    if jit_solution.converged:
        status = "optimal"
    elif jit_solution.iterations >= max_iter:
        status = "max_iter"
    else:
        status = "unknown"
    
    from cvxjax.api import Solution
    return Solution(
        status=status,
        obj_value=obj_value,
        primal=primal,
        dual=dual,
        info=info,
    )


def test_jit_ipm_integration():
    """Test JIT IPM integration with CVXJax."""
    print("🔗 TESTING JIT IPM INTEGRATION WITH CVXJAX")
    print("=" * 50)
    
    import time
    
    # Create test QP
    n = 3
    Q = jnp.array([[2.0, 0.5, 0.0],
                   [0.5, 1.0, 0.0],
                   [0.0, 0.0, 1.5]], dtype=jnp.float32)
    q = jnp.array([1.0, -2.0, 0.5], dtype=jnp.float32)
    
    # Equality constraint: x[0] + x[1] = 1
    A_eq = jnp.array([[1.0, 1.0, 0.0]], dtype=jnp.float32)
    b_eq = jnp.array([1.0], dtype=jnp.float32)
    
    # Box constraints
    lb = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)
    ub = jnp.array([2.0, 2.0, 2.0], dtype=jnp.float32)
    
    # Create QPData
    class DummyVariable:
        def __init__(self, name, shape):
            self.name = name
            self.shape = shape
    
    variables = [DummyVariable('x', (n,))]
    qp_data = QPData(
        Q=Q, q=q, constant=0.0,
        A_eq=A_eq, b_eq=b_eq,
        A_ineq=jnp.zeros((0, n)), b_ineq=jnp.zeros(0),
        lb=lb, ub=ub,
        variables=variables,
        n_vars=n, n_eq=1, n_ineq=0
    )
    
    print("Test problem:")
    print(f"  Variables: {n}")
    print(f"  Equality constraints: {qp_data.n_eq}")
    print(f"  Box constraints: {lb} <= x <= {ub}")
    print()
    
    # Test JIT IPM wrapper
    print("🚀 Testing JIT IPM wrapper...")
    start_time = time.time()
    solution = solve_qp_ipm_wrapper(qp_data, tol=1e-6, max_iter=30)
    solve_time = time.time() - start_time
    
    print(f"✅ Completed in {solve_time:.4f} seconds")
    print(f"Status: {solution.status}")
    print(f"Solution: x = {solution.primal['x']}")
    print(f"Objective: {solution.obj_value:.6f}")
    print(f"Iterations: {solution.info['iterations']}")
    print(f"Converged: {solution.info['converged']}")
    print()
    
    # Verify constraints
    x_opt = solution.primal['x']
    eq_residual = jnp.linalg.norm(A_eq @ x_opt - b_eq)
    bound_violations = jnp.sum(jnp.logical_or(x_opt < lb - 1e-6, x_opt > ub + 1e-6))
    
    print("🔍 Constraint verification:")
    print(f"  Equality residual: {eq_residual:.2e}")
    print(f"  Bound violations: {bound_violations}")
    print(f"  x in bounds: {jnp.all(jnp.logical_and(x_opt >= lb - 1e-6, x_opt <= ub + 1e-6))}")
    
    return solution


if __name__ == "__main__":
    test_jit_ipm_integration()