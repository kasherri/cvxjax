#!/usr/bin/env python3
"""Portfolio optimization with CVXJAX.

This example demonstrates portfolio optimization using CVXJAX, implementing
classic mean-variance optimization with various constraints. We show how to:

1. Basic mean-variance optimization
2. Long-only constraints
3. Box constraints (position limits)
4. Transaction cost modeling
5. Risk budgeting constraints

The portfolio optimization problem:
    maximize    μ^T w - (γ/2) w^T Σ w
    subject to  1^T w = 1              (budget constraint)
                w >= 0                 (long-only, optional)
                w_min <= w <= w_max    (position limits, optional)

Where:
- w: portfolio weights
- μ: expected returns
- Σ: covariance matrix
- γ: risk aversion parameter
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple

import cvxjax as cx


def generate_market_data(n_assets: int = 10, n_periods: int = 252, seed: int = 42) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate synthetic market data for portfolio optimization.
    
    Args:
        n_assets: Number of assets.
        n_periods: Number of time periods for return history.
        seed: Random seed.
        
    Returns:
        Tuple of (returns, expected_returns, covariance_matrix).
    """
    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)
    
    # Generate factor loadings for realistic correlation structure
    n_factors = min(3, n_assets // 2)
    factor_loadings = jax.random.normal(key1, (n_assets, n_factors)) * 0.3
    
    # Generate factor returns
    factor_returns = jax.random.normal(key2, (n_periods, n_factors)) * 0.02
    
    # Generate idiosyncratic returns
    idiosyncratic_vol = 0.01 + 0.02 * jax.random.uniform(key3, (n_assets,))
    idiosyncratic_returns = jax.random.normal(key3, (n_periods, n_assets)) * idiosyncratic_vol
    
    # Combine systematic and idiosyncratic returns
    systematic_returns = factor_returns @ factor_loadings.T
    returns = systematic_returns + idiosyncratic_returns
    
    # Add asset-specific drift
    asset_drifts = 0.05 + 0.10 * jax.random.uniform(key1, (n_assets,)) / 252  # Annualized to daily
    returns = returns + asset_drifts
    
    # Compute sample statistics
    expected_returns = jnp.mean(returns, axis=0)
    covariance_matrix = jnp.cov(returns.T)
    
    return returns, expected_returns, covariance_matrix


def solve_portfolio_basic(expected_returns: jnp.ndarray, 
                         covariance_matrix: jnp.ndarray,
                         risk_aversion: float = 1.0) -> cx.Solution:
    """Solve basic mean-variance portfolio optimization.
    
    Args:
        expected_returns: Expected returns vector (n_assets,).
        covariance_matrix: Covariance matrix (n_assets, n_assets).
        risk_aversion: Risk aversion parameter (higher = more conservative).
        
    Returns:
        Solution object.
    """
    n_assets = len(expected_returns)
    
    # Define portfolio weights variable
    w = cx.Variable(shape=(n_assets,), name="weights")
    
    # Objective: maximize expected return - risk penalty
    # Equivalently: minimize -expected_return + risk_penalty
    expected_return = expected_returns @ w
    risk_penalty = 0.5 * risk_aversion * cx.quad_form(w, covariance_matrix)
    objective = cx.Minimize(-expected_return + risk_penalty)
    
    # Budget constraint: weights sum to 1
    constraints = [cx.sum(w) == 1]
    
    # Create and solve problem
    problem = cx.Problem(objective, constraints)
    solution = problem.solve(solver="ipm", tol=1e-8)
    
    return solution, w


def solve_portfolio_long_only(expected_returns: jnp.ndarray,
                             covariance_matrix: jnp.ndarray,
                             risk_aversion: float = 1.0) -> cx.Solution:
    """Solve long-only mean-variance portfolio optimization.
    
    Args:
        expected_returns: Expected returns vector.
        covariance_matrix: Covariance matrix.
        risk_aversion: Risk aversion parameter.
        
    Returns:
        Solution object.
    """
    n_assets = len(expected_returns)
    
    # Define portfolio weights variable
    w = cx.Variable(shape=(n_assets,), name="weights")
    
    # Objective: maximize expected return - risk penalty
    expected_return = expected_returns @ w
    risk_penalty = 0.5 * risk_aversion * cx.quad_form(w, covariance_matrix)
    objective = cx.Minimize(-expected_return + risk_penalty)
    
    # Constraints
    constraints = [
        cx.sum(w) == 1,    # Budget constraint
        w >= 0             # Long-only constraint
    ]
    
    # Create and solve problem
    problem = cx.Problem(objective, constraints)
    solution = problem.solve(solver="ipm", tol=1e-8)
    
    return solution, w


def solve_portfolio_box_constraints(expected_returns: jnp.ndarray,
                                   covariance_matrix: jnp.ndarray,
                                   w_min: jnp.ndarray,
                                   w_max: jnp.ndarray,
                                   risk_aversion: float = 1.0) -> cx.Solution:
    """Solve portfolio optimization with box constraints.
    
    Args:
        expected_returns: Expected returns vector.
        covariance_matrix: Covariance matrix.
        w_min: Minimum weights (n_assets,).
        w_max: Maximum weights (n_assets,).
        risk_aversion: Risk aversion parameter.
        
    Returns:
        Solution object.
    """
    n_assets = len(expected_returns)
    
    # Define portfolio weights variable
    w = cx.Variable(shape=(n_assets,), name="weights")
    
    # Objective: maximize expected return - risk penalty
    expected_return = expected_returns @ w
    risk_penalty = 0.5 * risk_aversion * cx.quad_form(w, covariance_matrix)
    objective = cx.Minimize(-expected_return + risk_penalty)
    
    # Constraints
    constraints = [
        cx.sum(w) == 1,                    # Budget constraint
        w >= w_min,                        # Lower bounds
        w <= w_max                         # Upper bounds
    ]
    
    # Create and solve problem
    problem = cx.Problem(objective, constraints)
    solution = problem.solve(solver="ipm", tol=1e-8)
    
    return solution, w


def solve_portfolio_transaction_costs(expected_returns: jnp.ndarray,
                                    covariance_matrix: jnp.ndarray,
                                    current_weights: jnp.ndarray,
                                    transaction_cost: float = 0.001,
                                    risk_aversion: float = 1.0) -> cx.Solution:
    """Solve portfolio optimization with transaction costs.
    
    Args:
        expected_returns: Expected returns vector.
        covariance_matrix: Covariance matrix.
        current_weights: Current portfolio weights.
        transaction_cost: Transaction cost rate.
        risk_aversion: Risk aversion parameter.
        
    Returns:
        Solution object.
    """
    n_assets = len(expected_returns)
    
    # Define variables
    w = cx.Variable(shape=(n_assets,), name="weights")
    t_plus = cx.Variable(shape=(n_assets,), name="trades_plus")
    t_minus = cx.Variable(shape=(n_assets,), name="trades_minus")
    
    # Objective: expected return - risk penalty - transaction costs
    expected_return = expected_returns @ w
    risk_penalty = 0.5 * risk_aversion * cx.quad_form(w, covariance_matrix)
    transaction_costs = transaction_cost * cx.sum(t_plus + t_minus)
    objective = cx.Minimize(-expected_return + risk_penalty + transaction_costs)
    
    # Constraints
    constraints = [
        cx.sum(w) == 1,                           # Budget constraint
        w == current_weights + t_plus - t_minus,  # Trade decomposition
        t_plus >= 0,                              # Positive trades
        t_minus >= 0,                             # Negative trades
        w >= 0                                    # Long-only (optional)
    ]
    
    # Create and solve problem
    problem = cx.Problem(objective, constraints)
    solution = problem.solve(solver="ipm", tol=1e-8)
    
    return solution, w, t_plus, t_minus


def compute_portfolio_stats(weights: jnp.ndarray,
                          expected_returns: jnp.ndarray,
                          covariance_matrix: jnp.ndarray) -> dict:
    """Compute portfolio statistics.
    
    Args:
        weights: Portfolio weights.
        expected_returns: Expected returns vector.
        covariance_matrix: Covariance matrix.
        
    Returns:
        Dictionary with portfolio statistics.
    """
    expected_return = jnp.sum(weights * expected_returns)
    portfolio_variance = weights.T @ covariance_matrix @ weights
    portfolio_vol = jnp.sqrt(portfolio_variance)
    sharpe_ratio = expected_return / portfolio_vol if portfolio_vol > 0 else 0.0
    
    return {
        'expected_return': float(expected_return),
        'volatility': float(portfolio_vol),
        'sharpe_ratio': float(sharpe_ratio),
        'max_weight': float(jnp.max(weights)),
        'min_weight': float(jnp.min(weights)),
        'n_positions': int(jnp.sum(jnp.abs(weights) > 1e-4))
    }


def main():
    """Run portfolio optimization examples."""
    # Enable 64-bit precision
    jax.config.update("jax_enable_x64", True)
    
    print("Portfolio Optimization with CVXJAX")
    print("=" * 40)
    
    # Generate market data
    n_assets = 8
    returns, expected_returns, covariance_matrix = generate_market_data(
        n_assets=n_assets, n_periods=252, seed=42
    )
    
    print(f"Generated data for {n_assets} assets with 252 periods")
    print(f"Expected returns range: [{jnp.min(expected_returns):.4f}, {jnp.max(expected_returns):.4f}]")
    print(f"Expected returns (annualized %): {expected_returns * 252 * 100}")
    print(f"Volatilities (annualized %): {jnp.sqrt(jnp.diag(covariance_matrix)) * jnp.sqrt(252) * 100}")
    print()
    
    # 1. Basic mean-variance optimization
    print("1. Basic Mean-Variance Optimization")
    print("-" * 35)
    
    risk_aversions = [0.5, 1.0, 2.0, 5.0]
    
    for gamma in risk_aversions:
        solution, w = solve_portfolio_basic(expected_returns, covariance_matrix, risk_aversion=gamma)
        
        if solution.status == "optimal":
            weights = solution.primal[w]
            stats = compute_portfolio_stats(weights, expected_returns, covariance_matrix)
            
            print(f"Risk aversion γ = {gamma}:")
            print(f"  Expected return: {stats['expected_return']:.4f}")
            print(f"  Volatility: {stats['volatility']:.4f}")
            print(f"  Sharpe ratio: {stats['sharpe_ratio']:.4f}")
            print(f"  Weight range: [{stats['min_weight']:.3f}, {stats['max_weight']:.3f}]")
            print(f"  Positions: {stats['n_positions']}/{n_assets}")
        else:
            print(f"Risk aversion γ = {gamma}: Failed to solve ({solution.status})")
    
    print()
    
    # 2. Long-only optimization
    print("2. Long-Only Portfolio Optimization")
    print("-" * 35)
    
    for gamma in [1.0, 2.0]:
        solution_basic, w_basic = solve_portfolio_basic(expected_returns, covariance_matrix, risk_aversion=gamma)
        solution_long_only, w_long_only = solve_portfolio_long_only(expected_returns, covariance_matrix, risk_aversion=gamma)
        
        if solution_basic.status == "optimal" and solution_long_only.status == "optimal":
            weights_basic = solution_basic.primal[w_basic]
            weights_long_only = solution_long_only.primal[w_long_only]
            
            stats_basic = compute_portfolio_stats(weights_basic, expected_returns, covariance_matrix)
            stats_long_only = compute_portfolio_stats(weights_long_only, expected_returns, covariance_matrix)
            
            print(f"Risk aversion γ = {gamma}:")
            print(f"  Unconstrained: SR = {stats_basic['sharpe_ratio']:.4f}, "
                  f"min weight = {stats_basic['min_weight']:.3f}")
            print(f"  Long-only:     SR = {stats_long_only['sharpe_ratio']:.4f}, "
                  f"min weight = {stats_long_only['min_weight']:.3f}")
    
    print()
    
    # 3. Box constraints
    print("3. Portfolio with Position Limits")
    print("-" * 32)
    
    # Set position limits: 0% to 30% per asset
    w_min = jnp.zeros(n_assets)
    w_max = jnp.full(n_assets, 0.30)
    
    solution_box, w_box = solve_portfolio_box_constraints(
        expected_returns, covariance_matrix, w_min, w_max, risk_aversion=1.0
    )
    
    if solution_box.status == "optimal":
        weights_box = solution_box.primal[w_box]
        stats_box = compute_portfolio_stats(weights_box, expected_returns, covariance_matrix)
        
        print("Position limits [0%, 30%]:")
        print(f"  Expected return: {stats_box['expected_return']:.4f}")
        print(f"  Volatility: {stats_box['volatility']:.4f}")
        print(f"  Sharpe ratio: {stats_box['sharpe_ratio']:.4f}")
        print(f"  Max weight: {stats_box['max_weight']:.3f}")
        print(f"  Portfolio weights: {weights_box}")
        
        # Check if any constraints are active
        active_upper = jnp.sum(jnp.abs(weights_box - w_max) < 1e-4)
        print(f"  Active upper bounds: {active_upper}/{n_assets}")
    
    print()
    
    # 4. Transaction costs
    print("4. Portfolio Rebalancing with Transaction Costs")
    print("-" * 45)
    
    # Assume current portfolio is equally weighted
    current_weights = jnp.ones(n_assets) / n_assets
    transaction_costs = [0.0, 0.001, 0.005, 0.01]  # 0%, 0.1%, 0.5%, 1%
    
    for tc in transaction_costs:
        solution_tc, w_tc, t_plus_tc, t_minus_tc = solve_portfolio_transaction_costs(
            expected_returns, covariance_matrix, current_weights,
            transaction_cost=tc, risk_aversion=1.0
        )
        
        if solution_tc.status == "optimal":
            weights_tc = solution_tc.primal[w_tc]
            stats_tc = compute_portfolio_stats(weights_tc, expected_returns, covariance_matrix)
            
            # Compute total trades
            trades_plus = solution_tc.primal[t_plus_tc]
            trades_minus = solution_tc.primal[t_minus_tc]
            total_trades = jnp.sum(trades_plus + trades_minus)
            
            print(f"Transaction cost {tc*100:.1f}%:")
            print(f"  Sharpe ratio: {stats_tc['sharpe_ratio']:.4f}")
            print(f"  Total turnover: {total_trades:.4f}")
            print(f"  Max weight change: {jnp.max(jnp.abs(weights_tc - current_weights)):.4f}")
    
    print()
    
    # 5. Efficient frontier computation
    print("5. Efficient Frontier")
    print("-" * 18)
    
    # Compute efficient frontier by varying risk aversion
    gammas = jnp.logspace(-1, 1.5, 10)  # From 0.1 to ~31.6
    
    frontier_returns = []
    frontier_vols = []
    frontier_sharpes = []
    
    for gamma in gammas:
        solution, w = solve_portfolio_long_only(expected_returns, covariance_matrix, risk_aversion=gamma)
        
        if solution.status == "optimal":
            weights = solution.primal[w]
            stats = compute_portfolio_stats(weights, expected_returns, covariance_matrix)
            
            frontier_returns.append(stats['expected_return'])
            frontier_vols.append(stats['volatility'])
            frontier_sharpes.append(stats['sharpe_ratio'])
    
    if frontier_returns:
        frontier_returns = jnp.array(frontier_returns)
        frontier_vols = jnp.array(frontier_vols)
        frontier_sharpes = jnp.array(frontier_sharpes)
        
        # Find maximum Sharpe ratio portfolio
        max_sharpe_idx = jnp.argmax(frontier_sharpes)
        
        print(f"Efficient frontier: {len(frontier_returns)} points computed")
        print(f"Return range: [{jnp.min(frontier_returns):.4f}, {jnp.max(frontier_returns):.4f}]")
        print(f"Volatility range: [{jnp.min(frontier_vols):.4f}, {jnp.max(frontier_vols):.4f}]")
        print(f"Maximum Sharpe ratio: {frontier_sharpes[max_sharpe_idx]:.4f}")
        print(f"  at return = {frontier_returns[max_sharpe_idx]:.4f}, "
              f"vol = {frontier_vols[max_sharpe_idx]:.4f}")
    
    print()
    
    # Optional: Create visualization
    try:
        create_portfolio_plots(frontier_returns, frontier_vols, frontier_sharpes)
    except Exception:
        print("Plotting skipped (matplotlib may not be available)")
    
    print("Portfolio optimization examples completed!")


def create_portfolio_plots(returns, vols, sharpes):
    """Create portfolio optimization visualizations."""
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Efficient frontier
        ax1.plot(vols, returns, 'bo-', label='Efficient Frontier')
        ax1.set_xlabel('Portfolio Volatility')
        ax1.set_ylabel('Expected Return')
        ax1.set_title('Mean-Variance Efficient Frontier')
        ax1.grid(True)
        ax1.legend()
        
        # Plot 2: Sharpe ratio vs volatility
        ax2.plot(vols, sharpes, 'ro-', label='Sharpe Ratio')
        ax2.set_xlabel('Portfolio Volatility')
        ax2.set_ylabel('Sharpe Ratio')
        ax2.set_title('Sharpe Ratio vs Risk')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('portfolio_results.png', dpi=150, bbox_inches='tight')
        print("Plots saved to 'portfolio_results.png'")
        
    except ImportError:
        pass


if __name__ == "__main__":
    main()
