"""Robust min-max allocation via CVXPY SOCP (Blueprint §4.3.2).

Stage 4 implementation.
"""

from __future__ import annotations

# cvxpy import deferred until SOCP formulation is implemented (Stage 5).
# Importing now triggers a version-conflict warning because
# cvxpy 1.6.x only recognises ortools < 9.10 for its GLOP backend.


def solve_robust_minmax(
    suppliers: list[dict],
    demand: float,
    omega: float,
) -> dict:
    """Solve robust supplier allocation under cost uncertainty.

    Args:
        suppliers: [{cost_mean, cost_uncertainty, capacity}]
        demand: Total demand to fulfil.
        omega: Robustness parameter (uncertainty budget).

    Returns:
        {status, total_cost, allocations, price_of_robustness}
    """
    # TODO Stage 4: SOCP formulation with ECOS solver
    return {
        "status": "NOT_IMPLEMENTED",
        "total_cost": 0.0,
        "allocations": [],
        "price_of_robustness": 0.0,
    }
