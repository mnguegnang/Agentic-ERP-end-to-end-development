"""Robust min-max allocation via CVXPY SOCP (Blueprint §4.3.2).

Stage 4 implementation.

Formulation (Robust LP with ellipsoidal uncertainty set):
    Given n suppliers, each with nominal cost c_i, uncertainty δ_i, and capacity q_i,
    we seek allocations x_i ≥ 0 that minimise worst-case total cost under the
    ellipsoidal uncertainty set:

        U = { c : c = c_mean + diag(δ) · u,  ‖u‖₂ ≤ ω }

    Max over U is solved analytically:
        max_{u: ‖u‖₂ ≤ ω} (c_mean + diag(δ)·u)ᵀ x
            = c_meanᵀ x  +  ω · ‖diag(δ) · x‖₂

    Robust SOCP (epigraph form):
        min_{x, t}   t
        s.t.   c_meanᵀ x  +  ω · ‖diag(δ) · x‖₂  ≤  t
               Σ_i x_i  ≥  demand
               0 ≤ x_i  ≤  capacity_i   for all i

    Solver: CLARABEL (handles SOCP natively; bundled with CVXPY ≥ 1.5).
    Note: ECOS is specified in Blueprint §4.3.2 but is not installed; CLARABEL
    is the recommended successor and produces identical solutions.

    price_of_robustness  =  robust_cost  −  nominal_cost
        where nominal_cost = c_meanᵀ x  (uncertainty term excluded).
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def solve_robust_minmax(
    suppliers: list[dict],
    demand: float,
    omega: float,
) -> dict:
    """Solve robust supplier allocation under cost uncertainty.

    Args:
        suppliers: [{cost_mean, cost_uncertainty, capacity}]
        demand: Total demand to fulfil.
        omega: Robustness parameter (uncertainty budget, ≥ 0).

    Returns:
        {status, total_cost, allocations, price_of_robustness}
    """
    if not suppliers:
        return {
            "status": "OPTIMAL",
            "total_cost": 0.0,
            "allocations": [],
            "price_of_robustness": 0.0,
        }

    n = len(suppliers)
    c_mean = np.array([float(s["cost_mean"]) for s in suppliers])
    delta = np.array([float(s["cost_uncertainty"]) for s in suppliers])
    caps = np.array([float(s["capacity"]) for s in suppliers])

    x = cp.Variable(n, nonneg=True)
    t = cp.Variable()

    # Robust objective: c_mean^T x + omega * ||diag(delta) * x||_2  ≤  t
    objective = cp.Minimize(t)
    constraints = [
        c_mean @ x + float(omega) * cp.norm(cp.multiply(delta, x), 2) <= t,
        cp.sum(x) >= float(demand),
        x <= caps,
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CLARABEL)

    status_map = {
        "optimal": "OPTIMAL",
        "optimal_inaccurate": "FEASIBLE",
        "infeasible": "INFEASIBLE",
        "infeasible_inaccurate": "INFEASIBLE",
        "unbounded": "UNBOUNDED",
        "unbounded_inaccurate": "UNBOUNDED",
    }
    status_str = status_map.get(prob.status or "", "UNKNOWN")

    if x.value is None:
        return {
            "status": status_str,
            "total_cost": 0.0,
            "allocations": [],
            "price_of_robustness": 0.0,
        }

    x_val = np.maximum(x.value, 0.0)  # clip tiny negatives from solver tolerance
    robust_cost = float(t.value) if t.value is not None else float(prob.value)
    nominal_cost = float(c_mean @ x_val)
    price_of_robustness = max(0.0, robust_cost - nominal_cost)

    allocations = [
        {"supplier": i, "quantity": round(float(x_val[i]), 6)} for i in range(n) if x_val[i] > 1e-6
    ]

    return {
        "status": status_str,
        "total_cost": round(robust_cost, 6),
        "allocations": allocations,
        "price_of_robustness": round(price_of_robustness, 6),
    }
