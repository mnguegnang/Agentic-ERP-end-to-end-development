"""MEIO Guaranteed Service Model via SciPy SLSQP (Blueprint §4.3.2).

Stage 4 implementation.

Formulation (GSM — Graves & Willems 2000):

    The Guaranteed Service Model (GSM) places strategic safety stocks at each
    stage to hedge against demand variability over each stage's net lead time:

        SS[i] = z · demand_std[i] · sqrt(NLT[i])

    where NLT[i] = lead_time[i] + SI[i] - s[i]  is the net lead time,
    SI[i] = max_{j in predecessors[i]} s[j]  is the incoming service time, and
    s[i] is the outgoing service time committed by stage i.

    Objective:
        minimise sum_i holding_cost[i] · SS[i]
        = minimise sum_i holding_cost[i] · z · demand_std[i] · sqrt(NLT[i])

    Decision variables: outgoing service times s[i] >= 0.

    Constraints:
        NLT[i] >= 0  for all i    (non-negative net lead time)
        s[i]   >= 0  for all i
        s[i]   = 0   for all end/customer-facing stages i

    Note on convexity: the objective is concave in the service times (s[i]
    appears with a negative sign inside sqrt). GSM therefore lies outside the
    CVXPY DCP cone and is solved here with scipy SLSQP, which handles smooth
    non-convex constrained optimisation reliably for practical network sizes.
    Multiple restarts guard against local minima.

    References:
        Graves, S. C., & Willems, S. P. (2000). Optimizing strategic safety stock
        placement in supply chains. Manufacturing & Service Operations Management.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


def _compute_nlt(
    s: np.ndarray,
    predecessors: list[list[int]],
    lead_times: np.ndarray,
    n: int,
) -> np.ndarray:
    """Return net lead time vector for a given service-time vector."""
    NLT = np.zeros(n)
    for i in range(n):
        preds = predecessors[i]
        SI_i = float(max(s[j] for j in preds)) if preds else 0.0
        NLT[i] = float(lead_times[i]) + SI_i - float(s[i])
    return NLT


def solve_meio_gsm(
    stages: list[dict],
    service_level: float,
) -> dict:
    """Optimise safety stocks across multi-echelon inventory network.

    Args:
        stages: [{holding_cost, demand_std, lead_time, predecessors: [int]}]
        service_level: Target fill-rate (0-1).

    Returns:
        {status, total_ss_cost, service_times, safety_stocks}
    """
    if not stages:
        return {
            "status": "OPTIMAL",
            "total_ss_cost": 0.0,
            "service_times": [],
            "safety_stocks": [],
        }

    n = len(stages)
    z = float(norm.ppf(float(service_level)))
    holding = np.array([float(s["holding_cost"]) for s in stages])
    demand_std = np.array([float(s["demand_std"]) for s in stages])
    lead_times = np.array([float(s["lead_time"]) for s in stages])
    predecessors: list[list[int]] = [
        [int(p) for p in stage.get("predecessors", [])] for stage in stages
    ]

    # Customer-facing (end) stages: not a predecessor of any other stage.
    all_preds: set[int] = {p for preds in predecessors for p in preds}
    end_stages: set[int] = {i for i in range(n) if i not in all_preds}

    def objective(s: np.ndarray) -> float:
        NLT = _compute_nlt(s, predecessors, lead_times, n)
        NLT = np.maximum(NLT, 0.0)
        SS = z * demand_std * np.sqrt(NLT)
        return float(holding @ SS)

    # Variable bounds: s_i >= 0; end stages fixed to 0.
    bounds = [(0.0, None)] * n
    for i in end_stages:
        bounds[i] = (0.0, 0.0)

    # Constraints: NLT_i >= 0 for each stage.
    constraints = []
    for i in range(n):
        preds = predecessors[i]
        if preds:
            constraints.append({
                "type": "ineq",
                "fun": lambda s, i=i, preds=preds: (
                    float(lead_times[i])
                    + max(s[j] for j in preds)
                    - s[i]
                ),
            })
        else:
            constraints.append({
                "type": "ineq",
                "fun": lambda s, i=i: float(lead_times[i]) - s[i],
            })

    # Run with multiple random restarts to avoid local minima.
    best_result = None
    rng = np.random.default_rng(42)

    starts = [np.zeros(n)]
    for _ in range(4):
        s0 = rng.uniform(0.0, 1.0, size=n)
        for i in end_stages:
            s0[i] = 0.0
        # Clip to feasible region (rough estimate: s_i <= L_i)
        s0 = np.minimum(s0, lead_times)
        starts.append(s0)

    for s0 in starts:
        res = minimize(
            objective,
            s0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-10, "maxiter": 2000},
        )
        if best_result is None or res.fun < best_result.fun:
            best_result = res

    if best_result is None or best_result.x is None:
        return {
            "status": "INFEASIBLE",
            "total_ss_cost": 0.0,
            "service_times": [],
            "safety_stocks": [],
        }

    s_opt = np.maximum(best_result.x, 0.0)
    for i in end_stages:
        s_opt[i] = 0.0

    NLT_opt = np.maximum(_compute_nlt(s_opt, predecessors, lead_times, n), 0.0)
    SS_opt = z * demand_std * np.sqrt(NLT_opt)
    total_ss_cost = float(holding @ SS_opt)

    status = "OPTIMAL" if best_result.success else "FEASIBLE"

    return {
        "status": status,
        "total_ss_cost": round(total_ss_cost, 6),
        "service_times": [round(float(v), 6) for v in s_opt],
        "safety_stocks": [round(float(v), 6) for v in SS_opt],
    }
