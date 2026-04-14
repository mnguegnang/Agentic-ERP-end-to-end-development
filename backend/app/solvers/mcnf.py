"""MCNF solver — Min-Cost Network Flow via OR-Tools GLOP (Blueprint §4.3.1).

Stage 3 implementation.

Formulation (LP):
    Variables  : x[k][a] ≥ 0  — flow of commodity k on arc a
    Minimise   : Σ_{k,a}  cost_per_unit_a · x[k][a]
    Subject to :
        ∀ a     : Σ_k x[k][a]                        ≤ capacity_a   (capacity)
        ∀ k, n  : Σ_{a:tail=n} x[k][a]
                − Σ_{a:head=n} x[k][a]               = b_k(n)       (flow conservation)
    where b_k(n) = demand_k if n==source_k, −demand_k if n==sink_k, else 0.
"""

from __future__ import annotations

from ortools.linear_solver import pywraplp

_STATUS_MAP: dict[int, str] = {
    pywraplp.Solver.OPTIMAL: "OPTIMAL",
    pywraplp.Solver.FEASIBLE: "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.UNBOUNDED: "UNBOUNDED",
    pywraplp.Solver.ABNORMAL: "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


def solve_mcnf(
    nodes: list[str],
    arcs: list[dict],
    commodities: list[dict],
) -> dict:
    """Solve multi-commodity min-cost network flow via OR-Tools GLOP.

    Args:
        nodes:       Unique node IDs.
        arcs:        [{from, to, capacity, cost_per_unit}, ...]
        commodities: [{source, sink, demand}, ...]

    Returns:
        {
          status        : "OPTIMAL" | "FEASIBLE" | "INFEASIBLE" | ...
          total_cost    : float
          flows         : [{commodity, from, to, flow}, ...]   (positive flow only)
          shadow_prices : [{from, to, dual}, ...]              (one per arc)
        }
    """
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if solver is None:
        return {
            "status": "SOLVER_UNAVAILABLE",
            "total_cost": 0.0,
            "flows": [],
            "shadow_prices": [],
        }

    n_commodities = len(commodities)
    n_arcs = len(arcs)
    n_nodes = len(nodes)
    node_idx: dict[str, int] = {n: i for i, n in enumerate(nodes)}

    # ------------------------------------------------------------------
    # Decision variables  x[k][a] ∈ [0, +∞)
    # ------------------------------------------------------------------
    x: list[list] = [
        [solver.NumVar(0.0, solver.infinity(), f"x_{k}_{a}") for a in range(n_arcs)]
        for k in range(n_commodities)
    ]

    # ------------------------------------------------------------------
    # Objective: minimise Σ cost * x
    # ------------------------------------------------------------------
    obj = solver.Objective()
    for k in range(n_commodities):
        for a, arc in enumerate(arcs):
            obj.SetCoefficient(x[k][a], float(arc["cost_per_unit"]))
    obj.SetMinimization()

    # ------------------------------------------------------------------
    # Capacity constraints: Σ_k x[k][a] ≤ capacity_a
    # Stored so we can read dual values after solve.
    # ------------------------------------------------------------------
    cap_ctrs: list = []
    for a, arc in enumerate(arcs):
        ct = solver.Constraint(0.0, float(arc["capacity"]), f"cap_{a}")
        for k in range(n_commodities):
            ct.SetCoefficient(x[k][a], 1.0)
        cap_ctrs.append(ct)

    # ------------------------------------------------------------------
    # Flow-conservation constraints (one per commodity × node)
    # ------------------------------------------------------------------
    for k, com in enumerate(commodities):
        src_idx = node_idx[com["source"]]
        snk_idx = node_idx[com["sink"]]
        demand = float(com["demand"])

        for n in range(n_nodes):
            if n == src_idx:
                b = demand
            elif n == snk_idx:
                b = -demand
            else:
                b = 0.0
            ct = solver.Constraint(b, b, f"flow_{k}_{n}")
            for a, arc in enumerate(arcs):
                if node_idx[arc["from"]] == n:
                    ct.SetCoefficient(x[k][a], 1.0)  # outflow from n
                if node_idx[arc["to"]] == n:
                    ct.SetCoefficient(x[k][a], -1.0)  # inflow to n

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    status = solver.Solve()
    status_str = _STATUS_MAP.get(status, "UNKNOWN")

    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return {
            "status": status_str,
            "total_cost": 0.0,
            "flows": [],
            "shadow_prices": [],
        }

    # ------------------------------------------------------------------
    # Extract primal solution (omit arcs with negligible flow)
    # ------------------------------------------------------------------
    flows = []
    for k in range(n_commodities):
        for a, arc in enumerate(arcs):
            flow_val = x[k][a].solution_value()
            if flow_val > 1e-6:
                flows.append(
                    {
                        "commodity": k,
                        "from": arc["from"],
                        "to": arc["to"],
                        "flow": round(flow_val, 6),
                    }
                )

    # ------------------------------------------------------------------
    # Extract dual solution (shadow prices on capacity constraints)
    # GLOP exposes constraint.dual_value() for LP relaxations.
    # ------------------------------------------------------------------
    shadow_prices = [
        {
            "from": arc["from"],
            "to": arc["to"],
            "dual": round(ct.dual_value(), 6),
        }
        for arc, ct in zip(arcs, cap_ctrs)
    ]

    return {
        "status": status_str,
        "total_cost": round(solver.Objective().Value(), 6),
        "flows": flows,
        "shadow_prices": shadow_prices,
    }
