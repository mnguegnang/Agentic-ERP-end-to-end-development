"""Supply disruption resource allocation via OR-Tools CP-SAT MIP (Blueprint §4.3.1).

Stage 4 implementation.

Formulation (MIP):
    Variables:
        x[i] ∈ {0, 1, ..., capacity[i]} — integer units allocated from alt_supplier i.

    Minimise: Σ_i (cost[i] × x[i])

    Subject to:
        ∀ component c ∈ affected_components:
            Σ_{i : supplier[i].component == c} x[i]  ≥  demand[c]   (demand coverage)
        ∀ i:  x[i] ≤ supplier[i].capacity                            (capacity)
        ∀ i:  x[i] ≥ 0, integer

    Integer costs are obtained by scaling floats by _COST_SCALE and rounding;
    reported total_cost uses the original float costs.
"""

from __future__ import annotations

import math

from ortools.sat.python import cp_model

_COST_SCALE: int = 1_000  # float cost → integer for CP-SAT objective

_STATUS_MAP: dict[int, str] = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.UNKNOWN: "UNKNOWN",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
}


def solve_disruption(
    affected_components: list[str],
    alt_suppliers: list[dict],
    demands: list[dict],
) -> dict:
    """Re-allocate components to alternative suppliers minimising total cost.

    Args:
        affected_components: List of component IDs affected by disruption.
        alt_suppliers: [{id, component, cost, capacity}]
        demands: [{component, quantity}]

    Returns:
        {status, total_cost, allocations: [{supplier, component, quantity}]}
    """
    if not affected_components or not alt_suppliers or not demands:
        return {"status": "OPTIMAL", "total_cost": 0.0, "allocations": []}

    model = cp_model.CpModel()

    # Build demand map (only for affected components)
    affected_set = set(affected_components)
    demand_map: dict[str, int] = {
        d["component"]: math.ceil(float(d["quantity"]))
        for d in demands
        if d["component"] in affected_set
    }

    # Keep only suppliers that can serve affected components
    relevant: list[dict] = [
        s for s in alt_suppliers if s["component"] in affected_set
    ]

    if not relevant:
        return {"status": "INFEASIBLE", "total_cost": 0.0, "allocations": []}

    # Decision variables x[i] ∈ [0, capacity[i]]
    x_vars = [
        model.NewIntVar(0, int(math.floor(float(sup["capacity"]))), f"x_{i}")
        for i, sup in enumerate(relevant)
    ]

    # Demand satisfaction constraints
    for comp, qty_needed in demand_map.items():
        comp_vars = [
            x_vars[i] for i, sup in enumerate(relevant) if sup["component"] == comp
        ]
        if not comp_vars:
            # No supplier can cover this component → infeasible by definition
            return {"status": "INFEASIBLE", "total_cost": 0.0, "allocations": []}
        model.Add(sum(comp_vars) >= qty_needed)

    # Objective: minimise integer-scaled cost
    int_costs = [round(float(sup["cost"]) * _COST_SCALE) for sup in relevant]
    model.Minimize(sum(c * x for c, x in zip(int_costs, x_vars)))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    status_str = _STATUS_MAP.get(status, "UNKNOWN")

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": status_str, "total_cost": 0.0, "allocations": []}

    # Extract solution
    allocations = []
    total_cost = 0.0
    for i, sup in enumerate(relevant):
        qty = solver.Value(x_vars[i])
        if qty > 0:
            allocations.append(
                {
                    "supplier": sup["id"],
                    "component": sup["component"],
                    "quantity": qty,
                }
            )
            total_cost += qty * float(sup["cost"])

    return {
        "status": status_str,
        "total_cost": round(total_cost, 6),
        "allocations": allocations,
    }
