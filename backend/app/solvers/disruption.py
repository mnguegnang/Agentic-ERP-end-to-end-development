"""Supply disruption resource allocation via OR-Tools CP-SAT MIP (Blueprint §4.3.1).

Stage 4 implementation.
"""

from __future__ import annotations

from ortools.sat.python import cp_model


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
    # TODO Stage 4: CP-SAT MIP with capacity + demand constraints
    model = cp_model.CpModel()  # noqa: F841
    return {"status": "NOT_IMPLEMENTED", "total_cost": 0, "allocations": []}
