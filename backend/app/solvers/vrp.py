"""Vehicle Routing Problem via OR-Tools Routing (Blueprint §4.3.1).

Stage 4 implementation.
"""
from __future__ import annotations

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve_vrp(
    depot: int,
    locations: list[dict],
    vehicle_capacity: int,
    num_vehicles: int,
) -> dict:
    """Solve CVRP minimising total distance.

    Args:
        depot: Index of the depot in locations list.
        locations: [{id, x, y, demand}]
        vehicle_capacity: Max load per vehicle.
        num_vehicles: Fleet size.

    Returns:
        {status, total_distance, routes: [{vehicle, stops, distance}]}
    """
    # TODO Stage 4: OR-Tools RoutingModel with distance + capacity dimensions
    return {"status": "NOT_IMPLEMENTED", "total_distance": 0, "routes": []}
