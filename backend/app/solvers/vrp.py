"""Vehicle Routing Problem via OR-Tools Routing (Blueprint §4.3.1).

Stage 4 implementation.

Formulation (CVRP):
    Minimise: Σ_v Σ_{(i,j) ∈ route_v} dist(i, j)
    Subject to:
        • Each customer visited exactly once.
        • Each route begins and ends at the depot.
        • Σ_{i ∈ route_v} demand[i] ≤ vehicle_capacity  for every vehicle v.

    Arc costs are Euclidean distances scaled to integers (×_SCALE) so
    OR-Tools can use its integer LP / branch-and-bound internally.
    Reported distances are divided back to float.
"""

from __future__ import annotations

import math

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

_SCALE: int = 10_000  # float → integer precision multiplier


def _euclidean_dist_matrix(locations: list[dict]) -> list[list[int]]:
    """Return integer-scaled Euclidean distance matrix."""
    n = len(locations)
    return [
        [
            int(
                math.hypot(
                    locations[i]["x"] - locations[j]["x"],
                    locations[i]["y"] - locations[j]["y"],
                )
                * _SCALE
            )
            for j in range(n)
        ]
        for i in range(n)
    ]


def solve_vrp(
    depot: int,
    locations: list[dict],
    vehicle_capacity: int,
    num_vehicles: int,
) -> dict:
    """Solve CVRP minimising total Euclidean distance.

    Args:
        depot: Index of the depot in locations list.
        locations: [{id, x, y, demand}]
        vehicle_capacity: Max load per vehicle.
        num_vehicles: Fleet size.

    Returns:
        {status, total_distance, routes: [{vehicle, stops, distance}]}
    """
    if not locations:
        return {"status": "OPTIMAL", "total_distance": 0.0, "routes": []}

    n = len(locations)
    dist_matrix = _euclidean_dist_matrix(locations)

    manager = pywrapcp.RoutingIndexManager(n, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # --- Distance (cost) callback ---
    def distance_callback(from_index: int, to_index: int) -> int:
        return dist_matrix[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # --- Capacity dimension ---
    def demand_callback(from_index: int) -> int:
        node = manager.IndexToNode(from_index)
        return int(locations[node].get("demand", 0))

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,  # no slack
        [int(vehicle_capacity)] * num_vehicles,
        True,  # start cumul at zero
        "Capacity",
    )

    # --- Search parameters ---
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.seconds = 30

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        return {"status": "INFEASIBLE", "total_distance": 0.0, "routes": []}

    # --- Extract routes ---
    routes = []
    total_distance = 0.0

    for v in range(num_vehicles):
        index = routing.Start(v)
        stops: list[int] = []
        route_dist_scaled: int = 0

        while not routing.IsEnd(index):
            stops.append(manager.IndexToNode(index))
            next_index = solution.Value(routing.NextVar(index))
            route_dist_scaled += routing.GetArcCostForVehicle(index, next_index, v)
            index = next_index

        stops.append(manager.IndexToNode(index))  # end depot

        # Only include routes that visit at least one customer node
        if len(stops) > 2:
            route_dist = round(route_dist_scaled / _SCALE, 4)
            routes.append(
                {
                    "vehicle": v,
                    "stops": stops,
                    "distance": route_dist,
                }
            )
            total_distance += route_dist

    return {
        "status": "OPTIMAL",
        "total_distance": round(total_distance, 4),
        "routes": routes,
    }
