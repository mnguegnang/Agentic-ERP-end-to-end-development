"""Job-Shop Scheduling via OR-Tools CP-SAT (Blueprint §4.3.1).

Stage 4 implementation.
"""
from __future__ import annotations

from ortools.sat.python import cp_model


def solve_jsp(
    jobs: list[dict],
    time_limit_sec: float = 30.0,
) -> dict:
    """Minimise makespan for job-shop scheduling problem.

    Args:
        jobs: [{operations: [{machine, duration}]}]
        time_limit_sec: CP-SAT anytime time limit (returns best feasible on timeout).

    Returns:
        {status, makespan, schedule: [{job, op, machine, start, end}]}
    """
    # TODO Stage 4: CP-SAT no-overlap + precedence constraints
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    return {"status": "NOT_IMPLEMENTED", "makespan": 0, "schedule": []}
