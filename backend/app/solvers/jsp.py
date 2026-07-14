"""Job-Shop Scheduling via OR-Tools CP-SAT (Blueprint §4.3.1).

Stage 4 implementation.

Formulation (MIP / CP-SAT):
    Variables:
        start[j][o]    ∈ [0, horizon]  — start time of operation o of job j
        end[j][o]      ∈ [0, horizon]  — end time (= start + duration; enforced by IntervalVar)
        interval[j][o]                 — IntervalVar that owns the [start, end) window
        makespan       ∈ [0, horizon]  — max end time across all jobs

    Minimise: makespan

    Constraints:
        ∀ machine m   : AddNoOverlap({interval[j][o] : op[o].machine == m})
        ∀ job j, o<last: start[j][o+1] ≥ end[j][o]   (precedence / sequencing)
        makespan       ≥ end[j][last_op]  for every job j
"""

from __future__ import annotations

from ortools.sat.python import cp_model

_STATUS_MAP: dict[int, str] = {
    cp_model.OPTIMAL: "OPTIMAL",
    cp_model.FEASIBLE: "FEASIBLE",
    cp_model.INFEASIBLE: "INFEASIBLE",
    cp_model.UNKNOWN: "UNKNOWN",
    cp_model.MODEL_INVALID: "MODEL_INVALID",
}


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
    if not jobs or all(not job.get("operations") for job in jobs):
        return {"status": "OPTIMAL", "makespan": 0, "schedule": []}

    model = cp_model.CpModel()

    # Horizon: loose upper bound = sum of all durations
    horizon: int = sum(int(op["duration"]) for job in jobs for op in job.get("operations", []))

    # (job_idx, op_idx) → (start_var, end_var, interval_var)
    all_tasks: dict[tuple[int, int], tuple] = {}
    # machine_id → list of interval variables assigned to that machine
    machine_intervals: dict[int, list] = {}

    for j_idx, job in enumerate(jobs):
        for o_idx, op in enumerate(job.get("operations", [])):
            machine = int(op["machine"])
            duration = int(op["duration"])
            suffix = f"_{j_idx}_{o_idx}"
            start_var = model.NewIntVar(0, horizon, f"start{suffix}")
            end_var = model.NewIntVar(0, horizon, f"end{suffix}")
            interval_var = model.NewIntervalVar(start_var, duration, end_var, f"interval{suffix}")
            all_tasks[(j_idx, o_idx)] = (start_var, end_var, interval_var)
            machine_intervals.setdefault(machine, []).append(interval_var)

    # No-overlap constraint: no two operations on the same machine overlap
    for intervals in machine_intervals.values():
        model.AddNoOverlap(intervals)

    # Precedence constraint: each operation starts only after the previous one ends
    for j_idx, job in enumerate(jobs):
        ops = job.get("operations", [])
        for o_idx in range(len(ops) - 1):
            model.Add(all_tasks[(j_idx, o_idx + 1)][0] >= all_tasks[(j_idx, o_idx)][1])

    # Makespan = max end time over all last operations
    makespan_var = model.NewIntVar(0, horizon, "makespan")
    last_ends = [
        all_tasks[(j_idx, len(job["operations"]) - 1)][1]
        for j_idx, job in enumerate(jobs)
        if job.get("operations")
    ]
    model.AddMaxEquality(makespan_var, last_ends)
    model.Minimize(makespan_var)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    status = solver.Solve(model)
    # solver.Solve() returns CpSolverStatus; the stubs don't treat it as int,
    # but at runtime it is the same int constant used as the map keys.
    status_str = _STATUS_MAP.get(status, "UNKNOWN")  # type: ignore[call-overload]

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": status_str, "makespan": 0, "schedule": []}

    schedule = [
        {
            "job": j_idx,
            "op": o_idx,
            "machine": int(jobs[j_idx]["operations"][o_idx]["machine"]),
            "start": solver.Value(start_var),
            "end": solver.Value(end_var),
        }
        for (j_idx, o_idx), (start_var, end_var, _) in all_tasks.items()
    ]

    return {
        "status": status_str,
        "makespan": int(solver.ObjectiveValue()),
        "schedule": schedule,
    }
