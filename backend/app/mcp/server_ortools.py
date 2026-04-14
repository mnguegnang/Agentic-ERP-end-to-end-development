"""MCP server: mcp-solver-ortools (Blueprint §4.3.1).

Exposes solve_mcnf, solve_jsp, solve_vrp, solve_disruption.
Stage 4 implementation.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.solvers.disruption import solve_disruption
from app.solvers.jsp import solve_jsp
from app.solvers.mcnf import solve_mcnf
from app.solvers.vrp import solve_vrp

mcp = FastMCP("mcp-solver-ortools")


@mcp.tool()
async def tool_solve_mcnf(
    nodes: list[str], arcs: list[dict], commodities: list[dict]
) -> dict:
    """Min-cost network flow via OR-Tools GLOP."""
    return solve_mcnf(nodes, arcs, commodities)


@mcp.tool()
async def tool_solve_jsp(jobs: list[dict], time_limit_sec: float = 30.0) -> dict:
    """Job-shop scheduling via CP-SAT."""
    return solve_jsp(jobs, time_limit_sec)


@mcp.tool()
async def tool_solve_vrp(
    depot: int, locations: list[dict], vehicle_capacity: int, num_vehicles: int
) -> dict:
    """Capacitated VRP via OR-Tools Routing."""
    return solve_vrp(depot, locations, vehicle_capacity, num_vehicles)


@mcp.tool()
async def tool_solve_disruption(
    affected_components: list[str], alt_suppliers: list[dict], demands: list[dict]
) -> dict:
    """Supply disruption re-allocation via CP-SAT MIP."""
    return solve_disruption(affected_components, alt_suppliers, demands)
