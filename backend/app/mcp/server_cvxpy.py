"""MCP server: mcp-solver-cvxpy (Blueprint §4.3.2).

Exposes solve_robust_minmax, solve_meio_gsm.
Stage 4 implementation.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.solvers.meio_gsm import solve_meio_gsm
from app.solvers.robust_minmax import solve_robust_minmax

mcp = FastMCP("mcp-solver-cvxpy")


@mcp.tool()
async def tool_solve_robust_minmax(
    suppliers: list[dict], demand: float, omega: float
) -> dict:
    """Robust min-max supplier allocation via CVXPY SOCP."""
    return solve_robust_minmax(suppliers, demand, omega)


@mcp.tool()
async def tool_solve_meio_gsm(stages: list[dict], service_level: float) -> dict:
    """MEIO guaranteed service model via CVXPY SCS."""
    return solve_meio_gsm(stages, service_level)
