"""MCP server: mcp-solver-scipy (Blueprint §4.3.3).

Exposes analyze_bullwhip.
Stage 4 implementation.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.solvers.bullwhip import analyze_bullwhip

mcp = FastMCP("mcp-solver-scipy")


@mcp.tool()
async def tool_analyze_bullwhip(
    demand_series: list[float],
    lead_time: int,
    forecast_window: int,
    num_echelons: int,
) -> dict:
    """Bullwhip effect analysis: AR(1) rho, amplification ratios, spectral radius."""
    return analyze_bullwhip(demand_series, lead_time, forecast_window, num_echelons)
