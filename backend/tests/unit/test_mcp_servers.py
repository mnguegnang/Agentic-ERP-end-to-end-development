"""Unit tests — MCP tool servers (Blueprint §4.3).

The MCP servers are the EXTERNAL tool interface (stdio; for Claude Code,
MCP inspector, or other agents).  The in-process orchestrator calls the
solver functions directly.  These tests pin two guarantees:

  1. Each server registers exactly the advertised tools.
  2. Solver tools delegate to the same deterministic functions the
     orchestrator uses — identical inputs give identical outputs, so tool
     invocation precision reduces to intent-routing accuracy.
"""

from __future__ import annotations

import pytest
from app.mcp import (
    server_crag,
    server_cvxpy,
    server_erp,
    server_kg,
    server_ortools,
    server_scipy,
)

_EXPECTED_TOOLS = {
    "mcp-solver-ortools": (
        server_ortools,
        {"tool_solve_mcnf", "tool_solve_jsp", "tool_solve_vrp", "tool_solve_disruption"},
    ),
    "mcp-solver-cvxpy": (
        server_cvxpy,
        {"tool_solve_robust_minmax", "tool_solve_meio_gsm"},
    ),
    "mcp-solver-scipy": (server_scipy, {"tool_analyze_bullwhip"}),
    "mcp-contract-rag": (server_crag, {"search_contracts"}),
    "mcp-erp-postgres": (server_erp, {"query_erp", "get_product_bom"}),
    "mcp-knowledge-graph": (
        server_kg,
        {"traverse_supply_network", "find_affected_products", "get_supplier_alternatives"},
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("server_name", sorted(_EXPECTED_TOOLS))
async def test_server_registers_expected_tools(server_name: str) -> None:
    module, expected = _EXPECTED_TOOLS[server_name]
    assert module.mcp.name == server_name
    registered = {tool.name for tool in await module.mcp.list_tools()}
    assert registered == expected


@pytest.mark.asyncio
async def test_mcnf_tool_matches_direct_solver_call() -> None:
    """MCP tool output is byte-identical to the orchestrator's direct call."""
    from app.solvers.mcnf import solve_mcnf

    nodes = ["A", "B"]
    arcs = [{"from": "A", "to": "B", "capacity": 100.0, "cost_per_unit": 5.0}]
    commodities = [{"source": "A", "sink": "B", "demand": 50.0}]

    via_tool = await server_ortools.tool_solve_mcnf(nodes, arcs, commodities)
    direct = solve_mcnf(nodes, arcs, commodities)

    assert via_tool == direct
    assert via_tool["status"] == "OPTIMAL"
    assert via_tool["total_cost"] == pytest.approx(250.0)


@pytest.mark.asyncio
async def test_bullwhip_tool_matches_direct_solver_call() -> None:
    from app.solvers.bullwhip import analyze_bullwhip

    series = [100.0, 110.0, 95.0, 105.0, 120.0, 90.0, 100.0, 115.0]
    via_tool = await server_scipy.tool_analyze_bullwhip(
        demand_series=series, lead_time=2, forecast_window=3, num_echelons=2
    )
    direct = analyze_bullwhip(series, 2, 3, 2)

    assert via_tool == direct


@pytest.mark.asyncio
async def test_robust_tool_matches_direct_solver_call() -> None:
    from app.solvers.robust_minmax import solve_robust_minmax

    suppliers = [
        {"cost_mean": 10.0, "cost_uncertainty": 1.0, "capacity": 300.0},
        {"cost_mean": 12.0, "cost_uncertainty": 0.5, "capacity": 300.0},
    ]
    via_tool = await server_cvxpy.tool_solve_robust_minmax(
        suppliers=suppliers, demand=400.0, omega=1.0
    )
    direct = solve_robust_minmax(suppliers, 400.0, 1.0)

    assert via_tool["status"] == direct["status"]
    assert via_tool["total_cost"] == pytest.approx(direct["total_cost"])
