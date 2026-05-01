"""Unit tests — solver dispatch node (Blueprint §4.6).

Tests verify:
  1. Each intent routes to the correct solver (mocked).
  2. mcnf_solve calls _extract_mcnf_params then solve_mcnf.
  3. Unknown / kg_query intent returns no_solver_needed.
  4. Solver failure is caught and wrapped in error dict.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.agents.graph_state import AgentState
from app.agents.orchestrator import solver_dispatch_node

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(intent: str) -> AgentState:
    return AgentState(  # type: ignore[misc]
        messages=[{"role": "user", "content": "optimise the supply network flow"}],
        intent=intent,
        intent_confidence=0.9,
        ddd_context="logistics",
        solver_input=None,
        solver_output=None,
        kg_subgraph=None,
        kg_entities=None,
        kg_relation_path=None,
        rag_documents=None,
        rag_evaluation=None,
        human_approval_required=False,
        final_response=None,
        error=None,
    )


def _patch_dispatch_settings() -> MagicMock:
    """Return a mock settings object with values safe for all dispatch tests."""
    mock_settings = MagicMock()
    mock_settings.human_approval_cost_threshold = 10_000.0
    mock_settings.redis_url = "redis://localhost:6379"
    return mock_settings


# ---------------------------------------------------------------------------
# Non-mcnf solver dispatches — stubs called with empty params
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent,solver_module_path,solver_fn_name",
    [
        ("jsp_schedule", "app.agents.orchestrator.solve_jsp", "solve_jsp"),
        ("vrp_route", "app.agents.orchestrator.solve_vrp", "solve_vrp"),
        (
            "robust_allocate",
            "app.agents.orchestrator.solve_robust_minmax",
            "solve_robust_minmax",
        ),
        ("meio_optimize", "app.agents.orchestrator.solve_meio_gsm", "solve_meio_gsm"),
        (
            "bullwhip_analyze",
            "app.agents.orchestrator.analyze_bullwhip",
            "analyze_bullwhip",
        ),
        (
            "disruption_resource",
            "app.agents.orchestrator.solve_disruption",
            "solve_disruption",
        ),
    ],
)
async def test_solver_dispatch_calls_correct_solver(
    intent: str,
    solver_module_path: str,
    solver_fn_name: str,
) -> None:
    mock_solver = MagicMock(return_value={"status": "OPTIMAL", "objective": 100.0})
    state = _make_state(intent)

    with (
        patch(
            "app.agents.orchestrator.get_settings",
            return_value=_patch_dispatch_settings(),
        ),
        patch(solver_module_path, mock_solver),
    ):
        result = await solver_dispatch_node(state)

        mock_solver.assert_called_once()
        assert result["solver_output"] is not None
        assert result["solver_output"]["status"] == "OPTIMAL"


@pytest.mark.asyncio
async def test_solver_dispatch_kg_query_no_solver() -> None:
    """kg_query intent should not call any solver."""
    state = _make_state("kg_query")
    with patch(
        "app.agents.orchestrator.get_settings",
        return_value=_patch_dispatch_settings(),
    ):
        result = await solver_dispatch_node(state)
    assert result["solver_output"]["status"] == "no_solver_needed"
    assert result["solver_output"]["intent"] == "kg_query"


@pytest.mark.asyncio
async def test_solver_dispatch_mcnf_param_extraction_called() -> None:
    """mcnf_solve: _extract_mcnf_params is called; on success solve_mcnf is invoked."""
    from app.api.schemas import Arc, Commodity, SolveMcnfInput

    mock_params = SolveMcnfInput(
        nodes=["A", "B"],
        arcs=[Arc(**{"from": "A", "to": "B", "capacity": 100.0, "cost_per_unit": 5.0})],
        commodities=[Commodity(source="A", sink="B", demand=50.0)],
    )
    mock_solver = MagicMock(return_value={"status": "OPTIMAL", "total_cost": 250.0})
    state = _make_state("mcnf_solve")

    with (
        patch(
            "app.agents.orchestrator.get_settings",
            return_value=_patch_dispatch_settings(),
        ),
        patch(
            "app.agents.orchestrator._extract_mcnf_params",
            AsyncMock(return_value=mock_params),
        ),
        patch("app.agents.orchestrator.solve_mcnf", mock_solver),
    ):
        result = await solver_dispatch_node(state)

        mock_solver.assert_called_once()
        assert result["solver_output"]["status"] == "OPTIMAL"


@pytest.mark.asyncio
async def test_solver_dispatch_mcnf_param_extraction_failure() -> None:
    """If _extract_mcnf_params returns None, solver_output has param_extraction_failed."""
    state = _make_state("mcnf_solve")

    with (
        patch(
            "app.agents.orchestrator.get_settings",
            return_value=_patch_dispatch_settings(),
        ),
        patch(
            "app.agents.orchestrator._extract_mcnf_params",
            AsyncMock(return_value=None),
        ),
    ):
        result = await solver_dispatch_node(state)

    assert result["solver_output"]["status"] == "param_extraction_failed"


@pytest.mark.asyncio
async def test_solver_dispatch_solver_exception_caught() -> None:
    """Solver runtime error is caught and wrapped in error dict."""
    state = _make_state("bullwhip_analyze")
    mock_solver = MagicMock(side_effect=RuntimeError("OR-Tools crash"))

    with (
        patch(
            "app.agents.orchestrator.get_settings",
            return_value=_patch_dispatch_settings(),
        ),
        patch("app.agents.orchestrator.analyze_bullwhip", mock_solver),
    ):
        result = await solver_dispatch_node(state)

        assert result["solver_output"]["status"] == "error"
        assert "OR-Tools crash" in result["solver_output"]["error"]
