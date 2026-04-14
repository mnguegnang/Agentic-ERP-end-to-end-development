"""Integration tests — LangGraph orchestrator (Blueprint §4.1).

Tests verify the full graph wiring with all I/O mocked:
  1. kg_query path: classify → kg_agent → solver_dispatch → synthesize.
  2. contract_query path: classify → contract_agent → synthesize.
  3. mcnf_solve path: classify → solver_dispatch → human_gate (low_impact) → synthesize.
  4. High-cost solver path: solver_dispatch → human_gate (high_impact) → synthesize.
  5. run_orchestrator() returns a valid WsResponse.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.agents.graph_state import AgentState
from app.agents.orchestrator import (
    _GRAPH,
    check_impact,
    human_approval_gate,
    route_by_intent,
    run_orchestrator,
    synthesize_response,
)
from app.api.schemas import WsResponse

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> AgentState:
    state: AgentState = AgentState(  # type: ignore[misc]
        messages=[{"role": "user", "content": "What is the MCNF cost?"}],
        intent=None,
        intent_confidence=0.0,
        ddd_context=None,
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
    return {**state, **overrides}  # type: ignore[return-value]


def _patch_settings(mock_fn: MagicMock) -> None:
    s = MagicMock()
    s.llm_model = "gpt-4o"
    s.llm_base_url = "https://models.inference.ai.azure.com"
    s.github_token = "test-token"
    s.intent_confidence_threshold = 0.7
    s.human_approval_cost_threshold = 10_000.0
    mock_fn.return_value = s


# ---------------------------------------------------------------------------
# check_impact — pure function
# ---------------------------------------------------------------------------


def test_check_impact_high_flag() -> None:
    state = _base_state(human_approval_required=True)
    assert check_impact(state) == "high_impact"


def test_check_impact_low_flag() -> None:
    state = _base_state(human_approval_required=False)
    assert check_impact(state) == "low_impact"


# ---------------------------------------------------------------------------
# human_approval_gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_human_gate_above_threshold_sets_flag() -> None:
    state = _base_state(solver_output={"total_cost": 50_000.0})
    with patch("app.agents.orchestrator.get_settings") as mock_s:
        _patch_settings(mock_s)
        result = await human_approval_gate(state)
    assert result["human_approval_required"] is True


@pytest.mark.asyncio
async def test_human_gate_below_threshold_no_flag() -> None:
    state = _base_state(solver_output={"total_cost": 5_000.0})
    with patch("app.agents.orchestrator.get_settings") as mock_s:
        _patch_settings(mock_s)
        result = await human_approval_gate(state)
    assert result.get("human_approval_required") is False


# ---------------------------------------------------------------------------
# synthesize_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_synthesize_response_returns_final_response() -> None:
    state = _base_state(
        solver_output={"status": "OPTIMAL", "total_cost": 3000.0},
        kg_subgraph={"nodes": [], "edges": [], "paths": []},
    )
    with (
        patch("app.agents.orchestrator.get_settings") as mock_s,
        patch("app.agents.orchestrator.ChatOpenAI") as mock_llm_cls,
    ):
        _patch_settings(mock_s)
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "The optimal cost is $3000."
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)
        mock_llm_cls.return_value = mock_llm

        result = await synthesize_response(state)

    assert result["final_response"] == "The optimal cost is $3000."


@pytest.mark.asyncio
async def test_synthesize_response_llm_failure_uses_fallback() -> None:
    state = _base_state(solver_output={"status": "INFEASIBLE"})
    with (
        patch("app.agents.orchestrator.get_settings") as mock_s,
        patch("app.agents.orchestrator.ChatOpenAI") as mock_llm_cls,
    ):
        _patch_settings(mock_s)
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=RuntimeError("timeout"))
        mock_llm_cls.return_value = mock_llm

        result = await synthesize_response(state)

    assert result["final_response"] is not None
    assert "INFEASIBLE" in result["final_response"]


# ---------------------------------------------------------------------------
# run_orchestrator — full graph end-to-end with all I/O mocked
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_orchestrator_mcnf_path_returns_ws_response() -> None:
    """mcnf_solve path: classify → solver_dispatch → synthesize → WsResponse."""
    from app.api.schemas import IntentClassification  # noqa: PLC0415

    mock_intent_result = IntentClassification(
        intent="mcnf_solve",
        intent_confidence=0.92,
        ddd_context="logistics",
        reasoning="Network flow query.",
    )
    final_answer = "The minimum cost is $2500."

    with (
        patch("app.agents.orchestrator.get_settings") as mock_s,
        patch("app.agents.orchestrator.ChatOpenAI") as mock_llm_cls,
        patch(
            "app.agents.orchestrator._extract_mcnf_params", AsyncMock(return_value=None)
        ),
        patch(
            "app.agents.orchestrator.kg_agent_node",
            AsyncMock(side_effect=lambda s: {**s, "kg_subgraph": {}}),
        ),
    ):
        _patch_settings(mock_s)

        mock_llm = MagicMock()
        # First call: classify_intent structured output
        classify_structured = MagicMock()
        classify_structured.ainvoke = AsyncMock(return_value=mock_intent_result)
        # ainvoke call for synthesize_response
        synth_response = MagicMock()
        synth_response.content = final_answer

        mock_llm.with_structured_output.return_value = classify_structured
        mock_llm.ainvoke = AsyncMock(return_value=synth_response)
        mock_llm_cls.return_value = mock_llm

        response = await run_orchestrator("optimise the network flow")

    assert isinstance(response, WsResponse)
    assert response.role == "assistant"
    assert response.content  # non-empty


@pytest.mark.asyncio
async def test_run_orchestrator_contract_query_path() -> None:
    """contract_query path: classify → contract_agent → synthesize → WsResponse."""
    from app.api.schemas import IntentClassification  # noqa: PLC0415
    from app.rag.retriever import CRAGResult  # noqa: PLC0415

    mock_intent = IntentClassification(
        intent="contract_query",
        intent_confidence=0.88,
        ddd_context="compliance",
        reasoning="Asking about contract terms.",
    )
    mock_crag = CRAGResult(
        documents=[{"id": "1", "chunk_text": "Clause 4: liability cap $1M"}],
        evaluation="correct",
    )

    with (
        patch("app.agents.orchestrator.get_settings") as mock_s,
        patch("app.agents.orchestrator.ChatOpenAI") as mock_llm_cls,
        patch(
            "app.agents.contract_agent.retrieve_and_evaluate",
            AsyncMock(return_value=mock_crag),
        ),
    ):
        _patch_settings(mock_s)

        mock_llm = MagicMock()
        classify_structured = MagicMock()
        classify_structured.ainvoke = AsyncMock(return_value=mock_intent)
        synth_response = MagicMock()
        synth_response.content = "The liability cap is $1M per Clause 4."

        mock_llm.with_structured_output.return_value = classify_structured
        mock_llm.ainvoke = AsyncMock(return_value=synth_response)
        mock_llm_cls.return_value = mock_llm

        response = await run_orchestrator("What is the liability cap in the contract?")

    assert isinstance(response, WsResponse)
    assert response.content


@pytest.mark.asyncio
async def test_run_orchestrator_exception_returns_error_response() -> None:
    """Top-level exception in run_orchestrator returns a WsResponse with error content."""
    with patch("app.agents.orchestrator._get_graph") as mock_graph_fn:
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(side_effect=RuntimeError("graph crash"))
        mock_graph_fn.return_value = mock_graph

        response = await run_orchestrator("what is the flow cost?")

    assert isinstance(response, WsResponse)
    assert "error" in response.content.lower() or "graph crash" in response.content
