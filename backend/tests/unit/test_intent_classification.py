"""Unit tests — intent classification and routing (Blueprint §4.2).

Tests verify:
  1. route_by_intent() pure routing logic for all 10 intents.
  2. classify_intent() LLM path with mocked ChatOpenAI structured output.
  3. Confidence threshold gating (< 0.7 → "unclear").
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.agents.graph_state import AgentState
from app.agents.orchestrator import classify_intent, route_by_intent
from app.api.schemas import IntentClassification

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(intent: str | None, confidence: float = 1.0) -> AgentState:
    return AgentState(  # type: ignore[misc]
        messages=[{"role": "user", "content": "test query"}],
        intent=intent,
        intent_confidence=confidence,
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


def _patch_settings(mock_fn: MagicMock) -> None:
    s = MagicMock()
    s.llm_model = "gpt-4o"
    s.llm_base_url = "https://models.inference.ai.azure.com"
    s.github_token = "test-token"
    s.intent_confidence_threshold = 0.7
    mock_fn.return_value = s


# ---------------------------------------------------------------------------
# route_by_intent — pure function tests (no LLM, no I/O)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent,expected_route",
    [
        ("contract_query", "contract_query"),
        ("mcnf_solve", "or_solve"),
        ("meio_optimize", "or_solve"),
        ("bullwhip_analyze", "or_solve"),
        ("jsp_schedule", "or_solve"),
        ("vrp_route", "or_solve"),
        ("robust_allocate", "or_solve"),
        ("kg_query", "kg_query"),
        ("disruption_resource", "kg_query"),
        ("multi_step", "kg_query"),
        ("unclear", "kg_query"),
        ("unknown_intent", "kg_query"),
        (None, "kg_query"),
    ],
)
def test_route_by_intent_all_intents(intent: str | None, expected_route: str) -> None:
    state = _make_state(intent)
    result = route_by_intent(state)
    assert (
        result == expected_route
    ), f"intent={intent!r} should route to {expected_route!r}, got {result!r}"


def test_route_by_intent_solver_direct_set_is_complete() -> None:
    """All expected solver direct intents route to 'or_solve'."""
    solver_intents = {
        "mcnf_solve",
        "meio_optimize",
        "bullwhip_analyze",
        "jsp_schedule",
        "vrp_route",
        "robust_allocate",
    }
    for intent in solver_intents:
        state = _make_state(intent)
        assert (
            route_by_intent(state) == "or_solve"
        ), f"{intent} should route to or_solve"


# ---------------------------------------------------------------------------
# classify_intent — mocked LLM structured output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_intent_high_confidence() -> None:
    """High-confidence LLM result propagates intent unchanged."""
    mock_result = IntentClassification(
        intent="mcnf_solve",
        intent_confidence=0.95,
        ddd_context="logistics",
        reasoning="The query asks for network flow optimisation.",
    )
    with (
        patch("app.agents.orchestrator.get_settings") as mock_settings,
        patch("app.agents.orchestrator.ChatOpenAI") as mock_llm_cls,
    ):
        _patch_settings(mock_settings)
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_result)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_cls.return_value = mock_llm

        state = _make_state(None, confidence=0.0)
        result = await classify_intent(state)

    assert result["intent"] == "mcnf_solve"
    assert result["intent_confidence"] == 0.95
    assert result["ddd_context"] == "logistics"


@pytest.mark.asyncio
async def test_classify_intent_low_confidence_sets_unclear() -> None:
    """Confidence below threshold (0.7) resets intent to 'unclear'."""
    mock_result = IntentClassification(
        intent="mcnf_solve",
        intent_confidence=0.4,
        ddd_context="logistics",
        reasoning="Uncertain.",
    )
    with (
        patch("app.agents.orchestrator.get_settings") as mock_settings,
        patch("app.agents.orchestrator.ChatOpenAI") as mock_llm_cls,
    ):
        _patch_settings(mock_settings)
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_result)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_cls.return_value = mock_llm

        state = _make_state(None, confidence=0.0)
        result = await classify_intent(state)

    assert result["intent"] == "unclear"


@pytest.mark.asyncio
async def test_classify_intent_llm_failure_returns_unclear() -> None:
    """LLM call failure falls back to intent='unclear', confidence=0.0."""
    with (
        patch("app.agents.orchestrator.get_settings") as mock_settings,
        patch("app.agents.orchestrator.ChatOpenAI") as mock_llm_cls,
    ):
        _patch_settings(mock_settings)
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_cls.return_value = mock_llm

        state = _make_state(None, confidence=0.0)
        result = await classify_intent(state)

    assert result["intent"] == "unclear"
    assert result["intent_confidence"] == 0.0
