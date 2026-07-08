"""Unit tests — DSPy classifier opt-in fallback contract.

The DSPy path must be a strict opt-in: with no compiled artifact (or no dspy
package) it returns None and the orchestrator's zero-shot baseline runs
unchanged.  These tests pin that contract without requiring dspy.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.agents import dspy_classifier


@pytest.fixture(autouse=True)
def _reset_loader_cache():
    """Each test starts with a cold loader cache."""
    dspy_classifier._LOADED = None
    yield
    dspy_classifier._LOADED = None


@pytest.mark.asyncio
async def test_classify_returns_none_without_artifact(tmp_path: Path) -> None:
    with patch.object(dspy_classifier, "ARTIFACT_PATH", tmp_path / "missing.json"):
        assert await dspy_classifier.classify("route 500 units") is None
        assert dspy_classifier.is_available() is False


@pytest.mark.asyncio
async def test_classify_returns_none_when_dspy_missing(tmp_path: Path) -> None:
    """Artifact present but dspy not importable → clean None, no raise."""
    artifact = tmp_path / "compiled_intent_classifier.json"
    artifact.write_text("{}")
    with (
        patch.object(dspy_classifier, "ARTIFACT_PATH", artifact),
        patch.object(
            dspy_classifier,
            "build_program",
            side_effect=ImportError("No module named 'dspy'"),
        ),
    ):
        assert await dspy_classifier.classify("route 500 units") is None
        assert dspy_classifier.is_available() is False


@pytest.mark.asyncio
async def test_unavailability_is_cached() -> None:
    """After one failed load, subsequent calls short-circuit (no re-import)."""
    with patch.object(dspy_classifier, "ARTIFACT_PATH", Path("/nonexistent/artifact.json")):
        assert await dspy_classifier.classify("q") is None
        # cache records the failure
        assert dspy_classifier._LOADED is False
        assert await dspy_classifier.classify("q") is None


@pytest.mark.asyncio
async def test_llm_classify_intent_falls_back_to_baseline() -> None:
    """orchestrator.llm_classify_intent uses the zero-shot path when DSPy opts out."""
    from app.agents.orchestrator import llm_classify_intent
    from app.api.schemas import IntentClassification

    baseline_result = IntentClassification(
        intent="kg_query",
        intent_confidence=0.9,
        ddd_context="visibility",
        reasoning="baseline",
    )

    with (
        patch(
            "app.agents.dspy_classifier.classify",
            AsyncMock(return_value=None),
        ),
        patch("app.agents.orchestrator.get_settings") as mock_s,
        patch("app.agents.orchestrator.ChatOpenAI") as mock_llm_cls,
    ):
        s = MagicMock()
        s.llm_model = "gpt-4o"
        s.llm_base_url = "https://example.test"
        s.github_token = "test-token"
        mock_s.return_value = s

        structured = MagicMock()
        structured.ainvoke = AsyncMock(return_value=baseline_result)
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value = structured
        mock_llm_cls.return_value = mock_llm

        result = await llm_classify_intent("show the supply network")

    assert result is baseline_result


@pytest.mark.asyncio
async def test_llm_classify_intent_prefers_compiled_result() -> None:
    """When the DSPy path yields a result, the baseline LLM is never touched."""
    from app.agents.orchestrator import llm_classify_intent
    from app.api.schemas import IntentClassification

    compiled_result = IntentClassification(
        intent="mcnf_solve",
        intent_confidence=0.95,
        ddd_context="logistics",
        reasoning="compiled",
    )

    with (
        patch(
            "app.agents.dspy_classifier.classify",
            AsyncMock(return_value=compiled_result),
        ),
        patch("app.agents.orchestrator.ChatOpenAI") as mock_llm_cls,
    ):
        result = await llm_classify_intent("route 500 units from A to B")

    assert result is compiled_result
    mock_llm_cls.assert_not_called()
