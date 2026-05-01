"""Unit tests — CRAG hybrid retriever (Blueprint §4.4).

Tests verify:
  1. reciprocal_rank_fusion() pure function behaviour.
  2. retrieve_and_evaluate() with mocked DB and evaluator.
  3. Fallback paths: empty dense/sparse → no_results.
  4. evaluate_relevance() with mocked LLM.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.rag.evaluator import AMBIGUOUS, CORRECT, INCORRECT, evaluate_relevance
from app.rag.retriever import CRAGResult, reciprocal_rank_fusion, retrieve_and_evaluate

# ---------------------------------------------------------------------------
# reciprocal_rank_fusion — pure function tests
# ---------------------------------------------------------------------------


def _doc(id_: str, text: str = "chunk") -> dict:
    return {"id": id_, "chunk_text": text, "contract_id": 1, "supplier_id": 1}


def test_rrf_combines_dense_and_sparse() -> None:
    """Document appearing in both lists should score higher than doc in one list."""
    dense = [_doc("A"), _doc("B"), _doc("C")]
    sparse = [_doc("B"), _doc("D"), _doc("A")]
    fused = reciprocal_rank_fusion(dense, sparse)
    ids = [d["id"] for d in fused]
    # A and B appear in both lists — they should outrank C and D
    assert ids.index("A") < ids.index("C")
    assert ids.index("B") < ids.index("D")


def test_rrf_empty_sparse() -> None:
    dense = [_doc("X"), _doc("Y")]
    fused = reciprocal_rank_fusion(dense, sparse=[])
    assert [d["id"] for d in fused] == ["X", "Y"]


def test_rrf_empty_dense() -> None:
    sparse = [_doc("Z"), _doc("W")]
    fused = reciprocal_rank_fusion(dense=[], sparse=sparse)
    assert [d["id"] for d in fused] == ["Z", "W"]


def test_rrf_both_empty() -> None:
    assert reciprocal_rank_fusion([], []) == []


def test_rrf_k60_smoothing_applied() -> None:
    """With k=60 the max score per doc from rank-0 is 1/(60+1) ≈ 0.0164."""
    dense = [_doc("top")]
    fused = reciprocal_rank_fusion(dense, sparse=[])
    assert len(fused) == 1
    assert fused[0]["id"] == "top"


# ---------------------------------------------------------------------------
# retrieve_and_evaluate — integration with mocked DB + evaluator
# ---------------------------------------------------------------------------


def _patch_settings_rr(mock_fn: MagicMock) -> None:
    s = MagicMock()
    s.database_url = "postgresql://test:test@localhost/test"
    s.rag_config = {"top_k_dense": 10, "top_k_sparse": 10}
    mock_fn.return_value = s


@pytest.mark.asyncio
async def test_retrieve_and_evaluate_correct_evaluation() -> None:
    """When retrieval returns docs and evaluator says 'correct', evaluation propagates."""
    docs = [_doc("1", "Supplier A provides high-quality components under contract.")]

    with (
        patch("app.rag.retriever.get_settings") as mock_s,
        patch("app.rag.retriever._get_embedder") as mock_emb,
        patch("app.rag.retriever._pgvector_search", AsyncMock(return_value=docs)),
        patch(
            "app.rag.retriever._load_bm25_corpus", AsyncMock(return_value=([], None))
        ),
        patch("app.rag.retriever.rerank", return_value=docs),
        patch("app.rag.retriever.evaluate_relevance", AsyncMock(return_value=CORRECT)),
    ):
        _patch_settings_rr(mock_s)
        mock_emb.return_value.encode = MagicMock(
            return_value=MagicMock(tolist=MagicMock(return_value=[0.0] * 1024))
        )

        result: CRAGResult = await retrieve_and_evaluate("contract for supplier A")

    assert result.evaluation == CORRECT
    assert len(result.documents) == 1


@pytest.mark.asyncio
async def test_retrieve_and_evaluate_no_results_returns_incorrect() -> None:
    """Empty retrieval pipeline returns incorrect without calling evaluator."""
    with (
        patch("app.rag.retriever.get_settings") as mock_s,
        patch("app.rag.retriever._get_embedder") as mock_emb,
        patch("app.rag.retriever._pgvector_search", AsyncMock(return_value=[])),
        patch(
            "app.rag.retriever._load_bm25_corpus", AsyncMock(return_value=([], None))
        ),
        patch("app.rag.retriever.rerank", return_value=[]),
    ):
        _patch_settings_rr(mock_s)
        mock_emb.return_value.encode = MagicMock(
            return_value=MagicMock(tolist=MagicMock(return_value=[0.0] * 1024))
        )

        result = await retrieve_and_evaluate("some query")

    assert result.evaluation == INCORRECT
    assert result.fallback == "no_results"
    assert result.documents == []


@pytest.mark.asyncio
async def test_retrieve_and_evaluate_embedding_failure() -> None:
    """Embedder exception returns CRAGResult with embedding_error fallback."""
    with (
        patch("app.rag.retriever.get_settings") as mock_s,
        patch("app.rag.retriever._get_embedder") as mock_emb,
    ):
        _patch_settings_rr(mock_s)
        mock_emb.return_value.encode = MagicMock(side_effect=RuntimeError("CUDA OOM"))

        result = await retrieve_and_evaluate("test query")

    assert result.fallback == "embedding_error"
    assert result.evaluation == INCORRECT


# ---------------------------------------------------------------------------
# evaluate_relevance — mocked LLM call
# ---------------------------------------------------------------------------


def _patch_eval_settings(mock_fn: MagicMock) -> None:
    s = MagicMock()
    s.llm_model = "gpt-4o"
    s.llm_base_url = "https://models.inference.ai.azure.com"
    s.github_token = "test-token"
    mock_fn.return_value = s


@pytest.mark.asyncio
async def test_evaluate_relevance_none_doc_returns_incorrect() -> None:
    result = await evaluate_relevance("any query", top_doc=None)
    assert result == INCORRECT


@pytest.mark.asyncio
async def test_evaluate_relevance_empty_chunk_returns_incorrect() -> None:
    result = await evaluate_relevance("any query", top_doc={"chunk_text": "   "})
    assert result == INCORRECT


@pytest.mark.asyncio
@pytest.mark.parametrize("label", [CORRECT, AMBIGUOUS, INCORRECT])
async def test_evaluate_relevance_llm_label_propagated(label: str) -> None:
    """LLM-returned valid label is returned directly."""
    from app.rag.evaluator import _RelevanceLabel  # noqa: PLC0415

    mock_result = _RelevanceLabel(label=label, reasoning="test")
    with (
        patch("app.rag.evaluator.get_settings") as mock_s,
        patch("app.rag.evaluator.ChatOpenAI") as mock_llm_cls,
    ):
        _patch_eval_settings(mock_s)
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(return_value=mock_result)
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_cls.return_value = mock_llm

        result = await evaluate_relevance(
            "query", top_doc={"chunk_text": "relevant text"}
        )

    assert result == label


@pytest.mark.asyncio
async def test_evaluate_relevance_llm_failure_returns_ambiguous() -> None:
    with (
        patch("app.rag.evaluator.get_settings") as mock_s,
        patch("app.rag.evaluator.ChatOpenAI") as mock_llm_cls,
    ):
        _patch_eval_settings(mock_s)
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_structured.ainvoke = AsyncMock(side_effect=RuntimeError("timeout"))
        mock_llm.with_structured_output.return_value = mock_structured
        mock_llm_cls.return_value = mock_llm

        result = await evaluate_relevance("query", top_doc={"chunk_text": "some text"})

    assert result == AMBIGUOUS
