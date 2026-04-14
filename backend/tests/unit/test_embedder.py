"""Unit tests for backend.app.rag.embedder (Blueprint §2.3).

The SentenceTransformer model is mocked to avoid downloading ~1.3 GB during CI.
Tests verify the embedding interface contract and BGE prefix application.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_model(dim: int = 1024) -> MagicMock:
    """Return a mock SentenceTransformer that always outputs unit-norm vectors."""
    mock = MagicMock()
    mock.encode.side_effect = lambda texts, normalize_embeddings=True: (
        np.ones((len(texts), dim), dtype=np.float32) / np.sqrt(dim)
        if isinstance(texts, list)
        else np.ones(dim, dtype=np.float32) / np.sqrt(dim)
    )
    return mock


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("app.rag.embedder._model", None)  # ensure model is not pre-loaded
@patch("app.rag.embedder.SentenceTransformer")
def test_embed_returns_1024_floats(mock_st_cls: MagicMock) -> None:
    mock_st_cls.return_value = _make_mock_model(1024)
    from app.rag import embedder

    embedder._model = None  # reset singleton
    with patch.object(embedder, "SentenceTransformer", mock_st_cls):
        result = embedder.embed("What is the force majeure clause?")
    assert isinstance(result, list)
    assert len(result) == 1024
    assert all(isinstance(v, float) for v in result)


@patch("app.rag.embedder._model", None)
@patch("app.rag.embedder.SentenceTransformer")
def test_embed_batch_returns_correct_shape(mock_st_cls: MagicMock) -> None:
    mock_st_cls.return_value = _make_mock_model(1024)
    from app.rag import embedder

    embedder._model = None
    texts = ["query one", "query two", "query three"]
    with patch.object(embedder, "SentenceTransformer", mock_st_cls):
        result = embedder.embed_batch(texts)
    assert len(result) == 3
    assert all(len(v) == 1024 for v in result)


@patch("app.rag.embedder._model", None)
@patch("app.rag.embedder.SentenceTransformer")
def test_bge_prefix_is_applied(mock_st_cls: MagicMock) -> None:
    """BGE instruction prefix must be prepended before encoding."""
    mock_model = _make_mock_model(1024)
    mock_st_cls.return_value = mock_model
    from app.rag import embedder

    embedder._model = None
    query = "Find the termination clause."
    with patch.object(embedder, "SentenceTransformer", mock_st_cls):
        embedder.embed(query)

    # The string passed to encode must include the BGE prefix
    call_args = mock_model.encode.call_args
    encoded_text = call_args[0][0]  # first positional arg
    assert encoded_text.startswith("Represent this document for retrieval: ")
    assert query in encoded_text


@patch("app.rag.embedder._model", None)
@patch("app.rag.embedder.SentenceTransformer")
def test_embed_batch_prefix_applied_to_all(mock_st_cls: MagicMock) -> None:
    mock_model = _make_mock_model(1024)
    mock_st_cls.return_value = mock_model
    from app.rag import embedder

    embedder._model = None
    texts = ["clause a", "clause b"]
    with patch.object(embedder, "SentenceTransformer", mock_st_cls):
        embedder.embed_batch(texts)

    call_args = mock_model.encode.call_args
    prefixed_texts: list[str] = call_args[0][0]
    for t, pt in zip(texts, prefixed_texts):
        assert pt.startswith("Represent this document for retrieval: ")
        assert t in pt


@patch("app.rag.embedder._model", None)
@patch("app.rag.embedder.SentenceTransformer")
def test_model_singleton(mock_st_cls: MagicMock) -> None:
    """get_model() must only instantiate SentenceTransformer once."""
    mock_st_cls.return_value = _make_mock_model(1024)
    from app.rag import embedder

    embedder._model = None
    with patch.object(embedder, "SentenceTransformer", mock_st_cls):
        embedder.get_model()
        embedder.get_model()
        embedder.get_model()
    assert mock_st_cls.call_count == 1
