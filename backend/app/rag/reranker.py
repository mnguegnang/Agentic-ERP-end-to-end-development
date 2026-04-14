"""Cross-encoder reranking via ms-marco-MiniLM-L-12-v2 (Blueprint §4.4).

Stage 4 implementation.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-12-v2"
_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(_MODEL_NAME)
    return _reranker


def rerank(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    """Score query-document pairs and return top_k by relevance."""
    if not docs:
        return []
    reranker = get_reranker()
    pairs = [(query, d.get("chunk_text", "")) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
