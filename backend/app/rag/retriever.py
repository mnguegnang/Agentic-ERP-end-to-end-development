"""Hybrid retrieval: pgvector dense + BM25 sparse + RRF fusion (Blueprint §4.4).

Stage 4 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi


@dataclass
class CRAGResult:
    documents: list[dict]
    evaluation: str  # "correct" | "ambiguous" | "incorrect"
    fallback: str | None = None


def reciprocal_rank_fusion(
    dense: list[dict],
    sparse: list[dict],
    k: int = 60,
) -> list[dict]:
    """Cormack et al. RRF with smoothing constant k=60 (Blueprint §2.3)."""
    scores: dict[str, float] = {}
    id_to_doc: dict[str, dict] = {}

    for rank, doc in enumerate(dense):
        doc_id = str(doc.get("id", rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        id_to_doc[doc_id] = doc

    for rank, doc in enumerate(sparse):
        doc_id = str(doc.get("id", rank))
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        id_to_doc[doc_id] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [id_to_doc[doc_id] for doc_id, _ in ranked]


async def retrieve_and_evaluate(
    query: str,
    supplier_id: int | None = None,
    top_k: int = 5,
) -> CRAGResult:
    """Full CRAG pipeline: dense → sparse → RRF → rerank → evaluate."""
    # TODO Stage 4: implement pgvector search, BM25, cross-encoder reranking, evaluator
    return CRAGResult(documents=[], evaluation="not_implemented")
