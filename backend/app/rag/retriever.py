"""Hybrid retrieval: pgvector dense + BM25 sparse + RRF fusion (Blueprint §4.4).

Pipeline:
    1. Embed query with the same BGE-large model used at index time.
    2. pgvector cosine search → top-K dense candidates.
    3. BM25Okapi sparse search over corpus loaded lazily from Postgres.
    4. Reciprocal Rank Fusion (k=60) to merge candidate lists.
    5. CrossEncoder re-rank to final top-K.
    6. LLM relevance evaluation on the best chunk (CRAG gate).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import asyncpg
from pgvector.asyncpg import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.rag.evaluator import INCORRECT, evaluate_relevance
from app.rag.reranker import rerank

logger = logging.getLogger(__name__)

# BGE-large-en-v1.5 (1024-dim) — same model used at indexing time (Blueprint §2.2)
_EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedder


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


async def _pgvector_search(
    embedding: list[float],
    supplier_id: int | None,
    top_k: int,
    dsn: str,
) -> list[dict]:
    """Cosine similarity search via pgvector over contract_chunks table."""
    try:
        # Strip asyncpg prefix if present (e.g. "postgresql+asyncpg://...")
        raw_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(raw_dsn)
        await register_vector(conn)

        if supplier_id is not None:
            rows = await conn.fetch(
                """
                SELECT id, chunk_text, contract_id, supplier_id,
                       1 - (embedding <=> $1::vector) AS score
                FROM contract_chunks
                WHERE supplier_id = $2
                ORDER BY embedding <=> $1::vector
                LIMIT $3
                """,
                embedding,
                supplier_id,
                top_k,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT id, chunk_text, contract_id, supplier_id,
                       1 - (embedding <=> $1::vector) AS score
                FROM contract_chunks
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                embedding,
                top_k,
            )
        await conn.close()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("pgvector search failed: %s", exc)
        return []


async def _load_bm25_corpus(
    supplier_id: int | None,
    dsn: str,
) -> tuple[list[dict], BM25Okapi | None]:
    """Load chunk_text from DB and build BM25Okapi corpus.

    Returns (meta_list, bm25_index).  On failure returns empty corpus.
    """
    try:
        raw_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(raw_dsn)
        if supplier_id is not None:
            rows = await conn.fetch(
                "SELECT id, chunk_text, contract_id, supplier_id "
                "FROM contract_chunks WHERE supplier_id = $1",
                supplier_id,
            )
        else:
            rows = await conn.fetch(
                "SELECT id, chunk_text, contract_id, supplier_id "
                "FROM contract_chunks"
            )
        await conn.close()
        meta = [dict(r) for r in rows]
        if not meta:
            return [], None
        tokenised = [r["chunk_text"].lower().split() for r in meta]
        return meta, BM25Okapi(tokenised)
    except Exception as exc:
        logger.warning("BM25 corpus load failed: %s", exc)
        return [], None


def _bm25_search(
    query: str,
    meta: list[dict],
    bm25_index: BM25Okapi,
    top_k: int,
) -> list[dict]:
    """Return top-k BM25Okapi results sorted by descending score."""
    tokens = query.lower().split()
    scores = bm25_index.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return [meta[i] for i in ranked_indices[:top_k]]


async def retrieve_and_evaluate(
    query: str,
    supplier_id: int | None = None,
    top_k: int = 5,
) -> CRAGResult:
    """Full CRAG pipeline: dense → sparse → RRF → rerank → evaluate (Blueprint §4.4)."""
    s = get_settings()
    rag_cfg = s.rag_config
    dense_k: int = int(rag_cfg.get("top_k_dense", top_k * 2))
    sparse_k: int = int(rag_cfg.get("top_k_sparse", top_k * 2))

    # 1. Embed query
    try:
        embedder = _get_embedder()
        query_vec: list[float] = embedder.encode(
            query, normalize_embeddings=True
        ).tolist()
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return CRAGResult(
            documents=[], evaluation=INCORRECT, fallback="embedding_error"
        )

    # 2. Dense (pgvector) + sparse (BM25) in parallel via asyncio.gather
    import asyncio

    dense_task = _pgvector_search(query_vec, supplier_id, dense_k, s.database_url)
    corpus_task = _load_bm25_corpus(supplier_id, s.database_url)
    dense_results, (bm25_meta, bm25_index) = await asyncio.gather(
        dense_task, corpus_task
    )

    # 3. BM25 sparse search
    sparse_results: list[dict] = []
    if bm25_index is not None and bm25_meta:
        sparse_results = _bm25_search(query, bm25_meta, bm25_index, sparse_k)

    # 4. RRF fusion
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    # 5. CrossEncoder re-rank
    reranked = rerank(query, fused, top_k=top_k)

    if not reranked:
        return CRAGResult(documents=[], evaluation=INCORRECT, fallback="no_results")

    # 6. CRAG evaluation on best chunk
    top_doc = reranked[0]
    evaluation = await evaluate_relevance(query, top_doc)

    if evaluation == INCORRECT:
        return CRAGResult(documents=[], evaluation=INCORRECT, fallback="no_answer")

    return CRAGResult(documents=reranked, evaluation=evaluation)
