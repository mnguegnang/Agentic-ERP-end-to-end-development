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
import time
from dataclasses import dataclass

import asyncpg
from pgvector.asyncpg import register_vector
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.rag.evaluator import (
    AMBIGUOUS,
    CORRECT,
    INCORRECT,
    evaluate_relevance,  # noqa: F401 — kept for callers/tests using the single-doc API
    evaluate_relevance_batch,
    rewrite_query,
)
from app.rag.reranker import rerank

logger = logging.getLogger(__name__)

# BGE-large-en-v1.5 (1024-dim) — same model used at indexing time (Blueprint §2.2)
_EMBED_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_embedder: SentenceTransformer | None = None

# ivfflat probe count for the pgvector search.  The contract index is built with
# lists=100; probing all of them makes the ANN search exhaustive (exact) and
# deterministic on this small corpus.  Raise the index's lists and lower this
# proportionally if the corpus grows large enough for approximate search to pay off.
_IVFFLAT_PROBES = 100


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
    """Cosine similarity search via pgvector over supply_chain.contract_embeddings.

    ``supplier_id`` lives on supply_chain.contracts (not on the chunk rows), so
    supplier filtering joins through the contract_id foreign key.
    """
    try:
        # Strip asyncpg prefix if present (e.g. "postgresql+asyncpg://...")
        raw_dsn = dsn.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(raw_dsn)
        await register_vector(conn)

        # The contract corpus is indexed with ivfflat(lists=100).  For a small
        # corpus that is far more lists than rows-per-list warrants, so the
        # default probes=1 samples a near-empty list and can return too few
        # (even zero) neighbours — especially once a JOIN reshapes the plan.
        # Probing every list makes the search exhaustive (i.e. exact) and
        # deterministic; on this corpus that is effectively free.
        await conn.execute(f"SET ivfflat.probes = {_IVFFLAT_PROBES}")

        if supplier_id is not None:
            rows = await conn.fetch(
                """
                SELECT ce.id::text AS id, ce.chunk_text, ce.contract_id, c.supplier_id,
                       1 - (ce.embedding <=> $1::vector) AS score
                FROM supply_chain.contract_embeddings ce
                JOIN supply_chain.contracts c ON ce.contract_id = c.contract_id
                WHERE c.supplier_id = $2
                ORDER BY ce.embedding <=> $1::vector
                LIMIT $3
                """,
                embedding,
                supplier_id,
                top_k,
            )
        else:
            rows = await conn.fetch(
                """
                SELECT ce.id::text AS id, ce.chunk_text, ce.contract_id, c.supplier_id,
                       1 - (ce.embedding <=> $1::vector) AS score
                FROM supply_chain.contract_embeddings ce
                JOIN supply_chain.contracts c ON ce.contract_id = c.contract_id
                ORDER BY ce.embedding <=> $1::vector
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


# ---------------------------------------------------------------------------
# BM25 corpus cache — the contract corpus changes on ingest, not per query, so
# rebuilding BM25Okapi (full-table SELECT + tokenise) on every request is pure
# waste.  Cache per supplier_id; the TTL bounds staleness after out-of-process
# re-ingest (seed scripts), and in-process ingestion can call
# invalidate_bm25_cache() for an immediate refresh.
# ---------------------------------------------------------------------------
_BM25_CACHE: dict[str, tuple[float, list[dict], BM25Okapi | None]] = {}
_BM25_CACHE_TTL_SECONDS = 600.0


def invalidate_bm25_cache() -> None:
    """Drop all cached BM25 corpora (call after (re)ingesting contracts)."""
    _BM25_CACHE.clear()


async def _load_bm25_corpus(
    supplier_id: int | None,
    dsn: str,
) -> tuple[list[dict], BM25Okapi | None]:
    """Return (meta_list, bm25_index) for the contract corpus, cached with TTL.

    On DB failure returns an empty corpus (and does not cache the failure).
    """
    cache_key = f"supplier:{supplier_id}"
    cached = _BM25_CACHE.get(cache_key)
    if cached is not None:
        built_at, meta, index = cached
        if time.monotonic() - built_at < _BM25_CACHE_TTL_SECONDS:
            return meta, index
        del _BM25_CACHE[cache_key]

    meta, index = await _build_bm25_corpus(supplier_id, dsn)
    if meta:
        _BM25_CACHE[cache_key] = (time.monotonic(), meta, index)
    return meta, index


async def _build_bm25_corpus(
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
                "SELECT ce.id::text AS id, ce.chunk_text, ce.contract_id, c.supplier_id "
                "FROM supply_chain.contract_embeddings ce "
                "JOIN supply_chain.contracts c ON ce.contract_id = c.contract_id "
                "WHERE c.supplier_id = $1",
                supplier_id,
            )
        else:
            rows = await conn.fetch(
                "SELECT ce.id::text AS id, ce.chunk_text, ce.contract_id, c.supplier_id "
                "FROM supply_chain.contract_embeddings ce "
                "JOIN supply_chain.contracts c ON ce.contract_id = c.contract_id"
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


async def _hybrid_retrieve(
    query: str,
    supplier_id: int | None,
    top_k: int,
) -> list[dict] | None:
    """Embed → dense + sparse → RRF → rerank.  None signals an embedding error."""
    s = get_settings()
    rag_cfg = s.rag_config
    dense_k: int = int(rag_cfg.get("top_k_dense", top_k * 2))
    sparse_k: int = int(rag_cfg.get("top_k_sparse", top_k * 2))

    # 1. Embed query
    try:
        embedder = _get_embedder()
        query_vec: list[float] = embedder.encode(query, normalize_embeddings=True).tolist()
    except Exception as exc:
        logger.warning("Embedding failed: %s", exc)
        return None

    # 2. Dense (pgvector) + sparse-corpus load in parallel via asyncio.gather
    import asyncio

    dense_task = _pgvector_search(query_vec, supplier_id, dense_k, s.database_url)
    corpus_task = _load_bm25_corpus(supplier_id, s.database_url)
    dense_results, (bm25_meta, bm25_index) = await asyncio.gather(dense_task, corpus_task)

    # 3. BM25 sparse search
    sparse_results: list[dict] = []
    if bm25_index is not None and bm25_meta:
        sparse_results = _bm25_search(query, bm25_meta, bm25_index, sparse_k)

    # 4. RRF fusion
    fused = reciprocal_rank_fusion(dense_results, sparse_results)

    # 5. CrossEncoder re-rank
    return rerank(query, fused, top_k=top_k)


async def _evaluate_documents(query: str, reranked: list[dict]) -> CRAGResult | None:
    """Per-document CRAG gate: keep relevant chunks, or None if nothing survives.

    Unlike top-1 gating, one bad top hit no longer discards good chunks below
    it, and irrelevant chunks are dropped instead of riding along to synthesis.
    """
    labels = await evaluate_relevance_batch(query, reranked)
    kept = [doc for doc, label in zip(reranked, labels) if label != INCORRECT]
    if not kept:
        return None
    evaluation = CORRECT if CORRECT in labels else AMBIGUOUS
    return CRAGResult(documents=kept, evaluation=evaluation)


async def retrieve_and_evaluate(
    query: str,
    supplier_id: int | None = None,
    top_k: int = 5,
) -> CRAGResult:
    """Full CRAG pipeline (Blueprint §4.4): hybrid retrieval → per-document
    relevance gate → one corrective query-rewrite retry before giving up."""
    reranked = await _hybrid_retrieve(query, supplier_id, top_k)
    if reranked is None:
        return CRAGResult(documents=[], evaluation=INCORRECT, fallback="embedding_error")
    if not reranked:
        return CRAGResult(documents=[], evaluation=INCORRECT, fallback="no_results")

    result = await _evaluate_documents(query, reranked)
    if result is not None:
        return result

    # Corrective action (CRAG): rewrite the query once and retry retrieval.
    rewritten = await rewrite_query(query)
    if rewritten:
        logger.info("CRAG corrective rewrite: %r → %r", query, rewritten)
        reranked_retry = await _hybrid_retrieve(rewritten, supplier_id, top_k)
        if reranked_retry:
            # Relevance is judged against the user's original question.
            result = await _evaluate_documents(query, reranked_retry)
            if result is not None:
                return CRAGResult(
                    documents=result.documents,
                    evaluation=result.evaluation,
                    fallback="query_rewritten",
                )

    return CRAGResult(documents=[], evaluation=INCORRECT, fallback="no_answer")
