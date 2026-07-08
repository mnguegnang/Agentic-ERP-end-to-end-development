"""Redis semantic cache (Blueprint §4.8).

Two-level lookup, checked *before* the LangGraph traversal:

  1. Exact match  — SHA-256 of the normalised query.
  2. Semantic match — cosine similarity between the query embedding and the
     embeddings of recently cached queries (threshold is deliberately high:
     a false cache hit is worse than a miss).

The key is derived from the raw query only — intent is not known until after
classification, which is exactly the work the cache is meant to skip.

Embeddings are supplied via an injected ``embed_fn`` so unit tests don't need
sentence-transformers, and the production factory reuses the BGE model already
loaded for CRAG retrieval.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Callable

import numpy as np
import redis.asyncio as redis

logger = logging.getLogger(__name__)

_RESP_PREFIX = "cache:resp:"
_VEC_PREFIX = "cache:vec:"

#: Cosine threshold for treating two queries as duplicates.  BGE embeddings
#: are normalised, so the dot product is the cosine similarity.
_DEFAULT_SIMILARITY_THRESHOLD = 0.95

#: Upper bound on candidate vectors examined per semantic lookup (SCAN cost).
_DEFAULT_MAX_CANDIDATES = 512

# ---------------------------------------------------------------------------
# Lexical guard for semantic matches.
#
# Two supply-chain queries can be >0.95 cosine similar yet have completely
# different answers when they differ only in a *parameter*: "Allocate 400
# units…" vs "Allocate 1000 units…", or "payment terms for supplier 4" vs
# "supplier 5", or the graph for "TQ-Electronics" vs "SHA-Electronics".  The
# embedding washes those tokens out.  So a semantic hit is only accepted when
# the two queries share the SAME discriminating tokens: the multiset of
# numbers and the set of entity codes (hyphenated / alphanumeric identifiers).
# Pure-language paraphrases (same numbers/codes, different filler words) still
# hit; parameter changes do not.
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_CODE_RE = re.compile(r"\b[A-Za-z0-9]+-[A-Za-z0-9-]+\b")


def _discriminating_tokens(text: str) -> tuple[tuple[str, ...], frozenset[str]]:
    """Return (sorted number multiset, entity-code set) for ``text``."""
    numbers = tuple(sorted(_NUMBER_RE.findall(text)))
    codes = frozenset(m.lower() for m in _CODE_RE.findall(text))
    return numbers, codes


def _tokens_compatible(a: str, b: str) -> bool:
    """True when two queries share the same numbers and entity codes."""
    return _discriminating_tokens(a) == _discriminating_tokens(b)


class SemanticCache:
    def __init__(
        self,
        redis_client: redis.Redis,
        ttl: int = 3600,
        embed_fn: Callable[[str], list[float]] | None = None,
        similarity_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD,
        max_candidates: int = _DEFAULT_MAX_CANDIDATES,
    ) -> None:
        self.redis = redis_client
        self.ttl = ttl
        self.embed_fn = embed_fn
        self.similarity_threshold = similarity_threshold
        self.max_candidates = max_candidates

    @staticmethod
    def _digest(query: str) -> str:
        normalized = query.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()

    async def _embed(self, query: str) -> np.ndarray | None:
        if self.embed_fn is None:
            return None
        # encode() is CPU-bound — keep it off the event loop.
        vec = await asyncio.to_thread(self.embed_fn, query)
        return np.asarray(vec, dtype=np.float32)

    async def get(self, query: str) -> dict | None:
        """Return a cached response for ``query`` or None.

        Exact match first, then embedding similarity over cached queries.
        Any Redis/embedding failure degrades to a cache miss.
        """
        try:
            digest = self._digest(query)
            data = await self.redis.get(_RESP_PREFIX + digest)
            if data:
                return json.loads(data)
            return await self._semantic_get(query)
        except Exception as exc:
            logger.warning("semantic cache get failed: %s", exc)
            return None

    async def _semantic_get(self, query: str) -> dict | None:
        query_vec = await self._embed(query)
        if query_vec is None:
            return None

        best_digest: str | None = None
        best_score = self.similarity_threshold
        examined = 0
        async for vec_key in self.redis.scan_iter(match=_VEC_PREFIX + "*", count=100):
            if examined >= self.max_candidates:
                break
            examined += 1
            raw = await self.redis.get(vec_key)
            if not raw:
                continue
            payload = json.loads(raw)
            # New format: {"v": [...], "q": "..."}.  Old bare-list entries have
            # no stored query to guard against, so they are skipped (safe).
            if not isinstance(payload, dict):
                continue
            cached_query = payload.get("q")
            cached_vec = np.asarray(payload.get("v"), dtype=np.float32)
            if cached_vec.shape != query_vec.shape:
                continue
            score = float(np.dot(query_vec, cached_vec))
            if score < best_score:
                continue
            # Lexical guard: a high cosine is not enough — the discriminating
            # tokens (numbers, entity codes) must match, or a parametric query
            # would be answered from a different parameter's cached result.
            if not (cached_query and _tokens_compatible(query, cached_query)):
                continue
            best_score = score
            key = vec_key.decode() if isinstance(vec_key, bytes) else str(vec_key)
            best_digest = key.removeprefix(_VEC_PREFIX)

        if best_digest is None:
            return None
        data = await self.redis.get(_RESP_PREFIX + best_digest)
        if data:
            logger.info("semantic cache hit (cosine=%.3f)", best_score)
            return json.loads(data)
        return None

    async def set(self, query: str, result: dict) -> None:
        """Cache ``result`` under both the exact digest and the query embedding.

        The raw query is stored alongside the vector so the semantic lookup can
        apply the lexical guard (numbers / entity codes must match).  Failures
        are logged and swallowed — caching must never break the chat.
        """
        try:
            digest = self._digest(query)
            await self.redis.setex(_RESP_PREFIX + digest, self.ttl, json.dumps(result))
            query_vec = await self._embed(query)
            if query_vec is not None:
                payload = {"v": query_vec.tolist(), "q": query}
                await self.redis.setex(_VEC_PREFIX + digest, self.ttl, json.dumps(payload))
        except Exception as exc:
            logger.warning("semantic cache set failed: %s", exc)


# ---------------------------------------------------------------------------
# Process-wide instance used by the chat route
# ---------------------------------------------------------------------------

_CACHE: SemanticCache | None = None


def _production_embed(query: str) -> list[float]:
    """Embed with the same BGE model CRAG retrieval uses (loaded lazily once)."""
    from app.rag.retriever import _get_embedder

    return _get_embedder().encode(query, normalize_embeddings=True).tolist()


def get_semantic_cache() -> SemanticCache:
    global _CACHE
    if _CACHE is None:
        from app.config import get_settings

        s = get_settings()
        _CACHE = SemanticCache(
            redis_client=redis.from_url(s.redis_url),
            ttl=s.cache_ttl,
            embed_fn=_production_embed,
        )
    return _CACHE
