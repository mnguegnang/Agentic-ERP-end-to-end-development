"""Unit tests for the two-level semantic cache (exact + embedding similarity).

Uses a minimal in-memory fake Redis — no external services required.
"""

from __future__ import annotations

import fnmatch
import json

import pytest
from app.cache.semantic_cache import SemanticCache


class FakeRedis:
    """In-memory stand-in for redis.asyncio.Redis (get/setex/scan_iter only)."""

    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self.store.get(key)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.store[key] = value

    async def scan_iter(self, match: str = "*", count: int = 100):
        for key in list(self.store):
            if fnmatch.fnmatch(key, match):
                yield key


class BrokenRedis:
    """Raises on every operation — simulates Redis being down."""

    async def get(self, key: str) -> str | None:
        raise ConnectionError("redis down")

    async def setex(self, key: str, ttl: int, value: str) -> None:
        raise ConnectionError("redis down")

    async def scan_iter(self, match: str = "*", count: int = 100):
        raise ConnectionError("redis down")
        yield  # pragma: no cover


# Toy embeddings: normalised 2-d vectors so cosine is easy to reason about.
_EMBEDDINGS = {
    "route 500 units to berlin": [1.0, 0.0],
    "send 500 units to berlin": [0.999, 0.0447],  # cosine ≈ 0.999 vs above
    "what is the bullwhip effect?": [0.0, 1.0],  # orthogonal to the others
    # Same phrasing, different NUMBER — near-identical embedding, different answer.
    "route 1000 units to berlin": [0.9995, 0.0316],
    # Entity-code cases — near-identical embeddings, different / same code.
    "graph for tq-electronics": [1.0, 0.0],
    "network for tq-electronics": [0.999, 0.0447],  # paraphrase, same code
    "graph for sha-electronics": [0.9995, 0.0316],  # different code
}


def _toy_embed(query: str) -> list[float]:
    return _EMBEDDINGS[query.strip().lower()]


@pytest.fixture
def cache() -> SemanticCache:
    return SemanticCache(FakeRedis(), ttl=60, embed_fn=_toy_embed)


@pytest.mark.asyncio
async def test_exact_hit_ignores_case_and_whitespace(cache: SemanticCache) -> None:
    await cache.set("Route 500 units to Berlin", {"content": "answer"})
    assert await cache.get("  route 500 units to berlin ") == {"content": "answer"}


@pytest.mark.asyncio
async def test_semantic_hit_on_paraphrase(cache: SemanticCache) -> None:
    await cache.set("route 500 units to berlin", {"content": "answer"})
    assert await cache.get("send 500 units to berlin") == {"content": "answer"}


@pytest.mark.asyncio
async def test_miss_on_unrelated_query(cache: SemanticCache) -> None:
    await cache.set("route 500 units to berlin", {"content": "answer"})
    assert await cache.get("what is the bullwhip effect?") is None


@pytest.mark.asyncio
async def test_lexical_guard_rejects_different_number(cache: SemanticCache) -> None:
    """A near-identical embedding must NOT hit when the number differs —
    this is the 'Allocate 400' vs 'Allocate 1000' bug."""
    await cache.set("route 500 units to berlin", {"content": "answer-for-500"})
    assert await cache.get("route 1000 units to berlin") is None


@pytest.mark.asyncio
async def test_lexical_guard_rejects_different_entity_code(cache: SemanticCache) -> None:
    await cache.set("graph for tq-electronics", {"content": "answer-tq"})
    assert await cache.get("graph for sha-electronics") is None


@pytest.mark.asyncio
async def test_lexical_guard_allows_paraphrase_same_entity(cache: SemanticCache) -> None:
    """Same entity code, different filler words → genuine semantic hit."""
    await cache.set("graph for tq-electronics", {"content": "answer-tq"})
    assert await cache.get("network for tq-electronics") == {"content": "answer-tq"}


@pytest.mark.asyncio
async def test_exact_match_works_without_embedder() -> None:
    cache = SemanticCache(FakeRedis(), ttl=60, embed_fn=None)
    await cache.set("route 500 units to berlin", {"content": "answer"})
    assert await cache.get("route 500 units to berlin") == {"content": "answer"}
    # No embedder → paraphrase cannot hit, but must not raise either.
    _EMBEDDINGS.setdefault("send 500 units to berlin", [0.999, 0.0447])
    assert await cache.get("send 500 units to berlin") is None


@pytest.mark.asyncio
async def test_redis_failure_degrades_to_miss() -> None:
    cache = SemanticCache(BrokenRedis(), ttl=60, embed_fn=_toy_embed)
    await cache.set("route 500 units to berlin", {"content": "answer"})  # must not raise
    assert await cache.get("route 500 units to berlin") is None


@pytest.mark.asyncio
async def test_stored_payload_is_json_roundtripped(cache: SemanticCache) -> None:
    payload = {"content": "answer", "solver_result": {"total_cost": 42.0}}
    await cache.set("route 500 units to berlin", payload)
    fake: FakeRedis = cache.redis  # type: ignore[assignment]
    stored_keys = [k for k in fake.store if k.startswith("cache:resp:")]
    assert len(stored_keys) == 1
    assert json.loads(fake.store[stored_keys[0]]) == payload
