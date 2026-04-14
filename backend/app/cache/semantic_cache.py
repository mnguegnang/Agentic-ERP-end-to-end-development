"""Redis semantic cache with SHA-256 key (Blueprint §4.8).

Full implementation — all fields specified in blueprint.
"""

from __future__ import annotations

import hashlib
import json

import redis.asyncio as redis


class SemanticCache:
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600) -> None:
        self.redis = redis_client
        self.ttl = ttl

    def _key(self, query: str, intent: str) -> str:
        normalized = query.strip().lower()
        return f"cache:{intent}:{hashlib.sha256(normalized.encode()).hexdigest()}"

    async def get(self, query: str, intent: str) -> dict | None:
        data = await self.redis.get(self._key(query, intent))
        return json.loads(data) if data else None

    async def set(self, query: str, intent: str, result: dict) -> None:
        await self.redis.setex(self._key(query, intent), self.ttl, json.dumps(result))
