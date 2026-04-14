"""Neo4j async client (Blueprint §4.3.4).

Stage 4 implementation.
"""
from __future__ import annotations

from neo4j import AsyncGraphDatabase, AsyncDriver

from app.config import get_settings

_driver: AsyncDriver | None = None


async def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        s = get_settings()
        _driver = AsyncGraphDatabase.driver(s.neo4j_uri, auth=("neo4j", s.neo4j_password))
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver:
        await _driver.close()
        _driver = None
