"""Neo4j async client (Blueprint §4.3.4).

Provides a lazily-initialised singleton driver and a safe read helper
that only executes whitelisted Cypher from kg/queries.py.
"""

from __future__ import annotations

import logging

from neo4j import AsyncDriver, AsyncGraphDatabase

from app.config import get_settings

logger = logging.getLogger(__name__)

_driver: AsyncDriver | None = None


async def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        s = get_settings()
        _driver = AsyncGraphDatabase.driver(
            s.neo4j_uri, auth=("neo4j", s.neo4j_password)
        )
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver:
        await _driver.close()
        _driver = None


async def execute_read(cypher: str, **params: object) -> list[dict]:
    """Execute a read-only Cypher query and return rows as plain dicts.

    Only whitelisted Cypher strings from kg/queries.py should be passed here.
    The helper does NOT accept raw LLM-generated Cypher — callers must look up
    the query template from ``app.kg.queries.QUERIES`` before calling this.

    Parameters
    ----------
    cypher:
        Parameterised Cypher string (from the whitelist).
    **params:
        Named parameters bound to the Cypher placeholders.

    Returns
    -------
    list[dict]
        Rows from ``result.data()``.  Empty list on failure.
    """
    try:
        driver = await get_driver()
        async with driver.session() as session:
            result = await session.run(cypher, **params)  # type: ignore[arg-type]
            return await result.data()
    except Exception as exc:
        logger.warning("execute_read failed: %s", exc)
        return []
