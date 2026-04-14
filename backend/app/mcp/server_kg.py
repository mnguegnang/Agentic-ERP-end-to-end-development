"""MCP server: mcp-knowledge-graph (Blueprint §4.3.4).

Exposes traverse_supply_network, find_affected_products, get_supplier_alternatives.
All queries use whitelisted Cypher from kg/queries.py — no raw LLM Cypher executed.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP

from app.kg.client import execute_read
from app.kg.queries import QUERIES

logger = logging.getLogger(__name__)

mcp = FastMCP("mcp-knowledge-graph")

# Allowed relation types — same whitelist as kg_agent.py
_ALLOWED_RELATIONS = {
    "PROVIDES",
    "SUPPLIED_BY",
    "USED_IN",
    "STORED_AT",
    "SHIPS_TO",
    "ALTERNATIVE_FOR",
    "MANAGED_BY",
    "DISRUPTS",
}


@mcp.tool()
async def traverse_supply_network(
    seed_entity: str,
    relation_path: list[str],
    max_depth: int = 4,
) -> dict:
    """Traverse the Neo4j supply-chain KG from a seed entity."""
    safe_path = [r for r in relation_path if r in _ALLOWED_RELATIONS] or ["PROVIDES"]
    rows = await execute_read(
        QUERIES["traverse_supply_network"],
        seed_id=seed_entity,
        max_depth=max_depth,
        allowed_relations=safe_path,
        limit=50,
    )
    return {"nodes": rows, "edges": [], "paths": []}


@mcp.tool()
async def find_affected_products(supplier_id: int) -> dict:
    """Return products and components affected by a supplier disruption."""
    rows = await execute_read(
        QUERIES["find_affected_products"],
        supplier_id=supplier_id,
    )
    return {"results": rows, "affected_count": len(rows)}


@mcp.tool()
async def get_supplier_alternatives(component_id: int) -> dict:
    """Return alternative suppliers for a component, ordered by cost ASC."""
    rows = await execute_read(
        QUERIES["get_supplier_alternatives"],
        component_id=component_id,
    )
    return {"alternatives": rows}
