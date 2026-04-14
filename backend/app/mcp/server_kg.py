"""MCP server: mcp-knowledge-graph (Blueprint §4.3.4).

Exposes traverse_supply_network, find_affected_products, get_supplier_alternatives.
Stage 4 implementation.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-knowledge-graph")


@mcp.tool()
async def traverse_supply_network(
    seed_entity: str,
    relation_path: list[str],
    max_depth: int = 4,
) -> dict:
    """Traverse the Neo4j supply-chain KG from a seed entity."""
    # TODO Stage 4: Neo4j client in kg/client.py with whitelisted Cypher patterns
    return {"nodes": [], "edges": [], "paths": []}


@mcp.tool()
async def find_affected_products(supplier_id: int) -> dict:
    """Return products and components affected by a supplier disruption."""
    return {"affected_products": [], "affected_components": [], "paths": []}


@mcp.tool()
async def get_supplier_alternatives(component_id: int) -> dict:
    """Return alternative suppliers for a component with cost/capacity/reliability."""
    return {"alternatives": []}
