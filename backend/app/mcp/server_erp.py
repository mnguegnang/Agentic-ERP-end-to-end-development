"""MCP server: mcp-erp-postgres (Blueprint §4.3.6).

Exposes query_erp and get_product_bom tools over the MCP protocol.
Stage 4 implementation.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-erp-postgres")


@mcp.tool()
async def query_erp(query_type: str, filters: dict) -> dict:
    """Query AdventureWorks ERP data via parameterised ORM (no raw SQL)."""
    # TODO Stage 4: dispatch to SQLAlchemy DAL in db/session.py
    return {"results": []}


@mcp.tool()
async def get_product_bom(product_id: int) -> dict:
    """Return bill-of-materials tree for a product."""
    # TODO Stage 4: recursive CTE via SQLAlchemy
    return {"bom_tree": {}}
