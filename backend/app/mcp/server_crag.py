"""MCP server: mcp-contract-rag (Blueprint §4.3.5).

Exposes search_contracts via CRAG hybrid retrieval pipeline.
Stage 4 implementation.
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("mcp-contract-rag")


@mcp.tool()
async def search_contracts(
    query: str,
    supplier_id: int | None = None,
    top_k: int = 5,
) -> dict:
    """Hybrid dense+BM25 retrieval → cross-encoder rerank → CRAG evaluation."""
    # TODO Stage 4: wire to rag/retriever.py retrieve_and_evaluate()
    return {"documents": [], "evaluation": "not_implemented"}
