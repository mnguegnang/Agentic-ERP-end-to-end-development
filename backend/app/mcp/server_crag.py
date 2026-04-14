"""MCP server: mcp-contract-rag (Blueprint §4.3.5).

Exposes search_contracts via the CRAG hybrid retrieval pipeline:
  pgvector dense + BM25 sparse → RRF → CrossEncoder rerank → LLM evaluation.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from app.rag.retriever import retrieve_and_evaluate

mcp = FastMCP("mcp-contract-rag")


@mcp.tool()
async def search_contracts(
    query: str,
    supplier_id: int | None = None,
    top_k: int = 5,
) -> dict:
    """Hybrid dense+BM25 retrieval → cross-encoder rerank → CRAG evaluation."""
    result = await retrieve_and_evaluate(query, supplier_id=supplier_id, top_k=top_k)
    return {
        "documents": result.documents,
        "evaluation": result.evaluation,
        "fallback": result.fallback,
    }
