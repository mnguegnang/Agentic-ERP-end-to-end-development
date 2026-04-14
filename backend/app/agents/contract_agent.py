"""Contract Agent — CRAG retrieval pipeline (Blueprint §4.4).

Stage 4 implementation. Stub provided for scaffolding.
"""
from __future__ import annotations

from app.agents.graph_state import AgentState


async def contract_agent_node(state: AgentState) -> AgentState:
    """Invoke CRAG pipeline: hybrid retrieval → rerank → evaluate → synthesize."""
    # TODO Stage 4: call mcp_search_contracts, set rag_documents and rag_evaluation
    return {**state, "rag_documents": None, "rag_evaluation": None}
