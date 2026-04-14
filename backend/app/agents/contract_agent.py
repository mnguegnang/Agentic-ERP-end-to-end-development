"""Contract Agent — CRAG retrieval pipeline (Blueprint §4.4).

Calls the hybrid pgvector + BM25 + RRF + CrossEncoder + LLM evaluator pipeline
and sets ``rag_documents`` / ``rag_evaluation`` on the shared LangGraph state.
"""

from __future__ import annotations

import logging

from app.agents.graph_state import AgentState
from app.rag.retriever import retrieve_and_evaluate

logger = logging.getLogger(__name__)


async def contract_agent_node(state: AgentState) -> AgentState:
    """Invoke CRAG pipeline: hybrid retrieval → rerank → evaluate."""
    # Extract query from the latest human message in the state
    messages = state.get("messages") or []
    query: str = messages[-1].get("content", "") if messages else ""

    if not query:
        logger.warning("contract_agent: empty query in state messages")
        return {**state, "rag_documents": [], "rag_evaluation": "incorrect"}

    try:
        result = await retrieve_and_evaluate(query, supplier_id=None, top_k=5)
        return {
            **state,
            "rag_documents": result.documents,
            "rag_evaluation": result.evaluation,
        }
    except Exception as exc:
        logger.exception("contract_agent retrieve_and_evaluate failed: %s", exc)
        return {**state, "rag_documents": [], "rag_evaluation": "incorrect"}
