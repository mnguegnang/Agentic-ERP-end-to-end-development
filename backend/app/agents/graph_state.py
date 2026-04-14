from __future__ import annotations

from typing import Annotated, Sequence

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """LangGraph shared state schema (Blueprint §2.2)."""

    messages: Annotated[Sequence[dict], add_messages]
    intent: str | None                  # mcnf_solve | jsp_schedule | contract_query | ...
    intent_confidence: float            # 0.0–1.0; <0.7 triggers clarification
    ddd_context: str | None             # "visibility" | "inventory" | "compliance"
    solver_input: dict | None           # MCP-validated solver parameters
    solver_output: dict | None          # raw OR solver result
    kg_subgraph: dict | None            # Neo4j traversal result {nodes, edges, paths}
    rag_documents: list[dict] | None    # retrieved + reranked contract chunks
    rag_evaluation: str | None          # "correct" | "ambiguous" | "incorrect"
    human_approval_required: bool       # True → WebSocket approval request to frontend
    error: str | None
