"""KG Agent — Think-on-Graph Neo4j reasoning (Blueprint §4.5).

Stage 4 implementation. Stub provided for scaffolding.
"""
from __future__ import annotations

from app.agents.graph_state import AgentState


async def kg_agent_node(state: AgentState) -> AgentState:
    """Extract seed entities → select traversal path → execute via MCP → self-correct."""
    # TODO Stage 4:
    #   1. llm_extract_entities(query)
    #   2. llm_select_relations(query, seed_entities)
    #   3. mcp_traverse_supply_network(seed, path, max_depth=4)
    #   4. llm_evaluate_path → self-correction loop
    return {**state, "kg_subgraph": None}
