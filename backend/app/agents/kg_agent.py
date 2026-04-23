"""KG Agent — Think-on-Graph Neo4j reasoning (Blueprint §4.5).

Three-step pipeline with one self-correction retry:
  1. Entity extraction  — LLM structured output → seed entities + types.
  2. Relation selection — LLM structured output → ordered relation path.
  3. KG traversal       — whitelisted Cypher via kg/client.execute_read().
  4. Path evaluation    — if subgraph empty, retry with PROVIDES fallback.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.agents.graph_state import AgentState
from app.api.schemas import EntityExtractionResult, RelationSelectionResult
from app.config import get_settings
from app.kg.client import execute_read
from app.kg.queries import QUERIES

logger = logging.getLogger(__name__)

_ENTITY_SYSTEM = """\
You are an entity extractor for a supply-chain knowledge graph.
Given a natural-language query, identify all named entities that correspond to
graph nodes (suppliers, components, products, warehouses, distribution centres).
Return entity names as they appear in the query and classify each type."""

_RELATION_SYSTEM = """\
You are a relation selector for a supply-chain knowledge graph.
Given a query and a list of seed entities, select an ordered list of up to 4
relation types to traverse. Choose from:
  PROVIDES, SUPPLIED_BY, USED_IN, STORED_AT, SHIPS_TO,
  ALTERNATIVE_FOR, MANAGED_BY, DISRUPTS.
Return the list that most directly answers the query."""

# Allowed relation types (security whitelist — matches KG schema)
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


def _make_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.github_token,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=512,
    )


async def _extract_entities(query: str) -> EntityExtractionResult:
    """Step 1 — extract named entities from the user query."""
    try:
        llm = _make_llm()
        structured = llm.with_structured_output(EntityExtractionResult)
        result: EntityExtractionResult = await structured.ainvoke(
            [
                SystemMessage(_ENTITY_SYSTEM),
                HumanMessage(f"Query: {query}"),
            ]
        )
        return result
    except Exception as exc:
        logger.warning("KG entity extraction failed: %s", exc)
        return EntityExtractionResult(entities=[], entity_types=[])


async def _select_relations(query: str, entities: list[str]) -> RelationSelectionResult:
    """Step 2 — choose relation path for KG traversal."""
    try:
        llm = _make_llm()
        structured = llm.with_structured_output(RelationSelectionResult)
        result: RelationSelectionResult = await structured.ainvoke(
            [
                SystemMessage(_RELATION_SYSTEM),
                HumanMessage(f"Query: {query}\nEntities: {', '.join(entities)}"),
            ]
        )
        # Enforce whitelist: drop any relation types not in _ALLOWED_RELATIONS
        safe_path = [r for r in result.relation_path if r in _ALLOWED_RELATIONS]
        return RelationSelectionResult(
            relation_path=safe_path or ["PROVIDES"],
            reasoning=result.reasoning,
        )
    except Exception as exc:
        logger.warning("KG relation selection failed: %s", exc)
        return RelationSelectionResult(relation_path=["PROVIDES"], reasoning="fallback")


async def _traverse(
    seed_id: str,
    relation_path: list[str],
    max_depth: int = 4,
    limit: int = 50,
) -> list[dict]:
    """Step 3 — execute whitelisted traversal Cypher."""
    return await execute_read(
        QUERIES["traverse_supply_network"],
        seed_id=seed_id,
        max_depth=max_depth,
        allowed_relations=relation_path,
        limit=limit,
    )


async def kg_agent_node(state: AgentState) -> AgentState:
    """Think-on-Graph: entity extract → relation select → traverse → self-correct."""
    _msgs = state.get("messages") or []
    _last = _msgs[-1] if _msgs else None
    query: str = (
        _last.content if hasattr(_last, "content")  # LangChain message object
        else (_last.get("content", "") if isinstance(_last, dict) else "")
    ) if _last else ""

    # Step 1: entity extraction
    extraction = await _extract_entities(query)
    entities = extraction.entities

    if not entities:
        logger.info("KG agent: no entities found for query %r", query)
        return {
            **state,
            "kg_entities": [],
            "kg_relation_path": [],
            "kg_subgraph": {"nodes": [], "edges": [], "paths": []},
        }

    # Step 2: relation selection
    relation_result = await _select_relations(query, entities)
    relation_path = relation_result.relation_path

    # Step 3: traversal from first seed entity
    seed_id = entities[0]
    rows = await _traverse(seed_id, relation_path)

    # Step 4: self-correction — if empty, retry with PROVIDES fallback
    if not rows:
        logger.info(
            "KG traversal returned 0 rows for seed=%r path=%r; retrying with PROVIDES",
            seed_id,
            relation_path,
        )
        rows = await _traverse(seed_id, ["PROVIDES"])

    subgraph: dict = {
        "nodes": rows,
        "edges": [],
        "paths": [],
        "seed_entity": seed_id,
        "relation_path": relation_path,
    }

    return {
        **state,
        "kg_entities": entities,
        "kg_relation_path": relation_path,
        "kg_subgraph": subgraph,
    }
