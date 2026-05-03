"""KG Agent — Think-on-Graph Neo4j reasoning (Blueprint §4.5).

Three-step pipeline with one self-correction retry:
  1. Entity extraction  — LLM structured output → seed entities + types.
  2. Relation selection — LLM structured output → ordered relation path.
  3. KG traversal       — whitelisted Cypher via kg/client.execute_read().
  4. Path evaluation    — if subgraph empty, retry with PROVIDES fallback.
"""

from __future__ import annotations

import logging
import re

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
    "BOUND_BY",
}


def _make_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.github_token,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=512,  # type: ignore[call-arg]
    )


async def _extract_entities(query: str) -> EntityExtractionResult:
    """Step 1 — extract named entities from the user query."""
    try:
        llm = _make_llm()
        structured = llm.with_structured_output(EntityExtractionResult)
        result: EntityExtractionResult = await structured.ainvoke(  # type: ignore[assignment]
            [
                SystemMessage(_ENTITY_SYSTEM),
                HumanMessage(f"Query: {query}"),
            ]
        )
        if result.entities:
            return result
        # LLM returned empty — fall through to heuristic
        raise ValueError("LLM returned no entities")
    except Exception as exc:
        logger.warning("KG entity extraction failed (%s); using heuristic fallback", exc)
        return _heuristic_extract(query)


def _heuristic_extract(query: str) -> EntityExtractionResult:
    """Regex/keyword entity extraction when the LLM is unavailable.

    Matches capitalized proper-noun sequences (2+ tokens starting with a
    capital letter) that are likely company, product, or component names.
    Stops before common stop-words used in queries.
    """
    # Remove common stop-word phrases before extraction
    cleaned = re.sub(
        r'\b(show|me|the|for|of|in|all|relationships?|connections?|network|'
        r'supply chain|subgraph|graph|traverse|visuali[sz]e?|and|or|with|'
        r'across|from|to|about|using|how|what|who|where|why|when)\b',
        ' ', query, flags=re.IGNORECASE
    )
    # Match capitalized proper-noun sequences (1–5 words)
    pattern = r'(?:[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+){0,4})'
    matches = re.findall(pattern, cleaned)
    # Deduplicate while preserving order; drop very short generic tokens
    seen: set[str] = set()
    entities: list[str] = []
    for m in matches:
        m = m.strip()
        if len(m) >= 3 and m not in seen:
            seen.add(m)
            entities.append(m)
    logger.info("Heuristic entity extraction from %r → %s", query, entities)
    return EntityExtractionResult(entities=entities, entity_types=["Supplier"] * len(entities))


def _heuristic_relations(query: str) -> RelationSelectionResult:
    """Keyword-based relation selection used when LLM is unavailable."""
    q = query.lower()
    if any(kw in q for kw in ("alternative", "backup", "substitute")):
        path = ["ALTERNATIVE_FOR", "SUPPLIED_BY"]
    elif any(kw in q for kw in ("contract", "clause", "agreement")):
        path = ["BOUND_BY"]
    elif any(kw in q for kw in ("component", "part", "material")):
        path = ["USED_IN", "SUPPLIED_BY"]
    elif any(kw in q for kw in ("warehouse", "storage", "stored")):
        path = ["STORED_AT"]
    elif any(kw in q for kw in ("ship", "deliver", "route")):
        path = ["SHIPS_TO"]
    elif any(kw in q for kw in ("disruption", "disrupt")):
        path = ["DISRUPTS", "SUPPLIED_BY"]
    else:
        # Default: full supply chain traversal
        path = ["SUPPLIED_BY", "PROVIDES", "USED_IN"]
    return RelationSelectionResult(relation_path=path, reasoning="heuristic")


async def _select_relations(query: str, entities: list[str]) -> RelationSelectionResult:
    """Step 2 — choose relation path for KG traversal."""
    try:
        llm = _make_llm()
        structured = llm.with_structured_output(RelationSelectionResult)
        result: RelationSelectionResult = await structured.ainvoke(  # type: ignore[assignment]
            [
                SystemMessage(_RELATION_SYSTEM),
                HumanMessage(f"Query: {query}\nEntities: {', '.join(entities)}"),
            ]
        )
        # Enforce whitelist: drop any relation types not in _ALLOWED_RELATIONS
        safe_path = [r for r in result.relation_path if r in _ALLOWED_RELATIONS]
        if safe_path:
            return RelationSelectionResult(
                relation_path=safe_path,
                reasoning=result.reasoning,
            )
        raise ValueError("LLM returned no valid relations")
    except Exception as exc:
        logger.warning("KG relation selection failed (%s); using heuristic fallback", exc)
        return _heuristic_relations(query)


async def _traverse(
    seed_id: str,
    relation_path: list[str],
    limit: int = 50,
) -> list[dict]:
    """Step 3 — execute whitelisted traversal Cypher."""
    return await execute_read(
        QUERIES["traverse_supply_network"],
        seed_id=seed_id,
        allowed_relations=relation_path,
        limit=limit,
    )


async def kg_agent_node(state: AgentState) -> AgentState:
    """Think-on-Graph: entity extract → relation select → traverse → self-correct."""
    _msgs = state.get("messages") or []
    _last = _msgs[-1] if _msgs else None
    query: str = (
        (
            _last.content
            if hasattr(_last, "content")  # LangChain message object
            else (_last.get("content", "") if isinstance(_last, dict) else "")
        )
        if _last
        else ""
    )

    # Step 1: entity extraction
    extraction = await _extract_entities(query)
    entities = extraction.entities

    if not entities:
        # No entities found — use first supplier as default seed for generic traversal queries
        logger.info("KG agent: no entities found; defaulting to TQ-Electronics seed")
        entities = ["TQ-Electronics GmbH"]

    # Step 2: relation selection
    relation_result = await _select_relations(query, entities)
    relation_path = relation_result.relation_path

    # Step 3: traversal from first seed entity
    seed_id = entities[0]
    rows = await _traverse(seed_id, relation_path)

    # Step 4: self-correction — if empty, retry with PROVIDES + SUPPLIED_BY fallback
    if not rows:
        logger.info(
            "KG traversal returned 0 rows for seed=%r path=%r; retrying with SUPPLIED_BY/PROVIDES",
            seed_id,
            relation_path,
        )
        rows = await _traverse(seed_id, ["PROVIDES", "SUPPLIED_BY", "USED_IN"])

    # Step 5: also try remaining entities as seeds if still empty
    if not rows:
        for alt_seed in entities[1:]:
            rows = await _traverse(alt_seed, ["PROVIDES", "SUPPLIED_BY", "USED_IN"])
            if rows:
                seed_id = alt_seed
                break

    # Convert flat rows to vis-network compatible nodes/edges
    nodes_seen: dict[str, dict] = {}
    edges: list[dict] = []
    label_color = {
        "Supplier": "#1e3a5f",
        "Component": "#1a3a20",
        "Product": "#3a1a2f",
        "Warehouse": "#3a2a10",
        "Contract": "#2a1a3a",
    }
    for row in rows:
        from_name = str(row.get("from_name", "?"))
        to_name = str(row.get("to_name", "?"))
        from_label = row.get("from_label", "Node")
        to_label = row.get("to_label", "Node")
        rel_type = row.get("rel_type", "REL")

        if from_name not in nodes_seen:
            nodes_seen[from_name] = {
                "id": from_name,
                "label": from_name,
                "group": from_label,
                "title": f"{from_label}: {from_name}",
                "color": {"background": label_color.get(from_label, "#1e293b"),
                          "border": "#3b82f6"},
            }
        if to_name not in nodes_seen:
            nodes_seen[to_name] = {
                "id": to_name,
                "label": to_name,
                "group": to_label,
                "title": f"{to_label}: {to_name}",
                "color": {"background": label_color.get(to_label, "#1e293b"),
                          "border": "#3b82f6"},
            }
        edges.append({
            "from": from_name,
            "to": to_name,
            "label": rel_type,
            "arrows": "to",
        })

    subgraph: dict = {
        "nodes": list(nodes_seen.values()),
        "edges": edges,
        "paths": rows,
        "seed_entity": seed_id,
        "relation_path": relation_path,
    }

    return {
        **state,
        "kg_entities": entities,
        "kg_relation_path": relation_path,
        "kg_subgraph": subgraph,
    }
