"""Whitelisted Cypher query templates (Blueprint §2.4, §4.3.4).

All queries are parameterised. No Cypher is constructed from raw LLM output.
Stage 4 implementation.
"""

from __future__ import annotations

# Whitelisted query registry: maps query_type -> parameterised Cypher template
QUERIES: dict[str, str] = {
    "traverse_supply_network": """
        MATCH path = (s {id: $seed_id})-[r*1..$max_depth]->(n)
        WHERE type(r[-1]) IN $allowed_relations
        RETURN path LIMIT $limit
    """,
    "find_affected_products": """
        MATCH (s:Supplier {id: $supplier_id})-[:PROVIDES]->(c:Component)-[:USED_IN]->(p:Product)
        RETURN s, c, p
    """,
    "get_supplier_alternatives": """
        MATCH (s:Supplier)-[r:PROVIDES]->(c:Component {id: $component_id})
        RETURN s, r, c ORDER BY r.cost ASC LIMIT 10
    """,
}
