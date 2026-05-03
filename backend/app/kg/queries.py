"""Whitelisted Cypher query templates (Blueprint §2.4, §4.3.4).

All queries are parameterised. No Cypher is constructed from raw LLM output.
Stage 4 implementation.
"""

from __future__ import annotations

# Whitelisted query registry: maps query_type -> parameterised Cypher template
QUERIES: dict[str, str] = {
    "traverse_supply_network": """
        MATCH (s)
        WHERE toLower(s.name) CONTAINS toLower($seed_id)
           OR toLower(toString(s.id)) = toLower($seed_id)
        WITH s LIMIT 1
        MATCH path = (s)-[r*1..4]->(n)
        WHERE all(rel IN relationships(path) WHERE type(rel) IN $allowed_relations)
        WITH nodes(path) AS ns, relationships(path) AS rs
        UNWIND range(0, size(rs)-1) AS i
        RETURN
          coalesce(ns[i].name, labels(ns[i])[0] + '-' + toString(ns[i].id))   AS from_name,
          labels(ns[i])[0]                                                     AS from_label,
          type(rs[i])                                                          AS rel_type,
          coalesce(ns[i+1].name, labels(ns[i+1])[0] + '-' + toString(ns[i+1].id)) AS to_name,
          labels(ns[i+1])[0]                                                   AS to_label
        LIMIT $limit
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
