"""Entity and relationship type definitions for the supply-chain KG (Blueprint §2.1.2).

Stage 4 implementation.
"""

from __future__ import annotations

# Node labels
NODE_LABELS = frozenset(
    {
        "Supplier",
        "Component",
        "Product",
        "WorkCenter",
        "DistributionCenter",
        "Contract",
    }
)

# Relationship types
RELATION_TYPES = frozenset(
    {
        "PROVIDES",
        "USED_IN",
        "PROCESSED_AT",
        "SHIPS_TO",
        "BOUND_BY",
        "SUPPLIED_BY",
    }
)
