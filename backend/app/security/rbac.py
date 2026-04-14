"""Role-Based Access Control (Blueprint §2.4).

Stage 4 implementation.
"""

from __future__ import annotations

from enum import StrEnum


class Role(StrEnum):
    VIEWER = "viewer"  # read-only queries
    ANALYST = "analyst"  # queries + solver invocations
    ADMIN = "admin"  # full access including KG writes


# Map role → allowed intents
ROLE_PERMISSIONS: dict[Role, frozenset[str]] = {
    Role.VIEWER: frozenset({"kg_query", "contract_query"}),
    Role.ANALYST: frozenset(
        {
            "kg_query",
            "contract_query",
            "mcnf_solve",
            "disruption_resource",
            "meio_optimize",
            "bullwhip_analyze",
            "jsp_schedule",
            "vrp_route",
            "robust_allocate",
            "multi_step",
        }
    ),
    Role.ADMIN: frozenset({"*"}),
}


def is_allowed(role: Role, intent: str) -> bool:
    perms = ROLE_PERMISSIONS.get(role, frozenset())
    return "*" in perms or intent in perms
