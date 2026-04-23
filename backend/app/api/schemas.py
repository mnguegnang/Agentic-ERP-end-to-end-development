"""Pydantic API schemas — WebSocket contracts and MCP tool input schemas (Blueprint §3.3, §4.3.1).

Scope (Pydantic boundary rule):
    ✓  API request/response contracts (WsMessage, WsResponse)
    ✓  MCP tool input schemas (Arc, Commodity, SolveMcnfInput)
    ✗  Solver hot-paths (plain dicts passed to OR-Tools)
    ✗  Data-pipeline internals
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# WebSocket message schemas
# ---------------------------------------------------------------------------


class WsMessage(BaseModel):
    """Incoming WebSocket message from the frontend."""

    role: str = Field(default="user", description="Sender role: 'user' | 'assistant'")
    content: str = Field(..., min_length=1, description="Message text content")


class WsResponse(BaseModel):
    """Outgoing WebSocket message to the frontend."""

    role: str = Field(default="assistant")
    content: str
    tool_used: str | None = None
    solver_result: dict | None = None
    intent: str | None = None  # populated by Stage 4 orchestrator
    intent_confidence: float | None = None
    rag_documents: list[dict] | None = None  # populated by Stage 4 CRAG agent
    human_approval_required: bool = False  # True → frontend shows approval dialog
    decision_id: str | None = None  # UUID for the pending HiTL approval stored in Redis
    error: str | None = None  # error message if orchestrator raised an exception


# ---------------------------------------------------------------------------
# MCNF tool input schemas (Blueprint §4.3.1)
# Used by the LangChain StructuredTool and server_ortools MCP server.
# ---------------------------------------------------------------------------


class Arc(BaseModel):
    """Network arc.

    ``from_node`` uses alias ``"from"`` so JSON from the LLM uses the
    canonical field name while Python avoids the reserved keyword.
    """

    model_config = ConfigDict(populate_by_name=True)

    from_node: str = Field(..., alias="from", description="Source node ID")
    to: str = Field(..., description="Destination node ID")
    capacity: float = Field(..., gt=0, description="Arc capacity (units)")
    cost_per_unit: float = Field(..., ge=0, description="Cost per unit of flow")


class Commodity(BaseModel):
    """A source-sink demand pair for MCNF."""

    source: str = Field(..., description="Source node ID")
    sink: str = Field(..., description="Sink node ID")
    demand: float = Field(..., gt=0, description="Units to route from source to sink")


class SolveMcnfInput(BaseModel):
    """Input validation schema for the solve_mcnf tool (Blueprint §4.3.1).

    Constraints:
        * nodes  — at least 2 distinct node IDs
        * arcs   — at least 1 directed arc
        * commodities — at least 1 source/sink demand pair
    """

    nodes: list[str] = Field(
        ..., min_length=2, description="All node IDs in the network"
    )
    arcs: list[Arc] = Field(
        ..., min_length=1, description="Network arcs with capacity and unit cost"
    )
    commodities: list[Commodity] = Field(
        ..., min_length=1, description="Commodities to route (source, sink, demand)"
    )


# ---------------------------------------------------------------------------
# Stage 4: Structured LLM output schemas (Blueprint §4.2, §4.5)
# ---------------------------------------------------------------------------

#: All valid intent labels (Blueprint §4.2 Table)
VALID_INTENTS: frozenset[str] = frozenset(
    {
        "kg_query",
        "mcnf_solve",
        "disruption_resource",
        "meio_optimize",
        "bullwhip_analyze",
        "jsp_schedule",
        "vrp_route",
        "robust_allocate",
        "contract_query",
        "multi_step",
    }
)


class IntentClassification(BaseModel):
    """Structured output for intent classification (Blueprint §4.2).

    Used with ``llm.with_structured_output(IntentClassification)``.
    """

    intent: str = Field(
        ...,
        description=(
            "One of: kg_query, mcnf_solve, disruption_resource, meio_optimize, "
            "bullwhip_analyze, jsp_schedule, vrp_route, robust_allocate, "
            "contract_query, multi_step"
        ),
    )
    intent_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score 0.0–1.0"
    )
    ddd_context: str = Field(
        ...,
        description="DDD bounded context: 'visibility' | 'inventory' | 'compliance'",
    )
    reasoning: str = Field(
        ..., description="One-sentence reasoning for the classification"
    )


class EntityExtractionResult(BaseModel):
    """Structured output for KG entity extraction (Blueprint §4.5)."""

    entities: list[str] = Field(
        ..., description="Supply-chain entity names extracted from the query"
    )
    entity_types: list[str] = Field(
        ...,
        description=(
            "Entity type for each entity: "
            "'Supplier' | 'Component' | 'Product' | 'DistributionCenter'"
        ),
    )


class RelationSelectionResult(BaseModel):
    """Structured output for KG traversal relation path selection (Blueprint §4.5)."""

    relation_path: list[str] = Field(
        ...,
        description=(
            "Ordered list of Neo4j relationship types to traverse. "
            "Valid types: PROVIDES, USED_IN, PROCESSED_AT, SHIPS_TO, BOUND_BY, SUPPLIED_BY"
        ),
    )
    reasoning: str = Field(
        ..., description="One-sentence reasoning for the chosen traversal path"
    )
