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
    kg_subgraph: dict | None = None  # nodes/edges for vis-network graph panel
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

    nodes: list[str] = Field(..., min_length=2, description="All node IDs in the network")
    arcs: list[Arc] = Field(
        ..., min_length=1, description="Network arcs with capacity and unit cost"
    )
    commodities: list[Commodity] = Field(
        ..., min_length=1, description="Commodities to route (source, sink, demand)"
    )


# ---------------------------------------------------------------------------
# Solver input schemas — one per intent, used with structured LLM extraction
# (mirrors SolveMcnfInput; Blueprint §4.3.1 / §5.1.2 parameter-extraction gate)
# ---------------------------------------------------------------------------


class JspOperation(BaseModel):
    """One operation of a job: runs on ``machine`` for ``duration`` time units."""

    machine: int = Field(..., ge=0, description="Machine ID the operation runs on")
    duration: int = Field(..., gt=0, description="Processing time in periods")


class JspJob(BaseModel):
    operations: list[JspOperation] = Field(
        ..., min_length=1, description="Ordered operations of this job (precedence chain)"
    )


class SolveJspInput(BaseModel):
    """Input schema for the solve_jsp tool (job-shop scheduling)."""

    jobs: list[JspJob] = Field(..., min_length=1, description="Jobs to schedule")


class VrpLocation(BaseModel):
    id: str = Field(..., description="Location identifier")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")
    demand: int = Field(..., ge=0, description="Units demanded (0 for the depot)")


class SolveVrpInput(BaseModel):
    """Input schema for the solve_vrp tool (capacitated vehicle routing)."""

    depot: int = Field(default=0, ge=0, description="Index of the depot in locations")
    locations: list[VrpLocation] = Field(
        ..., min_length=1, description="All stops including the depot"
    )
    vehicle_capacity: int = Field(..., gt=0, description="Max load per vehicle")
    num_vehicles: int = Field(..., ge=1, description="Fleet size")


class RobustSupplier(BaseModel):
    cost_mean: float = Field(..., ge=0, description="Nominal unit cost")
    cost_uncertainty: float = Field(..., ge=0, description="Unit-cost uncertainty half-width")
    capacity: float = Field(..., gt=0, description="Maximum units this supplier can provide")


class SolveRobustInput(BaseModel):
    """Input schema for the solve_robust_minmax tool (robust allocation)."""

    suppliers: list[RobustSupplier] = Field(..., min_length=1)
    demand: float = Field(..., gt=0, description="Total demand to fulfil")
    omega: float = Field(default=1.0, ge=0, description="Robustness / uncertainty budget")


class MeioStage(BaseModel):
    holding_cost: float = Field(..., ge=0, description="Per-unit holding cost at this stage")
    demand_std: float = Field(..., ge=0, description="Demand standard deviation")
    lead_time: float = Field(..., ge=0, description="Processing lead time (periods)")
    predecessors: list[int] = Field(
        default_factory=list, description="Indices of upstream stages feeding this one"
    )


class SolveMeioInput(BaseModel):
    """Input schema for the solve_meio_gsm tool (multi-echelon inventory)."""

    stages: list[MeioStage] = Field(..., min_length=1)
    service_level: float = Field(default=0.95, gt=0, lt=1, description="Target fill rate")


class AnalyzeBullwhipInput(BaseModel):
    """Input schema for the analyze_bullwhip tool (demand amplification)."""

    demand_series: list[float] = Field(
        ..., min_length=3, description="Time-ordered end-customer demand observations"
    )
    lead_time: int = Field(default=1, ge=1, description="Replenishment lead time (periods)")
    forecast_window: int = Field(default=4, ge=1, description="Moving-average window (periods)")
    num_echelons: int = Field(default=2, ge=1, description="Number of supply-chain echelons")


class DisruptionSupplier(BaseModel):
    id: str = Field(..., description="Alternative supplier identifier")
    component: str = Field(..., description="Component this supplier can provide")
    cost: float = Field(..., ge=0, description="Per-unit cost from this supplier")
    capacity: float = Field(..., gt=0, description="Maximum units available")


class DisruptionDemand(BaseModel):
    component: str = Field(..., description="Affected component identifier")
    quantity: float = Field(..., gt=0, description="Units required")


class SolveDisruptionInput(BaseModel):
    """Input schema for the solve_disruption tool (re-allocation after disruption)."""

    affected_components: list[str] = Field(..., min_length=1)
    alt_suppliers: list[DisruptionSupplier] = Field(..., min_length=1)
    demands: list[DisruptionDemand] = Field(..., min_length=1)


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
    intent_confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0.0–1.0")
    ddd_context: str = Field(
        ...,
        description="DDD bounded context: 'visibility' | 'inventory' | 'compliance'",
    )
    reasoning: str = Field(..., description="One-sentence reasoning for the classification")


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
    reasoning: str = Field(..., description="One-sentence reasoning for the chosen traversal path")
