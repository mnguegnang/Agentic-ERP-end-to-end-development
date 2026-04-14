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
