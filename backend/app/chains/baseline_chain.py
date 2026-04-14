"""Baseline single-tool LangChain chain — Stage 3 PoC (Blueprint §3.3).

Architecture (Stage 3 only — no multi-agent orchestration):

    User query
        → SystemMessage + HumanMessage
        → ChatOpenAI.bind_tools([solve_mcnf])
        → Tool call? → execute solve_mcnf (OR-Tools GLOP)
                     → ToolMessage → ChatOpenAI (synthesis)
        → WsResponse

This module is REPLACED by the LangGraph orchestrator in Stage 4.
The outer interface (``run_baseline_chain``) is preserved so
``routes_chat.py`` requires no change during Stage 4 migration.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI

from app.api.schemas import Arc, Commodity, SolveMcnfInput, WsResponse
from app.config import get_settings
from app.solvers.mcnf import solve_mcnf

logger = logging.getLogger(__name__)

_MAX_ITERS = 5  # hard cap on agentic loop iterations

_SYSTEM_PROMPT = """\
You are an Agentic ERP Supply Chain Copilot for AdventureWorks Cycles, \
a fictional bicycle manufacturer. You assist supply-chain and operations managers.

AdventureWorks supply network (summary):
• Tier 1 suppliers : TQ-Electronics, Fabrikam Bearings, Lucerne Metals, \
Contoso Polymers, AWC Direct
• Tier 2 suppliers : TQ-Sub, Hanover Steel, Shanghai Polymers, \
Pacific Rubber, Nordic Precision
• Tier 3 suppliers : Australian Iron Ore, Canadian Aluminum, \
Borracha Brasileira, Korea Semiconductor
• Key components   : Ball Bearings (Fabrikam), Gear Control Unit (TQ-Electronics), \
Aluminum Frame Tube (Lucerne), Polymer Grip (Contoso), Brake Assembly (AWC)
• Products         : Mountain Bike Pro, Road Bike Elite, City Bike Standard, E-Bike Ultra
• Distribution     : Seattle DC-001, Amsterdam DC-002, Tokyo DC-003

Available tool:
• solve_mcnf — Min-Cost Network Flow via OR-Tools GLOP. Call this when the user asks \
about optimal flow routing, supply re-routing after disruptions, or logistics cost \
minimisation. Extract nodes, arcs (from/to/capacity/cost_per_unit) and commodities \
(source/sink/demand) from the context and query.

Respond concisely. If solver inputs cannot be inferred, ask the user for the \
missing values before calling the tool."""


# ---------------------------------------------------------------------------
# LangChain tool — wraps solve_mcnf with Pydantic input validation
# ---------------------------------------------------------------------------


def _make_solve_mcnf_tool() -> StructuredTool:
    """Build the solve_mcnf StructuredTool with Pydantic V2 input validation.

    Validation happens at the API boundary (tool invocation) before any
    OR-Tools code runs, consistent with the Blueprint Pydantic scope rule.
    """

    def _validated_solve(
        nodes: list[str],
        arcs: list[Arc],
        commodities: list[Commodity],
    ) -> dict:
        """Validate inputs then delegate to the GLOP LP solver."""
        arc_dicts = [a.model_dump(by_alias=True) for a in arcs]
        com_dicts = [c.model_dump() for c in commodities]
        return solve_mcnf(nodes, arc_dicts, com_dicts)

    return StructuredTool.from_function(
        func=_validated_solve,
        name="solve_mcnf",
        description=(
            "Min-Cost Network Flow via OR-Tools GLOP. "
            "Use for optimal routing, supply re-routing after disruptions, "
            "and logistics cost minimisation across the supply network. "
            "Arguments: nodes (list[str]), "
            "arcs (list[{from, to, capacity, cost_per_unit}]), "
            "commodities (list[{source, sink, demand}])."
        ),
        args_schema=SolveMcnfInput,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_baseline_chain(query: str) -> WsResponse:
    """Execute the Stage 3 single-tool agentic chain.

    Runs up to ``_MAX_ITERS`` LLM calls.  Returns a ``WsResponse`` containing
    the final assistant message, the tool invoked (if any), and the raw solver
    result (if the tool was called).

    Args:
        query: Raw user message text.

    Returns:
        WsResponse with role="assistant".
    """
    settings = get_settings()

    llm = ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.github_token,  # type: ignore[arg-type]
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    tool = _make_solve_mcnf_tool()
    llm_with_tools = llm.bind_tools([tool])

    messages: list = [SystemMessage(_SYSTEM_PROMPT), HumanMessage(query)]
    tool_used: str | None = None
    solver_result: dict | None = None

    for iteration in range(_MAX_ITERS):
        logger.debug("Chain iteration %d", iteration)
        ai_msg: AIMessage = await llm_with_tools.ainvoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            # LLM produced a plain text response — we are done.
            return WsResponse(
                content=ai_msg.content or "(no response)",
                tool_used=tool_used,
                solver_result=solver_result,
            )

        # Execute each tool call in the response.
        for tc in ai_msg.tool_calls:
            tc_name: str = tc["name"]
            tc_args: dict = tc["args"]
            tc_id: str = tc["id"]
            logger.info("Tool call: %s  args=%s", tc_name, tc_args)

            if tc_name == "solve_mcnf":
                result = tool.invoke(tc_args)
                solver_result = result
                tool_used = "solve_mcnf"
            else:
                result = {"error": f"Unknown tool requested: {tc_name}"}
                logger.warning("Unknown tool: %s", tc_name)

            messages.append(
                ToolMessage(
                    content=json.dumps(result),
                    tool_call_id=tc_id,
                )
            )

    # Exceeded max iterations — return last available AI content.
    last_ai = next(
        (m for m in reversed(messages) if isinstance(m, AIMessage)),
        None,
    )
    return WsResponse(
        content=(
            last_ai.content
            if last_ai and last_ai.content
            else "Max iteration limit reached without a final response."
        ),
        tool_used=tool_used,
        solver_result=solver_result,
    )
