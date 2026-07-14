"""Orchestrator hub agent — LangGraph StateGraph (Blueprint §4.1).

Graph topology (§4.1):
    classify → {kg_agent, contract_agent, solver_dispatch}
             → human_gate (conditional)
             → synthesize → END

Intent routing (§4.2):
    contract_query               → contract_agent
    mcnf_solve | meio_optimize   → solver_dispatch (directly)
    bullwhip_analyze | jsp_schedule | vrp_route | robust_allocate → solver_dispatch
    disruption_resource | multi_step | kg_query | unknown → kg_agent
"""

from __future__ import annotations

import json
import logging
import uuid

import redis.asyncio as aioredis
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from app.agents.contract_agent import contract_agent_node
from app.agents.graph_state import AgentState
from app.agents.kg_agent import kg_agent_node
from app.agents.llm_fallback import with_quota_fallback
from app.api.schemas import (
    AnalyzeBullwhipInput,
    IntentClassification,
    SolveDisruptionInput,
    SolveJspInput,
    SolveMcnfInput,
    SolveMeioInput,
    SolveRobustInput,
    SolveVrpInput,
    WsResponse,
)
from app.config import get_settings
from app.solvers.bullwhip import analyze_bullwhip
from app.solvers.disruption import solve_disruption
from app.solvers.jsp import solve_jsp
from app.solvers.mcnf import solve_mcnf
from app.solvers.meio_gsm import solve_meio_gsm
from app.solvers.robust_minmax import solve_robust_minmax
from app.solvers.vrp import solve_vrp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis client (lazy, shared) for HiTL pending-decision store
# ---------------------------------------------------------------------------
_REDIS: aioredis.Redis | None = None  # type: ignore[type-arg]

_HITL_TTL_SECONDS = 86_400  # pending decisions expire after 24 h


def _get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    global _REDIS
    if _REDIS is None:
        _REDIS = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _REDIS


# Intents that route directly to solver_dispatch without visiting kg_agent first
_SOLVER_DIRECT_INTENTS = frozenset(
    {
        "mcnf_solve",
        "meio_optimize",
        "bullwhip_analyze",
        "jsp_schedule",
        "vrp_route",
        "robust_allocate",
    }
)

_INTENT_SYSTEM = """\
You are an intent classifier for a supply-chain ERP copilot.
Classify the user query into exactly ONE of these 10 bounded-context intents:
  mcnf_solve, jsp_schedule, vrp_route, robust_allocate, meio_optimize,
  bullwhip_analyze, disruption_resource, kg_query, contract_query, multi_step.

Definitions:
  mcnf_solve         — minimum cost network flow / routing / capacity problem
  jsp_schedule       — job-shop or production scheduling
  vrp_route          — vehicle routing / last-mile delivery
  robust_allocate    — robust or worst-case resource allocation
  meio_optimize      — multi-echelon inventory optimisation
  bullwhip_analyze   — bullwhip effect / demand amplification analysis
  disruption_resource— supply disruption assessment without explicit solver
  kg_query           — pure knowledge-graph query (supplier, component, product graph)
  contract_query     — contract / procurement document search
  multi_step         — requires multiple of the above in sequence

Set intent_confidence to 0.0–1.0. Set ddd_context to one of:
  "visibility" | "inventory" | "compliance" | "sourcing" | "logistics"."""

_SYNTH_SYSTEM = """\
You are a senior supply-chain analyst composing a clear, actionable response.
Combine the knowledge-graph findings, solver results, and contract excerpts
provided in the user message into one coherent, well-structured answer.
Be concise but complete. Cite specific numbers from solver outputs where relevant.

Approval rules — follow these EXACTLY:
- Mention human approval ONLY if the context contains a line starting with
  "NOTE:" that talks about manager approval. Follow that NOTE's instruction:
  state clearly at the top whether the action is awaiting approval, was
  APPROVED, or was REJECTED.
- If there is NO such NOTE, the decision is below the approval threshold and
  needs no sign-off: do NOT mention approval, cost thresholds, managers, or
  review processes at all — not even to say approval is unnecessary."""


def _make_llm(max_tokens: int = 1024) -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        model=s.llm_model,
        base_url=s.llm_base_url,
        api_key=s.github_token,  # type: ignore[arg-type]
        temperature=0.0,
        max_tokens=max_tokens,  # type: ignore[call-arg]
    )


# ---------------------------------------------------------------------------
# Node: classify_intent
# ---------------------------------------------------------------------------


def _msg_content(msg: object) -> str:
    """Extract text content from a LangChain message object or plain dict."""
    if hasattr(msg, "content"):
        return str(msg.content)  # type: ignore[union-attr]
    if isinstance(msg, dict):
        return msg.get("content", "")
    return ""


async def llm_classify_intent(query: str) -> IntentClassification:
    """LLM intent classification.

    Prefers the DSPy-compiled program when its artifact exists (see
    ``dspy_classifier.py`` / ``scripts/compile_intent_classifier.py``);
    otherwise runs the zero-shot structured-output baseline below.

    Raises on LLM failure — callers decide the fallback policy.  Exposed
    separately from ``classify_intent`` so the live eval harness can measure
    the actual model (not the keyword fallback that masks its failures).
    """
    from app.agents import dspy_classifier

    compiled = await dspy_classifier.classify(query)
    if compiled is not None:
        return compiled

    llm = _make_llm(max_tokens=512)
    structured = with_quota_fallback(
        llm.with_structured_output(IntentClassification),
        max_tokens=512,
        structured_schema=IntentClassification,
    )
    result: IntentClassification = await structured.ainvoke(  # type: ignore[assignment]
        [
            SystemMessage(_INTENT_SYSTEM),
            HumanMessage(f"Query: {query}"),
        ]
    )
    return result


async def classify_intent(state: AgentState) -> AgentState:
    """Classify user query into one of 10 bounded-context intents (§4.2)."""
    messages = state.get("messages") or []
    query: str = _msg_content(messages[-1]) if messages else ""

    try:
        result = await llm_classify_intent(query)
        s = get_settings()
        intent = result.intent
        if result.intent_confidence < s.intent_confidence_threshold:
            intent = "unclear"
        return {
            **state,
            "intent": intent,
            "intent_confidence": result.intent_confidence,
            "ddd_context": result.ddd_context,
        }
    except Exception as exc:
        logger.warning("classify_intent failed: %s — falling back to keyword classifier", exc)
        intent, confidence, ddd = _keyword_classify(query)
        s = get_settings()
        if confidence < s.intent_confidence_threshold:
            intent = "unclear"
            confidence = 0.0
        return {
            **state,
            "intent": intent,
            "intent_confidence": confidence,
            "ddd_context": ddd,
        }


# Keyword-based fallback classifier — used when the LLM is unavailable (rate-limit, 429, etc.)
# Ordered from most-specific to least-specific.
_KEYWORD_RULES: list[tuple[frozenset[str], str, str]] = [
    # (required_keywords, intent, ddd_context)
    (frozenset({"arc", "cost_per_unit"}), "mcnf_solve", "logistics"),
    (frozenset({"route", "units", "capacity"}), "mcnf_solve", "logistics"),
    (frozenset({"route", "units", "demand"}), "mcnf_solve", "logistics"),
    (frozenset({"minimum cost", "network flow"}), "mcnf_solve", "logistics"),
    (frozenset({"mcnf"}), "mcnf_solve", "logistics"),
    (frozenset({"vehicle", "depot"}), "vrp_route", "logistics"),
    (frozenset({"vrp"}), "vrp_route", "logistics"),
    (frozenset({"schedule", "job", "machine"}), "jsp_schedule", "visibility"),
    (frozenset({"inventory", "echelon"}), "meio_optimize", "inventory"),
    (frozenset({"reorder", "safety stock"}), "meio_optimize", "inventory"),
    (frozenset({"bullwhip", "demand amplification"}), "bullwhip_analyze", "visibility"),
    (frozenset({"robust"}), "robust_allocate", "sourcing"),
    (frozenset({"allocate", "omega"}), "robust_allocate", "sourcing"),
    (frozenset({"allocate", "uncertainty"}), "robust_allocate", "sourcing"),
    (frozenset({"contract", "clause"}), "contract_query", "compliance"),
    (frozenset({"force majeure"}), "contract_query", "compliance"),
    (frozenset({"payment terms"}), "contract_query", "compliance"),
    (frozenset({"payment", "supplier"}), "contract_query", "compliance"),
    (frozenset({"liability"}), "contract_query", "compliance"),
    (frozenset({"warranty"}), "contract_query", "compliance"),
    # KG traversal queries
    (frozenset({"traverse"}), "kg_query", "visibility"),
    (frozenset({"visualis"}), "kg_query", "visibility"),
    (frozenset({"visualiz"}), "kg_query", "visibility"),
    (frozenset({"supply network"}), "kg_query", "sourcing"),
    (frozenset({"supplier network"}), "kg_query", "sourcing"),
    (frozenset({"supply chain", "relationship"}), "kg_query", "sourcing"),
    (frozenset({"graph", "relationship"}), "kg_query", "sourcing"),
    (frozenset({"supplier", "tier"}), "kg_query", "sourcing"),
    (frozenset({"network", "supplier"}), "kg_query", "sourcing"),
    (frozenset({"disruption", "alternative"}), "disruption_resource", "visibility"),
]


def _keyword_classify(query: str) -> tuple[str, float, str]:
    """Deterministic keyword-based intent classification (LLM fallback)."""
    q = query.lower()
    for keywords, intent, ddd in _KEYWORD_RULES:
        if all(kw in q for kw in keywords):
            logger.info("keyword_classify: intent=%s (matched %s)", intent, keywords)
            return intent, 0.75, ddd
    return "kg_query", 0.5, "visibility"


# ---------------------------------------------------------------------------
# Conditional edge: route_by_intent
# ---------------------------------------------------------------------------


def route_by_intent(state: AgentState) -> str:
    """Map intent → downstream node name."""
    intent = state.get("intent") or ""
    if intent == "contract_query":
        return "contract_query"
    if intent in _SOLVER_DIRECT_INTENTS:
        return "or_solve"
    # kg_query, disruption_resource, multi_step, unclear, unknown → kg_agent
    return "kg_query"


# ---------------------------------------------------------------------------
# Node: solver_dispatch_node
# ---------------------------------------------------------------------------


async def _extract_mcnf_params(query: str) -> SolveMcnfInput | None:
    """Use structured LLM output to extract MCNF solver parameters from the query."""
    try:
        llm = _make_llm(max_tokens=512)
        structured = with_quota_fallback(
            llm.with_structured_output(SolveMcnfInput),
            max_tokens=512,
            structured_schema=SolveMcnfInput,
        )
        result: SolveMcnfInput = await structured.ainvoke(  # type: ignore[assignment]
            [
                SystemMessage(
                    "Extract minimum-cost-network-flow parameters from the supply-chain "
                    "query.  Return arcs with (source, target, capacity, cost), a list of "
                    "commodities (origin, destination, demand), and node names."
                ),
                HumanMessage(f"Query: {query}"),
            ]
        )
        return result
    except Exception as exc:
        logger.warning("MCNF param extraction failed: %s — trying regex fallback", exc)
        return _regex_extract_mcnf_params(query)


def _regex_extract_mcnf_params(query: str) -> SolveMcnfInput | None:
    """Regex-based MCNF parameter extractor — used when the LLM is unavailable.

    Parses patterns produced by the canonical test query:
      "Route <demand> units from <source> (<node_A>) to <sink> (<node_B>).
       Arc capacity <cap>, cost_per_unit=<cost>. Demand at <node_B> is <demand>."
    """
    import re

    q = query

    # Extract node IDs from parentheses, e.g. "factory (node A)"
    node_ids = re.findall(r"\(([^)]+)\)", q)

    # Also try bare labels like "node A", "node B", "factory", "Tokyo DC"
    if len(node_ids) < 2:
        # fall back: split on "from ... to ..."
        m = re.search(
            r"from\s+([A-Za-z0-9_ ]+?)\s+to\s+([A-Za-z0-9_ ]+?)[\.\,\s]",
            q,
            re.IGNORECASE,
        )
        if m:
            node_ids = [m.group(1).strip(), m.group(2).strip()]

    if len(node_ids) < 2:
        logger.warning("regex_extract_mcnf_params: could not parse 2 node IDs from query")
        return None

    src, snk = node_ids[0], node_ids[1]

    # capacity
    cap_m = re.search(r"capacity\s*[=:]?\s*([\d,]+(?:\.\d+)?)", q, re.IGNORECASE)
    capacity = float(cap_m.group(1).replace(",", "")) if cap_m else 10_000.0

    # cost_per_unit
    cpu_m = re.search(r"cost_per_unit\s*=?\s*\$?([\d,]+(?:\.\d+)?)", q, re.IGNORECASE)
    cost_per_unit = float(cpu_m.group(1).replace(",", "")) if cpu_m else 1.0

    # demand — first integer/float followed by "units"
    dem_m = re.search(r"([\d,]+(?:\.\d+)?)\s+units", q, re.IGNORECASE)
    demand = float(dem_m.group(1).replace(",", "")) if dem_m else 1.0

    try:
        from app.api.schemas import Arc, Commodity  # local import to avoid circulars

        return SolveMcnfInput(
            nodes=[src, snk],
            arcs=[
                Arc.model_validate(
                    {
                        "from": src,
                        "to": snk,
                        "capacity": capacity,
                        "cost_per_unit": cost_per_unit,
                    }
                )
            ],
            commodities=[Commodity(source=src, sink=snk, demand=demand)],
        )
    except Exception as exc:
        logger.warning("regex_extract_mcnf_params: schema validation failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Generic structured parameter extraction — one schema per solver intent.
# Mirrors _extract_mcnf_params; on LLM failure returns None and the dispatch
# falls back to empty defaults (solvers handle empty inputs gracefully).
# ---------------------------------------------------------------------------

_PARAM_EXTRACT_INSTRUCTIONS: dict[str, str] = {
    "jsp_schedule": (
        "Extract job-shop scheduling parameters from the supply-chain query. "
        "Each job is an ordered list of operations; each operation names the "
        "machine (integer ID) and its processing duration."
    ),
    "vrp_route": (
        "Extract capacitated vehicle-routing parameters from the query: all "
        "locations with (id, x, y, demand) — the depot has demand 0 — plus the "
        "depot index, vehicle capacity, and fleet size."
    ),
    "robust_allocate": (
        "Extract robust supplier-allocation parameters from the query: each "
        "supplier's mean unit cost, cost uncertainty, and capacity, plus total "
        "demand and the robustness budget omega (default 1.0 if unstated)."
    ),
    "meio_optimize": (
        "Extract multi-echelon inventory parameters from the query: for each "
        "stage the holding cost, demand standard deviation, lead time, and the "
        "indices of its upstream predecessor stages, plus the target service "
        "level (default 0.95 if unstated)."
    ),
    "bullwhip_analyze": (
        "Extract bullwhip-analysis parameters from the query: the time-ordered "
        "demand series and, if stated, lead time, forecast moving-average "
        "window, and number of echelons."
    ),
    "disruption_resource": (
        "Extract disruption re-allocation parameters from the query: affected "
        "component IDs, alternative suppliers with (id, component, cost, "
        "capacity), and the demanded quantity per component."
    ),
}


async def _extract_solver_params(
    query: str,
    schema: type[BaseModel],
    instructions: str,
) -> BaseModel | None:
    """Structured-output extraction of solver parameters; None on failure."""
    try:
        llm = _make_llm(max_tokens=1024)
        structured = with_quota_fallback(
            llm.with_structured_output(schema),
            max_tokens=1024,
            structured_schema=schema,
        )
        result: BaseModel = await structured.ainvoke(  # type: ignore[assignment]
            [
                SystemMessage(instructions),
                HumanMessage(f"Query: {query}"),
            ]
        )
        return result
    except Exception as exc:
        logger.warning("param extraction failed for %s: %s", schema.__name__, exc)
        return None


async def solver_dispatch_node(
    state: AgentState,
    config: RunnableConfig | None = None,
) -> AgentState:
    """Dispatch intent to the correct solver (Blueprint §4.6).

    Solver parameters are extracted from the query via structured LLM output;
    if extraction fails the solver is called with empty defaults, which every
    solver answers with a trivial OPTIMAL result.

    When the result crosses the HiTL cost threshold the pending decision is
    stored in Redis under ``decision_id`` — which equals the LangGraph
    ``thread_id`` so the approval endpoint can resume the paused graph.
    """
    intent = state.get("intent") or ""
    messages = state.get("messages") or []
    query: str = _msg_content(messages[-1]) if messages else ""
    solver_out: dict = {}

    def _instructions(name: str) -> str:
        return _PARAM_EXTRACT_INSTRUCTIONS[name]

    try:
        if intent == "mcnf_solve":
            params = await _extract_mcnf_params(query)
            if params:
                arcs_raw = [a.model_dump(by_alias=True) for a in params.arcs]
                commodities_raw = [c.model_dump() for c in params.commodities]
                solver_out = solve_mcnf(
                    nodes=params.nodes,
                    arcs=arcs_raw,
                    commodities=commodities_raw,
                )
            else:
                solver_out = {"status": "param_extraction_failed"}

        elif intent == "jsp_schedule":
            p = await _extract_solver_params(query, SolveJspInput, _instructions(intent))
            jobs = [j.model_dump() for j in p.jobs] if isinstance(p, SolveJspInput) else []
            solver_out = solve_jsp(jobs=jobs)

        elif intent == "vrp_route":
            p = await _extract_solver_params(query, SolveVrpInput, _instructions(intent))
            if isinstance(p, SolveVrpInput):
                solver_out = solve_vrp(
                    depot=p.depot,
                    locations=[loc.model_dump() for loc in p.locations],
                    vehicle_capacity=p.vehicle_capacity,
                    num_vehicles=p.num_vehicles,
                )
            else:
                solver_out = solve_vrp(depot=0, locations=[], vehicle_capacity=1000, num_vehicles=1)

        elif intent == "robust_allocate":
            p = await _extract_solver_params(query, SolveRobustInput, _instructions(intent))
            if isinstance(p, SolveRobustInput):
                solver_out = solve_robust_minmax(
                    suppliers=[s.model_dump() for s in p.suppliers],
                    demand=p.demand,
                    omega=p.omega,
                )
            else:
                solver_out = solve_robust_minmax(suppliers=[], demand=0.0, omega=1.0)

        elif intent == "meio_optimize":
            p = await _extract_solver_params(query, SolveMeioInput, _instructions(intent))
            if isinstance(p, SolveMeioInput):
                solver_out = solve_meio_gsm(
                    stages=[st.model_dump() for st in p.stages],
                    service_level=p.service_level,
                )
            else:
                solver_out = solve_meio_gsm(stages=[], service_level=0.95)

        elif intent == "bullwhip_analyze":
            p = await _extract_solver_params(query, AnalyzeBullwhipInput, _instructions(intent))
            if isinstance(p, AnalyzeBullwhipInput):
                solver_out = analyze_bullwhip(
                    demand_series=p.demand_series,
                    lead_time=p.lead_time,
                    forecast_window=p.forecast_window,
                    num_echelons=p.num_echelons,
                )
            else:
                solver_out = analyze_bullwhip(
                    demand_series=[], lead_time=1, forecast_window=4, num_echelons=2
                )

        elif intent == "disruption_resource":
            p = await _extract_solver_params(query, SolveDisruptionInput, _instructions(intent))
            if isinstance(p, SolveDisruptionInput):
                solver_out = solve_disruption(
                    affected_components=p.affected_components,
                    alt_suppliers=[s.model_dump() for s in p.alt_suppliers],
                    demands=[d.model_dump() for d in p.demands],
                )
            else:
                solver_out = solve_disruption(affected_components=[], alt_suppliers=[], demands=[])

        else:
            # Fallback: kg_agent already populated kg_subgraph; nothing to dispatch
            solver_out = {"status": "no_solver_needed", "intent": intent}

    except Exception as exc:
        logger.exception("solver_dispatch_node failed for intent=%s: %s", intent, exc)
        solver_out = {"status": "error", "error": str(exc)}

    # Evaluate HiTL threshold here so check_impact() can route correctly.
    cost = float((solver_out or {}).get("total_cost", 0))
    threshold = get_settings().human_approval_cost_threshold
    needs_approval = cost > threshold
    decision_id: str | None = None

    if needs_approval:
        thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
        decision_id = str(thread_id) if thread_id else str(uuid.uuid4())
        messages = state.get("messages") or []
        query = _msg_content(messages[-1]) if messages else ""
        pending_record = json.dumps(
            {
                "decision_id": decision_id,
                "status": "pending",  # pending | approved | rejected
                "query": query,
                "intent": state.get("intent"),
                "solver_output": solver_out,
                "total_cost": cost,
                "approved_by": None,
                "reason": None,
            }
        )
        try:
            await _get_redis().setex(f"hitl:{decision_id}", _HITL_TTL_SECONDS, pending_record)
            logger.info(
                "solver_dispatch_node: cost=%.2f > threshold=%.2f → decision_id=%s stored in Redis",
                cost,
                threshold,
                decision_id,
            )
        except Exception as redis_exc:
            logger.warning("Redis store failed for decision_id=%s: %s", decision_id, redis_exc)

    return {
        **state,
        "solver_output": solver_out,
        "human_approval_required": needs_approval,
        "decision_id": decision_id,
    }


# ---------------------------------------------------------------------------
# Node: human_approval_gate
# ---------------------------------------------------------------------------


async def human_approval_gate(state: AgentState) -> AgentState:
    """PAUSE the graph until a manager approves or rejects (§4.7).

    ``interrupt()`` checkpoints the run and stops execution — no response is
    synthesized while the decision is pending.  When the manager posts to
    ``POST /api/approve/{decision_id}``, ``resume_orchestrator()`` re-enters
    this node and ``interrupt()`` returns the manager's decision payload.
    """
    solver_out = state.get("solver_output") or {}
    decision = interrupt(
        {
            "decision_id": state.get("decision_id"),
            "intent": state.get("intent"),
            "total_cost": solver_out.get("total_cost", 0),
            "reason": "solver cost exceeds the human-approval threshold",
        }
    )
    approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
    logger.info(
        "human_approval_gate: decision_id=%s resumed → %s",
        state.get("decision_id"),
        "approved" if approved else "rejected",
    )
    return {
        **state,
        "human_approval_required": True,
        "approval_status": "approved" if approved else "rejected",
    }


def check_impact(state: AgentState) -> str:
    return "high_impact" if state.get("human_approval_required") else "low_impact"


# ---------------------------------------------------------------------------
# Node: synthesize_response
# ---------------------------------------------------------------------------


async def synthesize_response(state: AgentState) -> AgentState:
    """Compose final NL response from KG, solver, and RAG outputs (Blueprint §4.1)."""
    messages = state.get("messages") or []
    query: str = _msg_content(messages[-1]) if messages else ""

    context_parts: list[str] = [f"Original query: {query}"]

    kg_sub = state.get("kg_subgraph")
    if kg_sub and kg_sub.get("nodes"):
        context_parts.append(
            f"Knowledge graph findings ({len(kg_sub['nodes'])} records): "
            + json.dumps(kg_sub["nodes"][:5], default=str)
        )

    solver_out = state.get("solver_output")
    if solver_out:
        context_parts.append(f"Solver result: {json.dumps(solver_out, default=str)[:800]}")

    rag_docs = state.get("rag_documents")
    rag_eval = state.get("rag_evaluation")
    if rag_docs and rag_eval in ("correct", "ambiguous"):
        top_chunk = rag_docs[0].get("chunk_text", "")[:500]
        context_parts.append(f"Contract excerpt (relevance={rag_eval}): {top_chunk}")

    approval_status = state.get("approval_status")
    if state.get("human_approval_required"):
        if approval_status == "approved":
            context_parts.append(
                "NOTE: A supply-chain manager APPROVED this high-cost action. "
                "State clearly at the top that the decision was approved and may proceed."
            )
        elif approval_status == "rejected":
            context_parts.append(
                "NOTE: A supply-chain manager REJECTED this high-cost action. "
                "Do NOT present the plan as actionable — state the rejection clearly "
                "and suggest lower-cost alternatives or next steps."
            )
        else:
            context_parts.append(
                "NOTE: This action requires human approval due to high cost impact."
            )

    context = "\n\n".join(context_parts)

    degraded: str | None = None
    try:
        llm = with_quota_fallback(_make_llm(max_tokens=1024), max_tokens=1024)
        response = await llm.ainvoke(
            [
                SystemMessage(_SYNTH_SYSTEM),
                HumanMessage(context),
            ]
        )
        final_response: str = response.content  # type: ignore[assignment]
    except Exception as exc:
        logger.warning("synthesize_response LLM failed: %s", exc)
        # Mark the state so this degraded template answer is never cached —
        # otherwise a transient LLM outage (429/401) poisons the semantic
        # cache with fallback text for the full TTL.
        degraded = f"llm_synthesis_failed: {exc}"
        status = (solver_out or {}).get("status", "N/A")
        cost = (solver_out or {}).get("total_cost", 0)
        if state.get("human_approval_required"):
            if approval_status == "approved":
                final_response = (
                    f"✅ APPROVED — a supply-chain manager approved this decision "
                    f"(total cost: ${cost:,.2f}). Solver status: {status}"
                )
            elif approval_status == "rejected":
                final_response = (
                    f"❌ REJECTED — a supply-chain manager rejected this decision "
                    f"(total cost: ${cost:,.2f}). The plan will not be executed."
                )
            else:
                final_response = (
                    f"⚠️ HUMAN APPROVAL REQUIRED\n"
                    f"This routing decision (total cost: ${cost:,.2f}) exceeds the "
                    f"$10,000 threshold and must be reviewed by a supply-chain manager "
                    f"before execution.\n\nSolver status: {status}"
                )
        else:
            final_response = f"Analysis complete. Solver status: {status}"

    return {**state, "final_response": final_response, "error": degraded or state.get("error")}


# ---------------------------------------------------------------------------
# Graph compilation
# ---------------------------------------------------------------------------

_GRAPH: object | None = None  # compiled CompiledStateGraph


def _build_graph() -> object:
    """Compile the LangGraph StateGraph (called once, lazy)."""
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("classify", classify_intent)
    builder.add_node("kg_agent", kg_agent_node)
    builder.add_node("contract_agent", contract_agent_node)
    builder.add_node("solver_dispatch", solver_dispatch_node)
    builder.add_node("human_gate", human_approval_gate)
    builder.add_node("synthesize", synthesize_response)

    builder.set_entry_point("classify")

    builder.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "kg_query": "kg_agent",
            "contract_query": "contract_agent",
            "or_solve": "solver_dispatch",
        },
    )

    # kg_agent always passes through to solver_dispatch
    # (for pure kg_query intents the solver_dispatch returns immediately)
    builder.add_edge("kg_agent", "solver_dispatch")

    # contract_agent → synthesize (no solver for contract queries)
    builder.add_edge("contract_agent", "synthesize")

    # solver_dispatch → human_gate or synthesize based on cost impact
    builder.add_conditional_edges(
        "solver_dispatch",
        check_impact,
        {
            "high_impact": "human_gate",
            "low_impact": "synthesize",
        },
    )

    builder.add_edge("human_gate", "synthesize")
    builder.add_edge("synthesize", END)

    # MemorySaver checkpointer lets human_gate interrupt() and later resume.
    # In-process only: a restart drops paused runs (the Redis record survives
    # for audit, but the graph cannot be resumed after a restart).
    return builder.compile(checkpointer=MemorySaver())


def _get_graph() -> object:
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _build_graph()
    return _GRAPH


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _state_to_response(state: AgentState) -> WsResponse:
    """Map a terminal AgentState onto the WebSocket response schema."""
    return WsResponse(
        role="assistant",
        content=state.get("final_response") or "Processing complete.",
        intent=state.get("intent"),
        intent_confidence=state.get("intent_confidence"),
        rag_documents=state.get("rag_documents"),
        solver_result=state.get("solver_output"),
        human_approval_required=bool(state.get("human_approval_required")),
        decision_id=state.get("decision_id"),
        kg_subgraph=state.get("kg_subgraph"),
        # Degraded-path marker (e.g. llm_synthesis_failed) — routes_chat uses
        # this to keep fallback answers out of the semantic cache.
        error=state.get("error"),
    )


async def run_orchestrator(query: str) -> WsResponse:
    """Entry point called from routes_chat.py WebSocket handler.

    Builds initial AgentState and invokes the compiled LangGraph under a fresh
    ``thread_id``.  If the run pauses at the human-approval gate, a pending
    response (with ``decision_id`` for the Approve/Reject banner) is returned
    instead of a synthesized answer — the graph stays checkpointed until
    ``resume_orchestrator()`` delivers the manager's decision.
    """
    initial_state: AgentState = {  # type: ignore[typeddict-item]
        "messages": [{"role": "user", "content": query}],
        "intent": None,
        "intent_confidence": 0.0,
        "ddd_context": None,
        "solver_input": None,
        "solver_output": None,
        "kg_subgraph": None,
        "kg_entities": None,
        "kg_relation_path": None,
        "rag_documents": None,
        "rag_evaluation": None,
        "human_approval_required": False,
        "decision_id": None,
        "approval_status": None,
        "final_response": None,
        "error": None,
    }
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    try:
        graph = _get_graph()
        final_state: AgentState = await graph.ainvoke(initial_state, config)  # type: ignore[attr-defined]

        snapshot = await graph.aget_state(config)  # type: ignore[attr-defined]
        if snapshot.next:  # paused at human_gate — no answer synthesized yet
            values: AgentState = snapshot.values
            solver_out = values.get("solver_output") or {}
            cost = float(solver_out.get("total_cost", 0) or 0)
            return WsResponse(
                role="assistant",
                content=(
                    f"⚠️ HUMAN APPROVAL REQUIRED\n"
                    f"This decision (total cost: ${cost:,.2f}) exceeds the approval "
                    f"threshold and is awaiting review by a supply-chain manager. "
                    f"No action will be taken until it is approved or rejected."
                ),
                intent=values.get("intent"),
                intent_confidence=values.get("intent_confidence"),
                solver_result=solver_out,
                human_approval_required=True,
                decision_id=values.get("decision_id"),
                kg_subgraph=values.get("kg_subgraph"),
            )

        return _state_to_response(final_state)

    except Exception as exc:
        logger.exception("run_orchestrator failed for query=%r: %s", query, exc)
        return WsResponse(
            role="assistant",
            content=f"Orchestrator error: {exc}",
            error=str(exc),
        )


async def resume_orchestrator(
    decision_id: str,
    approved: bool,
    approved_by: str | None = None,
    reason: str | None = None,
) -> WsResponse | None:
    """Resume a graph paused at the human-approval gate (§4.7).

    Called by ``POST /api/approve/{decision_id}``.  The ``decision_id`` doubles
    as the LangGraph ``thread_id``.  Returns the synthesized final response,
    or None if no paused run exists for that ID (expired or process restart).
    """
    graph = _get_graph()
    config: RunnableConfig = {"configurable": {"thread_id": decision_id}}

    try:
        snapshot = await graph.aget_state(config)  # type: ignore[attr-defined]
        if not snapshot.next:
            logger.warning("resume_orchestrator: no paused run for decision_id=%s", decision_id)
            return None

        final_state: AgentState = await graph.ainvoke(  # type: ignore[attr-defined]
            Command(resume={"approved": approved, "approved_by": approved_by, "reason": reason}),
            config,
        )
        return _state_to_response(final_state)
    except Exception as exc:
        logger.exception("resume_orchestrator failed for decision_id=%s: %s", decision_id, exc)
        return None
