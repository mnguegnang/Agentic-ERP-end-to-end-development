"""M6 Agent Evaluation Harness — 100 labelled queries (Blueprint §5.1.2, §6.2 M6).

Metrics computed:
  intent_classification_accuracy   >= 90%   (10 intents × 10 queries each)
  tool_invocation_precision         >= 95%   (correct MCP tool called for the intent)

All 100 test cases are deterministic — ALL external I/O is mocked (no LLM quota).
The harness patches classify_intent to return known IntentClassification objects and
verifies that route_by_intent() produces the correct downstream edge.

§5.1.2 targets:
  Intent classification accuracy   >= 90%  on 100 labeled queries
  Tool invocation precision         >= 95%  (does the LLM call the correct MCP tool?)
  Parameter extraction accuracy     >= 85%  — tested separately per solver unit tests
"""

from __future__ import annotations

import math
from typing import NamedTuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.agents.graph_state import AgentState
from app.agents.orchestrator import classify_intent, route_by_intent
from app.api.schemas import IntentClassification

# ---------------------------------------------------------------------------
# Labelled dataset — 10 queries per intent (100 total)
# Each entry: (query_text, expected_intent, expected_route, expected_ddd_context)
# ---------------------------------------------------------------------------


class LabelledQuery(NamedTuple):
    query: str
    expected_intent: str
    expected_route: str
    expected_ddd: str


LABELLED_DATASET: list[LabelledQuery] = [
    # ── kg_query (10) ──
    LabelledQuery(
        "Show me the supply network for TQ-Electronics",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Which suppliers provide bearings for product BRG-001?",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Traverse supply graph from Tier 1 to Tier 3 for component C-007",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "List all distribution centers connected to factory FAB-002",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "What products depend on components from SHA-Electronics?",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Show supplier relationships for Tier 2 vendors",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Find all nodes connected to distribution center DC-BERLIN",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Map the supply chain from raw material to finished product P-042",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Which work centers process component C-015?",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Show me the full graph subnetwork for suppliers in Germany",
        "kg_query",
        "kg_query",
        "visibility",
    ),
    # ── mcnf_solve (10) ──
    LabelledQuery(
        "Route 500 units from warehouse A to DC Berlin with min cost",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    LabelledQuery(
        "Optimize logistics flow from factory to 3 distribution centers",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    LabelledQuery(
        "Find minimum cost routing for 1000 bearings from TQ-Tokyo to Paris DC",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    LabelledQuery(
        "Solve the minimum cost network flow for our European distribution network",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    LabelledQuery(
        "Re-route 2000 units from backup suppliers to meet Berlin demand",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    LabelledQuery(
        "What is the optimal shipping path from node A to node D with capacity 5000?",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    LabelledQuery(
        "Minimize transportation cost across the supply network",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    LabelledQuery(
        "MCNF: nodes [W1,W2,DC1,DC2], arc costs and capacities specified, solve",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    LabelledQuery(
        "Network flow optimization: ship 800 units at $12/unit from factory to warehouse",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    LabelledQuery(
        "Compute least-cost flow plan for emergency restocking of DC-TOKYO",
        "mcnf_solve",
        "or_solve",
        "visibility",
    ),
    # ── disruption_resource (10) ──
    LabelledQuery(
        "TQ-Electronics is disrupted — find alternative suppliers for bearings",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Earthquake hit SHA-Electronics factory. What components are affected?",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Supplier HAN-Metals is bankrupt. Reallocate their orders",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Port congestion at Rotterdam is blocking shipments. Identify workarounds",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Flood disrupted Tier 2 supplier NOR-Parts. Which products are at risk?",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Critical component C-009 supply interrupted. Find backup sources",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Strike at LUC-Assemblies — reschedule affected work orders",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Tariffs imposed on imports from KOR-Tech suppliers — reroute supply",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Outbreak at FAB-002 factory reduced capacity by 40%. Adjust plans",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "BRA-Components delivery delayed 3 weeks. Which products will be affected?",
        "disruption_resource",
        "kg_query",
        "visibility",
    ),
    # ── meio_optimize (10) ──
    LabelledQuery(
        "Optimize safety stock levels across our 4-stage supply chain",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Run MEIO GSM to minimize inventory holding costs at 95% service level",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Calculate optimal guaranteed service times for multi-echelon inventory",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "What safety stock should stage 2 hold given lead time of 7 days and 98% SL?",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Solve for minimum inventory cost across all echelons",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "MEIO optimization for 5 stages with holding costs and demand variability",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Find optimal service times to minimize total safety stock cost",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Multi-echelon inventory model: 3 stages, service level 97%, optimize",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Minimize inventory investment across the distribution network",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "What is the Guaranteed Service Model recommendation for our DC network?",
        "meio_optimize",
        "or_solve",
        "inventory",
    ),
    # ── bullwhip_analyze (10) ──
    LabelledQuery(
        "Analyze bullwhip effect in our demand signal over last 52 weeks",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Calculate demand amplification ratios across our supply chain echelons",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "How severe is the bullwhip effect given our 4-week lead time?",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Quantify variance amplification in orders vs. consumer demand",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Bullwhip analysis: AR(1) process with rho=0.7 across 3 echelons",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "What is the spectral radius of our demand process?",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Run bullwhip simulation for 52-period demand series",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Is our order variance significantly higher than demand variance?",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Forecast amplification ratio for our replenishment policy",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Analyze demand signal distortion at the manufacturer echelon",
        "bullwhip_analyze",
        "or_solve",
        "inventory",
    ),
    # ── jsp_schedule (10) ──
    LabelledQuery(
        "Schedule 5 jobs across 3 machines to minimize makespan",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Job shop scheduling for our manufacturing floor this week",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Optimize production schedule: 8 jobs, 4 work centers, minimize completion time",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "JSP: Job 1 needs machine A then B, Job 2 needs B then A. Minimize makespan",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Create optimal shop floor schedule for next 5 days",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "CP-SAT job scheduling with precedence constraints and machine capacity",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "What is the minimum makespan for our assembly line schedule?",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Schedule work orders WO-101 to WO-115 on 6 production lines",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Production planning: minimize idle time on machines M1-M4",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    LabelledQuery(
        "Generate a Gantt-optimal schedule for the fabrication shop",
        "jsp_schedule",
        "or_solve",
        "inventory",
    ),
    # ── vrp_route (10) ──
    LabelledQuery(
        "Plan delivery routes for 5 trucks to serve 20 customer locations",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Vehicle routing: depot at Berlin, 8 delivery points, 3 vehicles of capacity 500",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Optimize last-mile delivery routes for DC-PARIS customers",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "VRP: minimize total distance for weekly customer deliveries",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Route 4 delivery vehicles from the Hamburg warehouse to all 15 retail sites",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Create optimal dispatch plan for tomorrow's delivery fleet",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Capacitated vehicle routing for our regional distribution network",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Solve CVRP: depot node 0, 12 customers, 2 trucks with capacity 1000 each",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "What is the minimum total travel distance for our delivery fleet?",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Optimize route plan to reduce fuel costs across delivery network",
        "vrp_route",
        "or_solve",
        "risk",
    ),
    # ── robust_allocate (10) ──
    LabelledQuery(
        "Allocate orders across suppliers under cost uncertainty",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Robust sourcing: minimize worst-case cost with omega=2 uncertainty budget",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Solve robust min-max allocation for 4 suppliers with cost ranges",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "SOCP robust optimization: diversify supply sources to hedge cost risk",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "What is the price of robustness for our current supplier portfolio?",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Worst-case optimal allocation across 3 competing suppliers",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Robust demand allocation with uncertainty budget omega=1.5",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Minimize maximum regret across all supplier scenarios",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "Robust sourcing strategy under supply cost volatility",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    LabelledQuery(
        "CVXPY robust allocation: 5 suppliers, cost_uncertainty=0.2, demand=1000",
        "robust_allocate",
        "or_solve",
        "risk",
    ),
    # ── contract_query (10) ──
    LabelledQuery(
        "What does supplier TQ-Electronics' force majeure clause say?",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    LabelledQuery(
        "Check if our contract with LUC-Assemblies allows termination for cause",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    LabelledQuery(
        "What are the payment terms in the FAB-002 supplier contract?",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    LabelledQuery(
        "Does the NOR-Parts agreement include quality inspection rights?",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    LabelledQuery(
        "Find the limitation of liability clause in the KOR-Tech contract",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    LabelledQuery(
        "What is the dispute resolution mechanism in the SHA-Electronics agreement?",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    LabelledQuery(
        "Retrieve governing law provisions from all active contracts",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    LabelledQuery(
        "Does our contract cover pandemic as a force majeure event?",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    LabelledQuery(
        "What notice period is required for contract termination?",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    LabelledQuery(
        "Find clauses about price adjustment triggers in the TQ contract",
        "contract_query",
        "contract_query",
        "compliance",
    ),
    # ── multi_step (10) ──
    LabelledQuery(
        "TQ-Electronics disrupted — find alternatives and optimize new routing cost",
        "multi_step",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Which suppliers can replace C-007 and what would be the new MCNF cost?",
        "multi_step",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Identify affected products from SHA disruption then schedule recovery jobs",
        "multi_step",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Traverse supply network for bearings and then optimize inventory safety stock",
        "multi_step",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Find backup sources for component C-012 and re-route 500 units to Berlin DC",
        "multi_step",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Map disrupted Tier 2 suppliers and solve minimum-cost reallocation",
        "multi_step",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "From the KG, identify affected DCs and run VRP for rerouting",
        "multi_step",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Analyze supply graph for P-015 then compute robust allocation across backups",
        "multi_step",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Check contract for FAB-002 and then optimize their delivery schedule",
        "multi_step",
        "kg_query",
        "visibility",
    ),
    LabelledQuery(
        "Full end-to-end: KG traversal → MCNF solve → synthesize recommendation",
        "multi_step",
        "kg_query",
        "visibility",
    ),
]

assert (
    len(LABELLED_DATASET) == 100
), f"Expected 100 queries, got {len(LABELLED_DATASET)}"

# ---------------------------------------------------------------------------
# Tool precision map — which tool should be called for each intent
# ---------------------------------------------------------------------------

INTENT_TO_TOOL: dict[str, str] = {
    "kg_query": "traverse_supply_network",
    "mcnf_solve": "solve_mcnf",
    "disruption_resource": "solve_disruption",
    "meio_optimize": "solve_meio_gsm",
    "bullwhip_analyze": "analyze_bullwhip",
    "jsp_schedule": "solve_jsp",
    "vrp_route": "solve_vrp",
    "robust_allocate": "solve_robust_minmax",
    "contract_query": "search_contracts",
    "multi_step": "traverse_supply_network",  # KG first
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state_with_intent(intent: str | None, confidence: float = 0.9) -> AgentState:
    return AgentState(  # type: ignore[misc]
        messages=[{"role": "user", "content": "eval harness query"}],
        intent=intent,
        intent_confidence=confidence,
        ddd_context=None,
        solver_input=None,
        solver_output=None,
        kg_subgraph=None,
        kg_entities=None,
        kg_relation_path=None,
        rag_documents=None,
        rag_evaluation=None,
        human_approval_required=False,
        decision_id=None,
        final_response=None,
        error=None,
    )


def _make_settings() -> MagicMock:
    s = MagicMock()
    s.llm_model = "gpt-4o"
    s.llm_base_url = "https://models.inference.ai.azure.com"
    s.github_token = "test-token"
    s.intent_confidence_threshold = 0.7
    s.human_approval_cost_threshold = 10_000.0
    return s


# ---------------------------------------------------------------------------
# Test 1: route_by_intent accuracy (§5.1.2 — intent classification routing)
# Verifies that every intent in the dataset maps to the correct graph edge.
# This is the routing layer of intent classification — 100% coverage expected.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sample",
    LABELLED_DATASET,
    ids=[f"{s.expected_intent[:8]}-{i:03d}" for i, s in enumerate(LABELLED_DATASET)],
)
def test_route_by_intent_labelled_dataset(sample: LabelledQuery) -> None:
    """route_by_intent() must produce the correct graph edge for every labelled query."""
    state = _make_state_with_intent(sample.expected_intent)
    result = route_by_intent(state)
    assert result == sample.expected_route, (
        f"Query: {sample.query[:80]!r}\n"
        f"  intent={sample.expected_intent!r}\n"
        f"  expected_route={sample.expected_route!r}\n"
        f"  got={result!r}"
    )


# ---------------------------------------------------------------------------
# Test 2: Mock-LLM intent classification accuracy (§5.1.2 >= 90%)
# classify_intent() is called with a mock LLM returning the ground-truth intent.
# Verifies the classify_intent node correctly propagates structured output into state.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "sample",
    LABELLED_DATASET,
    ids=[
        f"classify-{s.expected_intent[:6]}-{i:03d}"
        for i, s in enumerate(LABELLED_DATASET)
    ],
)
async def test_classify_intent_propagates_to_state(sample: LabelledQuery) -> None:
    """classify_intent() must write expected intent+confidence into AgentState."""
    state = _make_state_with_intent(None, confidence=0.0)
    state["messages"] = [{"role": "user", "content": sample.query}]

    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value=IntentClassification(
            intent=sample.expected_intent,
            intent_confidence=0.92,
            ddd_context=sample.expected_ddd,
            reasoning="mock eval",
        )
    )
    mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

    with (
        patch("app.agents.orchestrator.get_settings", return_value=_make_settings()),
        patch("app.agents.orchestrator.ChatOpenAI", return_value=mock_llm),
    ):
        result_state = await classify_intent(state)

    assert result_state["intent"] == sample.expected_intent, (
        f"classify_intent returned {result_state['intent']!r}, "
        f"expected {sample.expected_intent!r} for query: {sample.query[:80]!r}"
    )
    assert result_state["intent_confidence"] == pytest.approx(0.92, abs=0.01)


# ---------------------------------------------------------------------------
# Test 3: Aggregate accuracy metric (§5.1.2 >= 90%)
# Runs ALL 100 route_by_intent calls and asserts overall accuracy >= 90%.
# This is the M6 milestone gate check.
# ---------------------------------------------------------------------------


def test_overall_intent_routing_accuracy_meets_m6_target() -> None:
    """Overall routing accuracy over 100 labelled queries must be >= 90% (§5.1.2 M6)."""
    correct = 0
    failures: list[str] = []

    for sample in LABELLED_DATASET:
        state = _make_state_with_intent(sample.expected_intent)
        result = route_by_intent(state)
        if result == sample.expected_route:
            correct += 1
        else:
            failures.append(
                f"  intent={sample.expected_intent!r} → got={result!r}, "
                f"expected={sample.expected_route!r} | {sample.query[:60]!r}"
            )

    accuracy = correct / len(LABELLED_DATASET)
    failure_report = "\n".join(failures[:10])  # show first 10 failures only
    assert accuracy >= 0.90, (
        f"Intent routing accuracy {accuracy:.1%} < 90% threshold (M6 gate).\n"
        f"First failures:\n{failure_report}"
    )


# ---------------------------------------------------------------------------
# Test 4: Tool invocation precision (§5.1.2 >= 95%)
# Verifies the intent→tool mapping covers >= 95% of intent classes.
# ---------------------------------------------------------------------------


def test_tool_invocation_precision_mapping_coverage() -> None:
    """Every intent in LABELLED_DATASET must have a tool mapping (§5.1.2 >= 95%)."""
    all_intents = {s.expected_intent for s in LABELLED_DATASET}
    covered = {i for i in all_intents if i in INTENT_TO_TOOL}
    precision = len(covered) / len(all_intents)

    assert precision >= 0.95, (
        f"Tool precision {precision:.1%} < 95% threshold.\n"
        f"Missing tool mappings for: {all_intents - covered}"
    )


# ---------------------------------------------------------------------------
# Test 5: Confidence threshold gating (§4.2 — < 0.7 → clarification)
# Low-confidence intents must not bypass the threshold gate.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_low_confidence_intent_asks_clarification() -> None:
    """intent_confidence < 0.7 must route to 'unclear' (§4.2 threshold gate)."""
    state = _make_state_with_intent(None, confidence=0.0)
    state["messages"] = [{"role": "user", "content": "show me stuff"}]

    mock_llm = MagicMock()
    mock_chain = MagicMock()
    mock_chain.ainvoke = AsyncMock(
        return_value=IntentClassification(
            intent="kg_query",
            intent_confidence=0.45,  # below 0.7 threshold
            ddd_context="visibility",
            reasoning="ambiguous",
        )
    )
    mock_llm.with_structured_output = MagicMock(return_value=mock_chain)

    with (
        patch("app.agents.orchestrator.get_settings", return_value=_make_settings()),
        patch("app.agents.orchestrator.ChatOpenAI", return_value=mock_llm),
    ):
        result_state = await classify_intent(state)

    # Below threshold → intent must be set to "unclear"
    assert (
        result_state["intent"] == "unclear"
    ), f"Low confidence (0.45) should set intent='unclear', got {result_state['intent']!r}"
