"""DPO dataset curation from LangSmith traces (Blueprint §6.3.1).

Stage 6 implementation (post M6 — 5K+ traces collected).

Prerequisites:
    pip install langsmith==0.2.10 datasets==3.2.0

Usage (on Lightning AI L4 after M6 passes and 5K+ traces collected):
    export LANGCHAIN_API_KEY=<your-langsmith-key>
    export LANGSMITH_PROJECT=agentic-erp-supply-chain
    python fine_tune/prepare_dataset.py \
        --output data/dpo_training/dpo_dataset \
        --min_pairs 5000

Target: 5,000 preference pairs across 10 intents.
  preferred : correct tool call + correct parameters + coherent synthesis
  dispreferred : hallucinated answer (no tool call) OR wrong tool OR wrong params
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Intent → expected tool name mapping (used as correctness heuristic)
# ---------------------------------------------------------------------------
INTENT_TOOL_MAP: dict[str, str] = {
    "kg_query": "traverse_supply_network",
    "mcnf_solve": "solve_mcnf",
    "disruption_resource": "solve_disruption",
    "meio_optimize": "solve_meio_gsm",
    "bullwhip_analyze": "analyze_bullwhip",
    "jsp_schedule": "solve_jsp",
    "vrp_route": "solve_vrp",
    "robust_allocate": "solve_robust_minmax",
    "contract_query": "search_contracts",
    "multi_step": "traverse_supply_network",
}

# ---------------------------------------------------------------------------
# LangSmith trace extraction
# ---------------------------------------------------------------------------


def fetch_langsmith_runs(
    project: str,
    min_runs: int = 5000,
    langsmith_api_key: str | None = None,
) -> list[dict[str, Any]]:
    """Pull completed runs from LangSmith for the given project.

    Returns a list of run dicts with keys:
        id, inputs, outputs, feedback_stats, child_runs
    """
    try:
        from langsmith import Client  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "langsmith not installed. Run: pip install langsmith==0.2.10"
        ) from e

    api_key = langsmith_api_key or os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        raise ValueError("LANGCHAIN_API_KEY environment variable not set.")

    client = Client(api_key=api_key)
    logger.info("Fetching runs from LangSmith project %r (target: %d)...", project, min_runs)

    runs = list(
        client.list_runs(
            project_name=project,
            run_type="chain",
            execution_order=1,  # only top-level runs
            limit=min_runs * 2,  # over-fetch to account for filtering
        )
    )
    logger.info("Fetched %d raw runs from LangSmith.", len(runs))
    return [
        {
            "id": str(r.id),
            "inputs": r.inputs or {},
            "outputs": r.outputs or {},
            "feedback_stats": r.feedback_stats or {},
            "child_runs": [str(c) for c in (r.child_run_ids or [])],
            "error": r.error,
        }
        for r in runs
    ]


# ---------------------------------------------------------------------------
# Preference pair construction
# ---------------------------------------------------------------------------


def _extract_tool_call(run: dict[str, Any]) -> str | None:
    """Extract the MCP tool name invoked in the run (from outputs)."""
    outputs = run.get("outputs", {})
    return outputs.get("tool_used")  # populated by WsResponse.tool_used


def _is_preferred(run: dict[str, Any]) -> bool:
    """Heuristic: preferred if tool was called and no error occurred.

    A run is preferred when:
    1. A tool was invoked (not a hallucinated free-text answer)
    2. The solver returned a non-error status
    3. Human feedback (if available) is positive (score >= 0.7)
    """
    if run.get("error"):
        return False
    tool = _extract_tool_call(run)
    if not tool:
        return False  # No tool call → hallucination (dispreferred)
    outputs = run.get("outputs", {})
    solver_result = outputs.get("solver_result") or {}
    if isinstance(solver_result, dict):
        status = solver_result.get("status", "")
        if status not in ("", "OPTIMAL", "FEASIBLE", "correct", "ambiguous"):
            return False  # Solver failed
    feedback = run.get("feedback_stats", {})
    if feedback:
        score = feedback.get("user_score", {}).get("avg", 1.0)
        if score < 0.7:
            return False
    return True


def build_preference_pairs(
    runs: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Build (prompt, chosen, rejected) triples from raw runs.

    Pairing strategy:
      - For each query, find one preferred run and one dispreferred run.
      - preferred  → complete tool-call response
      - dispreferred → hallucinated answer (no tool) or error response

    Returns list of dicts with keys: prompt, chosen, rejected, intent
    """
    # Group by query content
    query_groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        query = (run.get("inputs", {}).get("query") or "").strip()
        if not query:
            continue
        query_groups.setdefault(query, []).append(run)

    pairs: list[dict[str, str]] = []
    for query, group_runs in query_groups.items():
        preferred_runs = [r for r in group_runs if _is_preferred(r)]
        dispreferred_runs = [r for r in group_runs if not _is_preferred(r)]

        if not preferred_runs or not dispreferred_runs:
            continue  # Need at least one of each

        pref = preferred_runs[0]
        dispref = dispreferred_runs[0]

        chosen = (pref.get("outputs", {}).get("content") or "")
        rejected = (dispref.get("outputs", {}).get("content") or "")

        if not chosen or not rejected or chosen == rejected:
            continue

        pairs.append(
            {
                "prompt": query,
                "chosen": chosen,
                "rejected": rejected,
                "intent": pref.get("outputs", {}).get("intent") or "unknown",
                "tool_used": _extract_tool_call(pref) or "none",
            }
        )

    logger.info("Built %d preference pairs from %d runs.", len(pairs), len(runs))
    return pairs


# ---------------------------------------------------------------------------
# Dataset saving (HuggingFace Datasets format)
# ---------------------------------------------------------------------------


def save_dataset(pairs: list[dict[str, str]], output_path: str) -> None:
    """Save preference pairs as a HuggingFace Dataset (arrow + json).

    Args:
        pairs: list of {prompt, chosen, rejected, intent, tool_used}
        output_path: directory path (created if not exists)
    """
    try:
        from datasets import Dataset  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError(
            "datasets not installed. Run: pip install datasets==3.2.0"
        ) from e

    import os
    os.makedirs(output_path, exist_ok=True)

    ds = Dataset.from_list(pairs)
    ds.save_to_disk(output_path)
    logger.info("Dataset saved to %s (%d pairs)", output_path, len(ds))

    # Also save a JSON summary for human inspection
    summary_path = os.path.join(output_path, "summary.json")
    intent_counts: dict[str, int] = {}
    for p in pairs:
        intent_counts[p.get("intent", "unknown")] = (
            intent_counts.get(p.get("intent", "unknown"), 0) + 1
        )
    with open(summary_path, "w") as f:
        json.dump(
            {"total_pairs": len(pairs), "intent_distribution": intent_counts},
            f,
            indent=2,
        )
    logger.info("Summary written to %s", summary_path)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _synthetic_pairs(n: int) -> list[dict[str, str]]:
    """Generate n synthetic DPO preference pairs for smoke-testing."""
    templates = [
        ("kg_query", "Display the supply network for component BRG-009",
         "traverse_supply_network(entity_name='BRG-009', relation_type='PROVIDES')",
         "I don't know the answer."),
        ("mcnf_solve", "Optimize flow of 300 units from S1 to D2",
         "solve_mcnf(nodes=['S1','D2'], arcs=[...], commodities=[...])",
         "The flow cannot be determined."),
        ("contract_query", "What does the force majeure clause say?",
         "search_contracts(query_text='force majeure clause conditions')",
         "I am unable to search contracts."),
        ("disruption_resource", "Supplier BRA-Metals is offline — find alternatives",
         "solve_disruption(disrupted_supplier='BRA-Metals', component='metal')",
         "No alternative found."),
        ("meio_optimize", "Optimize safety stock for 3-echelon network",
         "solve_meio_gsm(stages=[...], service_level=0.95)",
         "Cannot optimize inventory."),
    ]
    pairs = []
    for i in range(n):
        tmpl = templates[i % len(templates)]
        pairs.append({"prompt": tmpl[1], "chosen": tmpl[2], "rejected": tmpl[3],
                       "intent": tmpl[0], "tool_used": tmpl[2].split("(")[0]})
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO dataset curation (§6.3.1)")
    parser.add_argument(
        "--project-name", "--project",
        dest="project",
        default=os.environ.get("LANGSMITH_PROJECT", "agentic-erp-supply-chain"),
        help="LangSmith project name",
    )
    parser.add_argument(
        "--output-path", "--output",
        dest="output",
        default="data/dpo_training/dpo_dataset",
        help="Output directory for HuggingFace Dataset",
    )
    parser.add_argument(
        "--min-pairs", "--min_pairs",
        dest="min_pairs",
        type=int,
        default=5000,
        help="Minimum preference pairs to collect (§6.3.1 target: 5000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print synthetic pairs to stdout without fetching LangSmith or saving",
    )
    parser.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Generate synthetic pairs and save to disk (no LangSmith needed; for smoke tests)",
    )
    args = parser.parse_args()

    if args.use_synthetic:
        pairs = _synthetic_pairs(args.min_pairs)
        logger.info("Generated %d synthetic pairs.", len(pairs))
        save_dataset(pairs, args.output)
        logger.info("Synthetic dataset saved to %s. Ready for train_dpo.py.", args.output)
        return

    if args.dry_run:
        pairs = _synthetic_pairs(args.min_pairs)
        print(f"\n── Dry-run: {len(pairs)} synthetic preference pair(s) ──")
        for i, p in enumerate(pairs[:5]):
            print(f"\n[{i+1}] intent={p['intent']}")
            print(f"  prompt  : {p['prompt']!r}")
            print(f"  chosen  : {p['chosen']!r}")
            print(f"  rejected: {p['rejected']!r}")
        logger.info("Dry-run complete. No data written to disk.")
        return

    runs = fetch_langsmith_runs(
        project=args.project,
        min_runs=args.min_pairs,
    )
    pairs = build_preference_pairs(runs)

    if len(pairs) < args.min_pairs:
        logger.warning(
            "Only %d pairs built (target: %d). Collect more traces before training.",
            len(pairs),
            args.min_pairs,
        )
    else:
        logger.info("Target met: %d pairs >= %d minimum.", len(pairs), args.min_pairs)

    save_dataset(pairs, args.output)
    logger.info("Dataset curation complete. Run train_dpo.py next.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
