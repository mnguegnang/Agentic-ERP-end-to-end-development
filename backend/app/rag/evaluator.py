"""CRAG relevance evaluator (Blueprint §4.4).

Determines whether top retrieved chunk is relevant to the query.
Stage 4 implementation.
"""
from __future__ import annotations

# Evaluation labels
CORRECT = "correct"
AMBIGUOUS = "ambiguous"
INCORRECT = "incorrect"


async def evaluate_relevance(query: str, top_doc: dict | None) -> str:
    """Call LLM to classify retrieved document relevance.

    Returns "correct" | "ambiguous" | "incorrect".
    If "incorrect", the CRAG pipeline returns no-answer (no hallucination).
    """
    if top_doc is None:
        return INCORRECT
    # TODO Stage 4: LLM call with structured output (temperature=0.0)
    return AMBIGUOUS
