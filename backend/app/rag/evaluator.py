"""CRAG relevance evaluator (Blueprint §4.4).

Determines whether top retrieved chunk is relevant to the query using an LLM
structured-output call at temperature=0.0 for determinism.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)

# Evaluation labels
CORRECT = "correct"
AMBIGUOUS = "ambiguous"
INCORRECT = "incorrect"

_VALID_LABELS = {CORRECT, AMBIGUOUS, INCORRECT}

_EVAL_SYSTEM_PROMPT = """\
You are a relevance evaluator for a supply-chain contract RAG system.

Given a user query and a retrieved document chunk, classify the relevance:
- "correct"   : the chunk directly and substantially answers the query.
- "ambiguous" : the chunk is partially relevant or addresses a related topic.
- "incorrect" : the chunk is not relevant to the query at all.

Respond ONLY with valid JSON matching the schema."""


class _RelevanceLabel(BaseModel):
    label: str = Field(..., description="One of: correct, ambiguous, incorrect")
    reasoning: str = Field(..., description="One-sentence reasoning")


async def evaluate_relevance(query: str, top_doc: dict | None) -> str:
    """Call LLM to classify retrieved document relevance.

    Returns "correct" | "ambiguous" | "incorrect".
    Falls back to "ambiguous" if the LLM call fails.

    If ``top_doc`` is None, returns "incorrect" immediately (no LLM call).
    """
    if top_doc is None:
        return INCORRECT

    chunk_text = top_doc.get("chunk_text", "")
    if not chunk_text.strip():
        return INCORRECT

    try:
        s = get_settings()
        llm = ChatOpenAI(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.github_token,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=256,
        )
        structured = llm.with_structured_output(_RelevanceLabel)
        result: _RelevanceLabel = await structured.ainvoke(
            [
                SystemMessage(_EVAL_SYSTEM_PROMPT),
                HumanMessage(f"Query: {query}\n\nDocument chunk:\n{chunk_text[:800]}"),
            ]
        )
        label = result.label.strip().lower()
        return label if label in _VALID_LABELS else AMBIGUOUS

    except Exception as exc:
        logger.warning("evaluate_relevance LLM call failed: %s", exc)
        return AMBIGUOUS
