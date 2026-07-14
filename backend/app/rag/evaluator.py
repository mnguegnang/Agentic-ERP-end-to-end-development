"""CRAG relevance evaluator (Blueprint §4.4).

Determines whether top retrieved chunk is relevant to the query using an LLM
structured-output call at temperature=0.0 for determinism.
"""

from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.llm_fallback import with_quota_fallback
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


class _RelevanceLabels(BaseModel):
    labels: list[str] = Field(
        ...,
        description=(
            "One label per numbered document, in order. "
            "Each label is one of: correct, ambiguous, incorrect"
        ),
    )


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
            max_tokens=256,  # type: ignore[call-arg]
        )
        structured = with_quota_fallback(
            llm.with_structured_output(_RelevanceLabel),
            max_tokens=256,
            structured_schema=_RelevanceLabel,
        )
        result: _RelevanceLabel = await structured.ainvoke(  # type: ignore[assignment]
            [
                SystemMessage(_EVAL_SYSTEM_PROMPT),
                HumanMessage(f"Query: {query}\n\nDocument chunk:\n{chunk_text}"),
            ]
        )
        label = result.label.strip().lower()
        return label if label in _VALID_LABELS else AMBIGUOUS

    except Exception as exc:
        logger.warning("evaluate_relevance LLM call failed: %s", exc)
        return AMBIGUOUS


_BATCH_EVAL_SYSTEM_PROMPT = """\
You are a relevance evaluator for a supply-chain contract RAG system.

Given a user query and several numbered document chunks, classify EACH chunk:
- "correct"   : the chunk directly and substantially answers the query.
- "ambiguous" : the chunk is partially relevant or addresses a related topic.
- "incorrect" : the chunk is not relevant to the query at all.

Return exactly one label per chunk, in the same order as the chunks.
Respond ONLY with valid JSON matching the schema."""


async def evaluate_relevance_batch(query: str, docs: list[dict]) -> list[str]:
    """Label the relevance of every retrieved chunk in ONE LLM call (CRAG).

    Per-document labels let the caller drop only the irrelevant chunks instead
    of gating the whole result set on the top-1 verdict.  Chunks with empty
    text are labelled "incorrect" without consulting the LLM.  On LLM failure
    every remaining chunk falls back to "ambiguous" (same policy as the
    single-document evaluator).
    """
    if not docs:
        return []

    labels: list[str] = [INCORRECT] * len(docs)
    to_eval = [(i, doc) for i, doc in enumerate(docs) if (doc.get("chunk_text") or "").strip()]
    if not to_eval:
        return labels

    numbered = "\n\n".join(
        f"[{n + 1}] {doc.get('chunk_text') or ''}" for n, (_, doc) in enumerate(to_eval)
    )
    try:
        s = get_settings()
        llm = ChatOpenAI(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.github_token,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=512,  # type: ignore[call-arg]
        )
        structured = with_quota_fallback(
            llm.with_structured_output(_RelevanceLabels),
            max_tokens=512,
            structured_schema=_RelevanceLabels,
        )
        result: _RelevanceLabels = await structured.ainvoke(  # type: ignore[assignment]
            [
                SystemMessage(_BATCH_EVAL_SYSTEM_PROMPT),
                HumanMessage(f"Query: {query}\n\nDocument chunks:\n{numbered}"),
            ]
        )
        returned = [lbl.strip().lower() for lbl in result.labels]
        for n, (i, _) in enumerate(to_eval):
            lbl = returned[n] if n < len(returned) else AMBIGUOUS
            labels[i] = lbl if lbl in _VALID_LABELS else AMBIGUOUS
    except Exception as exc:
        logger.warning("evaluate_relevance_batch LLM call failed: %s", exc)
        for i, _ in to_eval:
            labels[i] = AMBIGUOUS
    return labels


_REWRITE_SYSTEM_PROMPT = """\
You rewrite search queries for a supply-chain contract retrieval system.
The original query failed to retrieve relevant contract chunks.
Rewrite it as a self-contained keyword-rich search query: expand abbreviations,
add likely contract terminology (clause names, section topics), and drop
conversational filler. Return ONLY the rewritten query text."""


async def rewrite_query(query: str) -> str | None:
    """One-shot corrective query rewrite (CRAG corrective action).

    Returns the rewritten query, or None if the LLM is unavailable or returns
    nothing usable — callers treat None as "skip the retry".
    """
    try:
        s = get_settings()
        llm = ChatOpenAI(
            model=s.llm_model,
            base_url=s.llm_base_url,
            api_key=s.github_token,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=128,  # type: ignore[call-arg]
        )
        llm_with_fallback = with_quota_fallback(llm, max_tokens=128)
        response = await llm_with_fallback.ainvoke(
            [
                SystemMessage(_REWRITE_SYSTEM_PROMPT),
                HumanMessage(query),
            ]
        )
        rewritten = str(response.content).strip()
        if rewritten and rewritten.lower() != query.strip().lower():
            return rewritten
        return None
    except Exception as exc:
        logger.warning("rewrite_query LLM call failed: %s", exc)
        return None
