"""Quota-exhaustion LLM fallback — Claude Haiku 4.5 via Anthropic.

Every LLM call in this codebase goes through GitHub Models (GPT-4o /
GPT-4o-mini). Free-tier quotas there are small (50-150 requests/day) and, once
exhausted, every call fails with an HTTP 429 (``openai.RateLimitError``) until
the daily window resets.

``with_quota_fallback`` wraps a primary (GitHub Models) runnable so that,
*specifically and only* on that exception, the same call is retried once
against Claude Haiku 4.5 instead. Any other exception (validation error,
timeout, genuine auth failure, ...) is NOT caught here — it propagates
unchanged to the caller's existing error handling (e.g. the keyword-classifier
fallback in ``orchestrator.classify_intent``), exactly as before this module
existed.

If ``ANTHROPIC_API_KEY`` is not configured, every function here is a no-op:
``make_fallback_llm`` returns None and ``with_quota_fallback`` returns the
primary runnable completely unwrapped, so behavior is identical to before
this module was added.
"""

from __future__ import annotations

import logging

import openai
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)

#: The only exception type that triggers a fallback attempt — a GitHub
#: Models 429 (daily quota exhausted or per-minute throttle hit). Every other
#: failure mode is left to the caller's own error handling.
QUOTA_EXCEPTIONS: tuple[type[BaseException], ...] = (openai.RateLimitError,)


def make_fallback_llm(max_tokens: int) -> ChatAnthropic | None:
    """Return a Claude Haiku 4.5 chat model, or None if unavailable.

    None whenever the fallback genuinely can't be used: no API key configured
    (the normal "feature disabled" case), OR settings fail to produce valid
    field values — which in practice means an unrelated test mocked
    ``get_settings()`` with a bare ``MagicMock()`` without this module in
    mind. Either way, "can't construct the fallback client" and "no fallback
    configured" should behave identically to every caller: return the
    primary runnable unwrapped.
    """
    try:
        s = get_settings()
        if not s.anthropic_api_key:
            return None
        return ChatAnthropic(
            model=s.fallback_llm_model,  # type: ignore[call-arg]
            anthropic_api_key=s.anthropic_api_key,  # type: ignore[arg-type]
            temperature=0.0,
            max_tokens=max_tokens,  # type: ignore[call-arg]
        )
    except Exception as exc:
        logger.debug("Fallback LLM unavailable (%s); primary result stands as-is", exc)
        return None


def with_quota_fallback(
    primary: Runnable,
    *,
    max_tokens: int,
    structured_schema: type[BaseModel] | None = None,
) -> Runnable:
    """Wrap ``primary`` to retry on Claude Haiku 4.5 after a 429 from GitHub Models.

    ``primary`` should already have ``.with_structured_output(schema)`` applied
    if this call site uses structured output — pass the SAME ``schema`` as
    ``structured_schema`` so the fallback model is configured identically.

    Returns ``primary`` unchanged if no ``ANTHROPIC_API_KEY`` is configured, or
    if ``primary`` isn't an actual ``Runnable`` (e.g. a bare ``MagicMock`` in a
    unit test that stubs the LLM directly) — ``.with_fallbacks()`` only means
    something on a real LangChain object.
    """
    if not isinstance(primary, Runnable):
        return primary

    fallback_llm = make_fallback_llm(max_tokens)
    if fallback_llm is None:
        return primary

    fallback: Runnable = (
        fallback_llm.with_structured_output(structured_schema)
        if structured_schema is not None
        else fallback_llm
    )
    return primary.with_fallbacks([fallback], exceptions_to_handle=QUOTA_EXCEPTIONS)
