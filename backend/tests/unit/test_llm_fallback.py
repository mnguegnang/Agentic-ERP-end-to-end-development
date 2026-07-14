"""Unit tests — quota-exhaustion LLM fallback (Claude Haiku 4.5).

Contract pinned here:
  * no ANTHROPIC_API_KEY configured -> fallback is a total no-op
  * primary raises openai.RateLimitError -> retried on the fallback model
  * primary raises anything else -> propagates unchanged, fallback untouched
  * a non-Runnable primary (e.g. a bare MagicMock in another test's mocking
    style) is never wrapped -> avoids breaking unrelated tests' mocks
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import openai
import pytest
from app.agents.llm_fallback import make_fallback_llm, with_quota_fallback
from langchain_core.runnables import RunnableLambda


def _settings(anthropic_api_key: str = "") -> MagicMock:
    s = MagicMock()
    s.anthropic_api_key = anthropic_api_key
    s.fallback_llm_model = "claude-haiku-4-5-20251001"
    return s


def _rate_limit_error() -> openai.RateLimitError:
    response = httpx.Response(
        status_code=429,
        request=httpx.Request("POST", "https://models.inference.ai.azure.com/chat/completions"),
    )
    return openai.RateLimitError(
        "Error code: 429 - {'error': {'code': 'RateLimitReached', "
        "'message': 'Rate limit of 50 per 86400s exceeded for UserByModelByDay.'}}",
        response=response,
        body=None,
    )


# ---------------------------------------------------------------------------
# make_fallback_llm
# ---------------------------------------------------------------------------


def test_make_fallback_llm_none_without_api_key() -> None:
    with patch("app.agents.llm_fallback.get_settings", return_value=_settings("")):
        assert make_fallback_llm(max_tokens=100) is None


def test_make_fallback_llm_returns_chat_anthropic_when_configured() -> None:
    with patch("app.agents.llm_fallback.get_settings", return_value=_settings("sk-ant-fake")):
        llm = make_fallback_llm(max_tokens=100)
    assert llm is not None
    assert llm.model == "claude-haiku-4-5-20251001"


def test_make_fallback_llm_none_on_garbage_settings() -> None:
    """Settings that can't produce valid field values (e.g. an unrelated test's
    bare MagicMock) must degrade to 'no fallback', not raise."""
    garbage = MagicMock()  # garbage.anthropic_api_key is a truthy MagicMock
    with patch("app.agents.llm_fallback.get_settings", return_value=garbage):
        assert make_fallback_llm(max_tokens=100) is None


# ---------------------------------------------------------------------------
# with_quota_fallback — no API key configured => no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_api_key_is_a_total_noop() -> None:
    primary = RunnableLambda(lambda _: "primary result")
    with patch("app.agents.llm_fallback.get_settings", return_value=_settings("")):
        wrapped = with_quota_fallback(primary, max_tokens=50)
    assert wrapped is primary
    assert await wrapped.ainvoke("x") == "primary result"


# ---------------------------------------------------------------------------
# with_quota_fallback — non-Runnable primary (bare mock) is never wrapped
# ---------------------------------------------------------------------------


def test_non_runnable_primary_returned_unchanged() -> None:
    mock_primary = MagicMock()
    with patch("app.agents.llm_fallback.get_settings", return_value=_settings("sk-ant-fake")):
        result = with_quota_fallback(mock_primary, max_tokens=50)
    assert result is mock_primary


# ---------------------------------------------------------------------------
# with_quota_fallback — the actual quota-exhaustion behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rate_limit_error_triggers_fallback() -> None:
    def _primary(_input: str) -> str:
        raise _rate_limit_error()

    primary = RunnableLambda(_primary)
    fake_fallback = RunnableLambda(lambda _: "fallback answered")

    with (
        patch("app.agents.llm_fallback.get_settings", return_value=_settings("sk-ant-fake")),
        patch("app.agents.llm_fallback.ChatAnthropic", return_value=fake_fallback),
    ):
        wrapped = with_quota_fallback(primary, max_tokens=50)
        result = await wrapped.ainvoke("anything")

    assert result == "fallback answered"


@pytest.mark.asyncio
async def test_non_rate_limit_error_propagates_without_fallback() -> None:
    """Only openai.RateLimitError triggers the fallback — every other
    failure (validation, timeout, genuine auth error, ...) must propagate
    exactly as it did before this module existed."""

    def _primary(_input: str) -> str:
        raise ValueError("some unrelated failure")

    primary = RunnableLambda(_primary)
    fallback_should_never_run = RunnableLambda(
        lambda _: (_ for _ in ()).throw(AssertionError("fallback must not run"))
    )

    with (
        patch("app.agents.llm_fallback.get_settings", return_value=_settings("sk-ant-fake")),
        patch(
            "app.agents.llm_fallback.ChatAnthropic",
            return_value=fallback_should_never_run,
        ),
    ):
        wrapped = with_quota_fallback(primary, max_tokens=50)
        with pytest.raises(ValueError, match="some unrelated failure"):
            await wrapped.ainvoke("anything")


@pytest.mark.asyncio
async def test_fallback_not_invoked_when_primary_succeeds() -> None:
    primary = RunnableLambda(lambda _: "primary succeeded")
    fallback_should_never_run = RunnableLambda(
        lambda _: (_ for _ in ()).throw(AssertionError("fallback must not run"))
    )

    with (
        patch("app.agents.llm_fallback.get_settings", return_value=_settings("sk-ant-fake")),
        patch(
            "app.agents.llm_fallback.ChatAnthropic",
            return_value=fallback_should_never_run,
        ),
    ):
        wrapped = with_quota_fallback(primary, max_tokens=50)
        result = await wrapped.ainvoke("anything")

    assert result == "primary succeeded"
