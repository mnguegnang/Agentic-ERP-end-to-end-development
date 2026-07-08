"""Live intent-classification eval — measures the ACTUAL model, not the harness.

``test_agent_eval.py`` patches ``classify_intent`` and therefore only verifies
the routing table (a fine unit test, but not a model eval).  This harness runs
the same 100 labelled queries through the real structured-output LLM call and
reports per-intent accuracy against the §5.1.2 target (>= 90%).

Cost control:
  * marked ``slow`` + ``live_llm`` → excluded from the per-PR integration job
    (which runs ``-m "not slow"``); intended for nightly / pre-release runs.
  * skipped automatically when MOCK_LLM=true or GITHUB_TOKEN is absent.

Run:
    pytest backend/tests/integration/test_intent_eval_live.py -v -s
"""

from __future__ import annotations

import asyncio
import collections
import os
import re
from typing import NamedTuple

import pytest
from app.agents.orchestrator import llm_classify_intent
from app.config import get_settings

from tests.integration.test_agent_eval import LABELLED_DATASET

pytestmark = [pytest.mark.slow, pytest.mark.live_llm]

_ACCURACY_TARGET = 0.90
#: Max share of queries allowed to fail for infrastructure reasons (429s etc.)
#: before the eval result is considered meaningless.
_MAX_ERROR_RATE = 0.10
_CONCURRENCY = 4
_RETRIES = 3

#: Pulls a short, stable signature out of an exception message so 100 near-
#: identical 429s collapse into one line instead of flooding the summary —
#: e.g. "Error code: 429 - {...'code': 'RateLimitReached'...}" -> "429 RateLimitReached".
_ERROR_CODE_RE = re.compile(r"'code':\s*'([^']+)'")
_HTTP_STATUS_RE = re.compile(r"Error code:\s*(\d+)")


def _error_signature(exc: Exception) -> str:
    """Collapse an exception into a short, groupable signature for reporting."""
    text = str(exc)
    status = _HTTP_STATUS_RE.search(text)
    code = _ERROR_CODE_RE.search(text)
    if status and code:
        return f"{status.group(1)} {code.group(1)}"
    if status:
        return f"HTTP {status.group(1)}"
    return f"{type(exc).__name__}: {text[:80]}"


def _skip_unless_live() -> None:
    if os.getenv("MOCK_LLM", "").lower() == "true":
        pytest.skip("MOCK_LLM=true — live eval disabled")
    # Check the resolved setting, NOT os.getenv("GITHUB_TOKEN") directly —
    # pydantic-settings loads .env into the Settings object without ever
    # copying it back into the process environment, so a token that only
    # exists in .env (the normal case) would make a raw os.getenv() check
    # always skip, even though the app is fully configured to run.
    try:
        if not get_settings().github_token:
            pytest.skip(
                "GITHUB_TOKEN resolves empty (.env / shell) — cannot reach GitHub Models API"
            )
    except pytest.skip.Exception:
        raise
    except Exception as exc:  # e.g. pydantic ValidationError — no .env at all
        pytest.skip(f"Settings failed to load (no .env / GITHUB_TOKEN?): {exc}")


class ClassifyOutcome(NamedTuple):
    intent: str | None
    error_signature: str | None  # None on success


async def _classify_with_retry(query: str) -> ClassifyOutcome:
    """Return the predicted intent, or the LAST error signature after
    exhausting retries — never silently swallowed, so a failed eval run is
    self-diagnosing from its own printed summary (no manual repro needed).

    Retries with exponential backoff so transient 429s do not get counted
    as misclassifications.
    """
    last_error = "unknown error"
    for attempt in range(_RETRIES):
        try:
            result = await llm_classify_intent(query)
            return ClassifyOutcome(result.intent, None)
        except Exception as exc:
            last_error = _error_signature(exc)
            if attempt < _RETRIES - 1:
                await asyncio.sleep(2.0 * 2**attempt)
    return ClassifyOutcome(None, last_error)


async def test_intent_classification_accuracy_live() -> None:
    _skip_unless_live()

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def run_one(query: str) -> ClassifyOutcome:
        async with semaphore:
            return await _classify_with_retry(query)

    outcomes = await asyncio.gather(*(run_one(item.query) for item in LABELLED_DATASET))

    per_intent_total: collections.Counter[str] = collections.Counter()
    per_intent_correct: collections.Counter[str] = collections.Counter()
    error_signatures: collections.Counter[str] = collections.Counter()
    for item, outcome in zip(LABELLED_DATASET, outcomes):
        if outcome.error_signature is not None:
            error_signatures[outcome.error_signature] += 1
            continue
        per_intent_total[item.expected_intent] += 1
        if outcome.intent == item.expected_intent:
            per_intent_correct[item.expected_intent] += 1

    errors = sum(error_signatures.values())
    evaluated = sum(per_intent_total.values())
    correct = sum(per_intent_correct.values())
    accuracy = correct / evaluated if evaluated else 0.0

    print("\n─── Live intent eval ───")
    for intent in sorted(per_intent_total):
        total = per_intent_total[intent]
        print(f"  {intent:<20} {per_intent_correct[intent]}/{total}")
    print(f"  overall: {correct}/{evaluated} = {accuracy:.1%}  (errors: {errors})")
    if error_signatures:
        print("  error breakdown (self-diagnosing — no manual repro needed):")
        for sig, count in error_signatures.most_common():
            print(f"    {count:>3}x  {sig}")

    error_rate = errors / len(LABELLED_DATASET)
    assert error_rate <= _MAX_ERROR_RATE, (
        f"{errors} queries failed with API errors ({error_rate:.0%}) — "
        "eval result unreliable, rerun when the API is healthy. "
        f"Error breakdown: {dict(error_signatures.most_common())}"
    )
    assert accuracy >= _ACCURACY_TARGET, (
        f"Live intent accuracy {accuracy:.1%} below {_ACCURACY_TARGET:.0%} target; "
        f"per-intent breakdown printed above"
    )
