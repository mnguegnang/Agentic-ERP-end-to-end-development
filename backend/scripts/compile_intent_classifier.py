"""Compile the DSPy intent classifier against the 100-query labelled dataset.

The labelled queries in ``tests/integration/test_agent_eval.py`` become DSPy
training/dev examples; the optimizer selects instructions and few-shot
demonstrations that maximize intent accuracy, then saves the compiled program
to ``app/agents/compiled_intent_classifier.json``.  Once that artifact exists,
``orchestrator.llm_classify_intent`` uses it automatically.

Usage (from the repo root, venv active):

    pip install -e ".[dspy]"
    python backend/scripts/compile_intent_classifier.py                 # MIPROv2 (default)
    python backend/scripts/compile_intent_classifier.py --optimizer bootstrap
    python backend/scripts/compile_intent_classifier.py --dry-run       # baseline eval only

Requires GITHUB_TOKEN in .env.  A compile run makes real LLM calls (typically
a few hundred with MIPROv2 auto="light"); rerun the live eval afterwards:

    pytest backend/tests/integration/test_intent_eval_live.py -v -s
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_TRAIN_FRACTION = 0.8
_SEED = 42


def _is_auth_error(exc: Exception) -> bool:
    """Best-effort detection for provider authentication failures."""
    text = str(exc).lower()
    if _is_rate_limit_error(exc):
        return False
    markers = (
        "unauthorized",
        "bad credentials",
        "invalid api key",
        "error code: 401",
        "status code 401",
    )
    return any(m in text for m in markers)


def _is_rate_limit_error(exc: Exception) -> bool:
    """Best-effort detection for provider throttling/quota failures."""
    text = str(exc).lower()
    markers = (
        "ratelimit",
        "rate limit",
        "429",
        "quota",
        "too many requests",
    )
    return any(m in text for m in markers)


def _preflight_llm_auth() -> None:
    """Fail fast on invalid credentials before DSPy evaluation/optimization.

    A compile run can issue many LLM calls; this lightweight check avoids
    spending minutes only to fail mid-run with a deep DSPy/litellm traceback.
    """
    from app.config import get_settings
    from openai import OpenAI

    s = get_settings()
    client = OpenAI(base_url=s.llm_base_url, api_key=s.github_token)
    client.chat.completions.create(
        model=s.llm_model,
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
    )


def _ensure_repo_root_on_path() -> None:
    """Add repo root to sys.path so backend.* imports work from script execution."""
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


def _load_examples():
    """Turn the labelled eval dataset into dspy.Example objects."""
    _ensure_repo_root_on_path()

    import dspy

    try:
        from backend.tests.integration.test_agent_eval import LABELLED_DATASET
    except ModuleNotFoundError:
        # Fallback for legacy environments that expose tests as a top-level package.
        from tests.integration.test_agent_eval import LABELLED_DATASET

    examples = [
        dspy.Example(
            query=item.query,
            intent=item.expected_intent,
            ddd_context=item.expected_ddd,
            confidence=0.9,
        ).with_inputs("query")
        for item in LABELLED_DATASET
    ]
    random.Random(_SEED).shuffle(examples)
    split = int(len(examples) * _TRAIN_FRACTION)
    return examples[:split], examples[split:]


def _intent_match(example, prediction, trace=None) -> bool:
    """Optimization metric: exact intent match."""
    return str(prediction.intent).strip() == example.intent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--optimizer",
        choices=["mipro", "bootstrap"],
        default="mipro",
        help="mipro = MIPROv2 (instruction + demo search); bootstrap = BootstrapFewShot (cheaper)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate the uncompiled baseline on the dev split and exit (no compile)",
    )
    parser.add_argument(
        "--skip-auth-check",
        action="store_true",
        help="Skip the fast preflight auth request (not recommended)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    try:
        import dspy
        from dspy.evaluate import Evaluate
    except ImportError:
        print("dspy is not installed. Run:  pip install -e '.[dspy]'", file=sys.stderr)
        return 1

    from app.agents.dspy_classifier import ARTIFACT_PATH, build_program, make_lm

    if not args.skip_auth_check:
        try:
            _preflight_llm_auth()
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(
                    "Rate limit/quota reached during preflight check. "
                    "Retry after quota reset, or use a higher-quota token.",
                    file=sys.stderr,
                )
                return 3
            if _is_auth_error(exc):
                print(
                    "Authentication to GitHub Models failed (401/Bad credentials). "
                    "Check GITHUB_TOKEN in .env, then retry.",
                    file=sys.stderr,
                )
                return 2
            print(f"LLM preflight check failed: {exc}", file=sys.stderr)
            return 1

    dspy.configure(lm=make_lm())

    trainset, devset = _load_examples()
    logger.info("Dataset: %d train / %d dev", len(trainset), len(devset))

    program = build_program()
    evaluate = Evaluate(devset=devset, metric=_intent_match, display_progress=True, num_threads=4)

    try:
        baseline = evaluate(program)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            print(
                "Rate limit/quota reached during baseline evaluation. "
                "Retry later, or run with fewer calls (e.g., --dry-run after quota resets).",
                file=sys.stderr,
            )
            return 3
        if _is_auth_error(exc):
            print(
                "Authentication failed during baseline evaluation. "
                "Check GITHUB_TOKEN in .env, then retry.",
                file=sys.stderr,
            )
            return 2
        raise
    logger.info("Baseline (uncompiled) dev accuracy: %s", baseline)

    if args.dry_run:
        return 0

    if args.optimizer == "mipro":
        optimizer = dspy.MIPROv2(metric=_intent_match, auto="light")
        try:
            compiled = optimizer.compile(
                program, trainset=trainset, requires_permission_to_run=False
            )
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(
                    "Rate limit/quota reached during DSPy optimization. "
                    "Retry later, switch optimizer, or use a higher-quota token.",
                    file=sys.stderr,
                )
                return 3
            if _is_auth_error(exc):
                print(
                    "Authentication failed during DSPy optimization. "
                    "Check GITHUB_TOKEN in .env, then retry.",
                    file=sys.stderr,
                )
                return 2
            raise
    else:
        optimizer = dspy.BootstrapFewShot(
            metric=_intent_match,
            max_bootstrapped_demos=8,
            max_labeled_demos=8,
        )
        try:
            compiled = optimizer.compile(program, trainset=trainset)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                print(
                    "Rate limit/quota reached during DSPy optimization. "
                    "Retry later, switch optimizer, or use a higher-quota token.",
                    file=sys.stderr,
                )
                return 3
            if _is_auth_error(exc):
                print(
                    "Authentication failed during DSPy optimization. "
                    "Check GITHUB_TOKEN in .env, then retry.",
                    file=sys.stderr,
                )
                return 2
            raise

    try:
        score = evaluate(compiled)
    except Exception as exc:
        if _is_rate_limit_error(exc):
            print(
                "Rate limit/quota reached during compiled-model evaluation. "
                "Retry later, or evaluate with a higher-quota token.",
                file=sys.stderr,
            )
            return 3
        if _is_auth_error(exc):
            print(
                "Authentication failed during compiled-model evaluation. "
                "Check GITHUB_TOKEN in .env, then retry.",
                file=sys.stderr,
            )
            return 2
        raise
    logger.info("Compiled dev accuracy: %s (baseline was %s)", score, baseline)

    compiled.save(str(ARTIFACT_PATH))
    logger.info("Saved compiled program → %s", ARTIFACT_PATH)
    logger.info(
        "The orchestrator will now use the compiled classifier automatically. "
        "Verify with:  pytest backend/tests/integration/test_intent_eval_live.py -v -s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
