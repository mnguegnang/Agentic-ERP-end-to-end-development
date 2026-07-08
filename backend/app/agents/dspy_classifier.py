"""DSPy-compiled intent classifier — optional upgrade over the zero-shot prompt.

The hand-written ``_INTENT_SYSTEM`` prompt in ``orchestrator.py`` is the
zero-shot baseline.  ``scripts/compile_intent_classifier.py`` uses DSPy
(MIPROv2 / BootstrapFewShot) to optimize instructions and few-shot
demonstrations against the 100-query labelled dataset, and saves the compiled
program next to this module.

At runtime this module is a strict opt-in:
  * ``dspy`` not installed        → ``classify()`` returns None
  * compiled artifact not present → ``classify()`` returns None
  * any DSPy/LLM failure          → ``classify()`` returns None

``orchestrator.llm_classify_intent`` tries this first and falls back to the
zero-shot structured-output call, so nothing changes until you compile.

All ``dspy`` imports are deferred — the package is an optional dependency
(``pip install -e ".[dspy]"``).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from app.api.schemas import VALID_INTENTS, IntentClassification
from app.config import get_settings

logger = logging.getLogger(__name__)

#: Compiled DSPy program saved by scripts/compile_intent_classifier.py
ARTIFACT_PATH = Path(__file__).parent / "compiled_intent_classifier.json"

_VALID_DDD = ("visibility", "inventory", "compliance", "sourcing", "logistics")

#: Confidence assigned when the compiled program emits an unparseable score.
_DEFAULT_CONFIDENCE = 0.8

# Cached (program, lm) pair — loaded once per process.  ``False`` means
# "tried and unavailable" so we don't re-attempt the import on every query.
_LOADED: object = None


def build_program():  # -> dspy.Module
    """Construct the (uncompiled) DSPy program.

    Shared by the compile script and the runtime loader so the saved artifact
    always matches the program structure it is loaded into.
    """
    import dspy

    class ClassifyIntent(dspy.Signature):
        """Classify a supply-chain ERP copilot query into exactly ONE
        bounded-context intent."""

        query: str = dspy.InputField(desc="Natural-language supply-chain query")
        intent: str = dspy.OutputField(desc="Exactly one of: " + ", ".join(sorted(VALID_INTENTS)))
        ddd_context: str = dspy.OutputField(
            desc="Exactly one of: visibility, inventory, compliance, sourcing, logistics"
        )
        confidence: float = dspy.OutputField(desc="Classification confidence between 0.0 and 1.0")

    return dspy.ChainOfThought(ClassifyIntent)


def make_lm():  # -> dspy.LM
    """DSPy language model bound to the configured provider (GitHub Models)."""
    import dspy

    s = get_settings()
    return dspy.LM(
        f"openai/{s.llm_model}",
        api_base=s.llm_base_url,
        api_key=s.github_token,
        temperature=0.0,
        max_tokens=1024,
    )


def _load_compiled():
    """Return (program, lm) or None; caches the outcome either way."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED if _LOADED is not False else None

    if not ARTIFACT_PATH.exists():
        _LOADED = False
        return None
    try:
        program = build_program()
        program.load(str(ARTIFACT_PATH))
        lm = make_lm()
        _LOADED = (program, lm)
        logger.info("DSPy compiled intent classifier loaded from %s", ARTIFACT_PATH)
        return _LOADED
    except Exception as exc:
        logger.warning("DSPy classifier unavailable (%s); using zero-shot baseline", exc)
        _LOADED = False
        return None


def is_available() -> bool:
    """True when dspy is installed AND the compiled artifact loads."""
    return _load_compiled() is not None


async def classify(query: str) -> IntentClassification | None:
    """Classify with the compiled DSPy program; None → caller should fall back."""
    loaded = _load_compiled()
    if loaded is None:
        return None
    program, lm = loaded

    try:
        import dspy

        def _run():
            with dspy.context(lm=lm):
                return program(query=query)

        # DSPy prediction is synchronous — keep it off the event loop.
        pred = await asyncio.to_thread(_run)

        intent = str(pred.intent).strip()
        if intent not in VALID_INTENTS:
            logger.warning("DSPy classifier returned unknown intent %r; falling back", intent)
            return None

        ddd = str(pred.ddd_context).strip().lower()
        if ddd not in _VALID_DDD:
            ddd = "visibility"

        try:
            confidence = max(0.0, min(1.0, float(pred.confidence)))
        except (TypeError, ValueError):
            confidence = _DEFAULT_CONFIDENCE

        return IntentClassification(
            intent=intent,
            intent_confidence=confidence,
            ddd_context=ddd,
            reasoning=str(getattr(pred, "reasoning", "dspy-compiled classifier"))[:500],
        )
    except Exception as exc:
        logger.warning("DSPy classify failed (%s); falling back to zero-shot baseline", exc)
        return None
