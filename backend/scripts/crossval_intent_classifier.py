"""Cross-validated DSPy metric — honest baseline→compiled delta on the existing
100 labelled queries, with NO new labelling and NO train/test contamination.

Why this exists
---------------
A single 80/20 split gives a 1-example-noisy number (1 dev example = 5 pp) and,
when the same 100 queries are reused for the live eval, a *contaminated* number
(80 of them were in the optimizer's trainset). k-fold cross-validation fixes
both: each of the 100 queries is scored exactly once by a fold model that never
trained on it, and the per-fold deltas give a mean ± std instead of a point
estimate. Authoritative basis: scikit-learn User Guide §3.1 / §3.1.2.1.1
("does not waste too much data … a major advantage … when the number of samples
is very small"); Hastie et al., ESL §7.10; James et al., ISL §5.1.3.

For each of K stratified folds (stratified by intent, so every intent appears in
every test fold):
  * compile MIPROv2 on the fold's train split (never sees the test fold),
  * evaluate the uncompiled baseline AND the compiled program on the test fold.

Reports per-fold baseline/compiled/delta, the pooled held-out accuracy over all
100 queries, and the mean ± std of the per-fold deltas.

This is a MEASUREMENT harness — it runs K full compiles and saves NO artifact.
Use compile_intent_classifier.py to produce the deployable artifact.

Single model via --llm (same rationale as compile_intent_classifier.py: a DSPy
prompt is tuned for one LM). Defaults match the run that produced the +5 pp dev
result: --llm haiku is required explicitly; --auto medium.

Usage (repo root, venv active):
    python backend/scripts/crossval_intent_classifier.py --llm haiku
    python backend/scripts/crossval_intent_classifier.py --llm haiku --auto light -k 5

Cost: K full MIPROv2 compiles (K≈4 min each on Haiku at num_threads=1) plus
2*K test-fold evals. Real LLM calls; --llm haiku needs ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import logging
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

_SEED = 42


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def _load_examples():
    """Load the 100 labelled queries as dspy.Example objects."""
    _ensure_repo_root_on_path()
    import dspy

    try:
        from backend.tests.integration.test_agent_eval import LABELLED_DATASET
    except ModuleNotFoundError:
        from tests.integration.test_agent_eval import LABELLED_DATASET

    return [
        dspy.Example(
            query=item.query,
            intent=item.expected_intent,
            ddd_context=item.expected_ddd,
            confidence=0.9,
        ).with_inputs("query")
        for item in LABELLED_DATASET
    ]


def _intent_match(example, prediction, trace=None) -> bool:
    return str(prediction.intent).strip() == example.intent


class _HardStopError(Exception):
    """Raised on a provider hard-failure (billing/credit or rate limit) so the
    run stops cleanly instead of recording corrupted folds."""


def _is_billing_error(exc: Exception) -> bool:
    t = str(exc).lower()
    return "credit balance" in t or "plans & billing" in t or ("too low" in t and "balance" in t)


def _is_rate_limit_error(exc: Exception) -> bool:
    t = str(exc).lower()
    return any(m in t for m in ("ratelimit", "rate limit", "429", "too many requests", "quota"))


def _hard_stop_reason(exc: Exception) -> str | None:
    """Return a user-facing reason if exc is a provider hard-failure, else None."""
    if _is_billing_error(exc):
        return (
            "Anthropic credit balance exhausted (HTTP 400 billing error). "
            "Top up credits at console.anthropic.com → Plans & Billing, then re-run."
        )
    if _is_rate_limit_error(exc):
        return (
            "Provider rate limit / quota exhausted. Retry later, lower the budget "
            "(--auto light -k 3), or use a higher-tier key."
        )
    return None


def _is_noop_compile(compiled, build_program) -> bool:
    """True if the compiled program equals the seed (empty demos, unchanged
    instruction) — the signature of a compile whose optimizer trials were
    corrupted by throttling/credit errors (or that found no improvement)."""
    try:
        fresh = build_program()
        demos = getattr(compiled.predict, "demos", [])
        compiled_instr = str(compiled.predict.signature.instructions).strip()
        base_instr = str(fresh.predict.signature.instructions).strip()
        return not demos and compiled_instr == base_instr
    except Exception:
        return False


def _stratified_folds(examples, k: int, seed: int) -> list[list[int]]:
    """Partition example indices into k folds, stratified by intent."""
    by_intent: dict[str, list[int]] = defaultdict(list)
    for i, ex in enumerate(examples):
        by_intent[ex.intent].append(i)
    rng = random.Random(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for idxs in by_intent.values():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        for j, idx in enumerate(shuffled):
            folds[j % k].append(idx)
    return folds


def _resolve_lm(tag: str):
    """Return (lm, label) for the whole run. haiku uses ANTHROPIC_API_KEY."""
    import dspy
    from app.agents.dspy_classifier import make_lm
    from app.config import get_settings

    s = get_settings()
    if tag == "haiku":
        if not s.anthropic_api_key:
            print("--llm haiku needs ANTHROPIC_API_KEY in .env. Aborting.", file=sys.stderr)
            sys.exit(3)
        lm = dspy.LM(
            f"anthropic/{s.fallback_llm_model}",
            api_key=s.anthropic_api_key,
            temperature=0.0,
            max_tokens=1024,
        )
        return lm, f"Claude Haiku 4.5 ({s.fallback_llm_model})"
    return make_lm(), f"GitHub Models ({s.llm_model})"


def _eval_correct(program, testset) -> int:
    """Count exact-intent matches of program over testset (held-out fold).

    A provider hard-failure (billing/credit or rate limit) raises _HardStopError
    so the caller aborts cleanly rather than scoring a fold at 0% from failed
    calls. Genuine per-example errors are logged and counted as misses.
    """
    correct = 0
    for ex in testset:
        try:
            pred = program(query=ex.query)
            if str(pred.intent).strip() == ex.intent:
                correct += 1
        except Exception as exc:
            reason = _hard_stop_reason(exc)
            if reason:
                raise _HardStopError(reason) from exc
            logger.warning("eval call failed on %r: %s", ex.query[:60], exc)
    return correct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", choices=["gpt", "haiku"], default="haiku")
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="medium")
    parser.add_argument("-k", "--folds", type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    # MIPROv2 is very chatty; keep the fold summaries readable.
    logging.getLogger("dspy").setLevel(logging.WARNING)

    try:
        import dspy
    except ImportError:
        print("dspy is not installed. Run:  pip install -e '.[dspy]'", file=sys.stderr)
        return 1

    from app.agents.dspy_classifier import build_program

    lm, provider = _resolve_lm(args.llm)
    dspy.configure(lm=lm)

    examples = _load_examples()
    n = len(examples)
    folds = _stratified_folds(examples, args.folds, _SEED)

    logger.info(
        "Cross-validation: %d examples, k=%d, model=%s, optimizer=mipro, auto=%s",
        n,
        args.folds,
        provider,
        args.auto,
    )

    per_fold = []  # (baseline_acc, compiled_acc, delta) as fractions — valid folds only
    pooled_baseline_correct = 0
    pooled_compiled_correct = 0
    pooled_total = 0
    noop_folds: list[int] = []
    aborted_reason: str | None = None

    for f, test_idx in enumerate(folds, start=1):
        test_set = [examples[i] for i in test_idx]
        train_set = [examples[i] for i in range(n) if i not in set(test_idx)]

        try:
            base_c = _eval_correct(build_program(), test_set)
            optimizer = dspy.MIPROv2(metric=_intent_match, auto=args.auto)
            compiled = optimizer.compile(
                build_program(), trainset=train_set, requires_permission_to_run=False
            )
            comp_c = _eval_correct(compiled, test_set)
        except _HardStopError as stop:
            aborted_reason = str(stop)
            logger.error("Fold %d/%d ABORTED: %s", f, args.folds, aborted_reason)
            break
        except Exception as exc:
            reason = _hard_stop_reason(exc)
            if reason is None:
                raise
            aborted_reason = reason
            logger.error("Fold %d/%d ABORTED: %s", f, args.folds, aborted_reason)
            break

        # Guard: a no-op compile means the optimizer trials were corrupted
        # (throttling/credit) or found no improvement — its delta is not a
        # trustworthy measurement, so exclude it from the aggregate.
        if _is_noop_compile(compiled, build_program):
            noop_folds.append(f)
            logger.warning(
                "Fold %d/%d: compile is a NO-OP (empty demos, unchanged instruction) — "
                "excluded from the aggregate (corrupted trials or no improvement).",
                f,
                args.folds,
            )
            continue

        t = len(test_set)
        base_acc, comp_acc = base_c / t, comp_c / t
        per_fold.append((base_acc, comp_acc, comp_acc - base_acc))
        pooled_baseline_correct += base_c
        pooled_compiled_correct += comp_c
        pooled_total += t

        logger.info(
            "Fold %d/%d (n_test=%d): baseline %.1f%% (%d/%d) | compiled %.1f%% (%d/%d) | "
            "delta %+.1f pp",
            f,
            args.folds,
            t,
            base_acc * 100,
            base_c,
            t,
            comp_acc * 100,
            comp_c,
            t,
            (comp_acc - base_acc) * 100,
        )

    if aborted_reason:
        print("\n" + "=" * 72, file=sys.stderr)
        print(f"⚠  RUN ABORTED — {aborted_reason}", file=sys.stderr)
        print("=" * 72, file=sys.stderr)

    if not per_fold:
        print(
            "\nNo valid folds completed — cannot report a cross-validated metric. "
            f"(no-op folds: {noop_folds or 'none'}).",
            file=sys.stderr,
        )
        return 3

    deltas = [d * 100 for _, _, d in per_fold]
    base_mean = statistics.mean(b * 100 for b, _, _ in per_fold)
    comp_mean = statistics.mean(c * 100 for _, c, _ in per_fold)
    delta_mean = statistics.mean(deltas)
    delta_sd = statistics.stdev(deltas) if len(deltas) > 1 else 0.0

    valid = len(per_fold)
    partial = valid < args.folds
    print("\n" + "=" * 72)
    header = "Cross-validated DSPy metric" + (" (PARTIAL — not all folds valid)" if partial else "")
    print(f"  {header} — {provider}, auto={args.auto}")
    print("=" * 72)
    print(
        f"  valid folds          : {valid}/{args.folds}"
        + (f"  (no-op: {noop_folds})" if noop_folds else "")
    )
    print(f"  per-fold deltas (pp) : {[round(d, 1) for d in deltas]}")
    print(f"  baseline (mean)      : {base_mean:.1f}%")
    print(f"  compiled (mean)      : {comp_mean:.1f}%")
    print(f"  delta (mean ± std)   : {delta_mean:+.1f} ± {delta_sd:.1f} pp")
    print(
        f"  pooled held-out      : baseline {pooled_baseline_correct}/{pooled_total} "
        f"({100 * pooled_baseline_correct / pooled_total:.1f}%)  →  "
        f"compiled {pooled_compiled_correct}/{pooled_total} "
        f"({100 * pooled_compiled_correct / pooled_total:.1f}%)"
    )
    print(
        f"  pooled delta         : {100 * (pooled_compiled_correct - pooled_baseline_correct) / pooled_total:+.1f} pp "
        f"(each of {pooled_total} valid-fold queries scored by a model that never trained on it)"
    )
    print("=" * 72)
    if partial:
        print(
            "  NOTE: partial run — treat the metric as indicative, not final. "
            "Re-run clean (top up credit / lower budget) for the full k-fold number.",
        )
    return 2 if partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
