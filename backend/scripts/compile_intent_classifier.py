"""Compile the DSPy intent classifier against the 100-query labelled dataset.

The labelled queries in ``tests/integration/test_agent_eval.py`` become DSPy
training/dev examples; the optimizer selects instructions and few-shot
demonstrations that maximize intent accuracy, then saves the compiled program
to ``app/agents/compiled_intent_classifier.json``.  Once that artifact exists,
``orchestrator.llm_classify_intent`` uses it automatically.

The whole run — baseline eval, optimization, and final eval — uses ONE model,
chosen with ``--llm`` (a DSPy-compiled prompt is tuned for a specific LM, so a
mixed-model before/after would be meaningless). Serve the artifact on the same
model you compile with.

Usage (from the repo root, venv active):

    pip install -e ".[dspy]"
    python backend/scripts/compile_intent_classifier.py                 # --llm gpt (default), MIPROv2
    python backend/scripts/compile_intent_classifier.py --llm haiku     # Claude Haiku 4.5 (needs ANTHROPIC_API_KEY)
    python backend/scripts/compile_intent_classifier.py --optimizer bootstrap
    python backend/scripts/compile_intent_classifier.py --dry-run       # baseline eval only

``--llm gpt`` needs GITHUB_TOKEN; ``--llm haiku`` needs ANTHROPIC_API_KEY. A
compile makes many real LLM calls (a few hundred with MIPROv2 auto="light"),
which the GitHub Models free tier (24 req/60s) cannot sustain — the script
detects that up front and tells you to re-run with ``--llm haiku``. Rerun the
live eval afterwards:

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


def _score_value(result) -> float:
    """Numeric percentage from a DSPy EvaluationResult (or a bare float)."""
    return float(getattr(result, "score", result))


_RATE_LIMIT_BANNER = (
    "\n" + "=" * 72 + "\n⚠  KILLED BY THE GITHUB MODELS PER-MINUTE RATE LIMIT (e.g. 24 req/60s).\n"
    "   The free tier cannot sustain a DSPy compile, which fires many\n"
    "   requests in a short window. Left to run, MIPROv2 silently scores the\n"
    "   throttled trials 0.0 and saves a no-op artifact that would DOWNGRADE\n"
    "   your classifier.\n\n"
    "   Fix: re-run the whole compile on Claude Haiku 4.5 (much higher rate\n"
    "   limits), which needs ANTHROPIC_API_KEY:\n\n"
    "       python backend/scripts/compile_intent_classifier.py --llm haiku\n\n"
    "   Running end-to-end on one model is required anyway — a DSPy optimizer\n"
    "   tunes the prompt for one specific LM, so the baseline eval,\n"
    "   optimization, and final eval must all use the same model.\n" + "=" * 72
)


def _burst_probe(n: int = 30) -> bool:
    """Fire a rapid burst of tiny calls to detect a *per-minute* rate cap.

    A single preflight ping cannot reveal a sustained-throughput limit — only a
    burst can. A DSPy compile issues many requests in a short window, so if the
    primary endpoint throttles under this burst we flag it in seconds instead of
    discovering it ~10 minutes into a compile (where MIPROv2 quietly scores the
    throttled trials 0.0 and saves a no-op). Returns True iff the endpoint
    returns a rate-limit error under load.
    """
    from app.config import get_settings
    from openai import OpenAI, RateLimitError

    s = get_settings()
    client = OpenAI(base_url=s.llm_base_url, api_key=s.github_token)
    for _ in range(n):
        try:
            client.chat.completions.create(
                model=s.llm_model,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
        except RateLimitError:
            return True
        except Exception:
            # Not what this probe is for — let the real compile surface it.
            return False
    return False


def _prompt_for_anthropic_key() -> str | None:
    """Interactively obtain an Anthropic API key when ``--llm haiku`` has none.

    Offers the .env key (press Enter) or accepts a pasted key (hidden input).
    Returns None if nothing is available — e.g. non-interactive stdin (CI) with
    no key configured — so the caller can abort cleanly instead of hanging.
    """
    import getpass

    from app.config import get_settings

    existing = get_settings().anthropic_api_key
    if not sys.stdin.isatty():
        return existing or None
    if existing:
        resp = input(
            "Press Enter to use ANTHROPIC_API_KEY from your .env, " "or paste a different key: "
        ).strip()
        return resp or existing
    return getpass.getpass("Paste your ANTHROPIC_API_KEY (input hidden): ").strip() or None


def _make_anthropic_lm(api_key: str):  # -> dspy.LM
    """DSPy LM bound to Claude Haiku 4.5 with an explicit key."""
    import dspy
    from app.config import get_settings

    return dspy.LM(
        f"anthropic/{get_settings().fallback_llm_model}",
        api_key=api_key,
        temperature=0.0,
        max_tokens=1024,
    )


def _resolve_llm(tag: str):
    """Build the single LM for the whole run from the --llm tag.

    Returns (lm, label, is_github) or None if a required key is unavailable.
    'gpt' → GitHub Models (config.yaml llm.model); 'haiku' → Claude Haiku 4.5,
    using ANTHROPIC_API_KEY from .env or an interactive prompt if unset.
    """
    from app.agents.dspy_classifier import make_lm
    from app.config import get_settings

    s = get_settings()
    if tag == "haiku":
        key = s.anthropic_api_key
        if not key:
            key = _prompt_for_anthropic_key()
            if not key:
                return None
            import os

            os.environ["ANTHROPIC_API_KEY"] = key
        return _make_anthropic_lm(key), f"Claude Haiku 4.5 ({s.fallback_llm_model})", False
    return make_lm(), f"GitHub Models ({s.llm_model})", True


def _is_noop_compile(compiled, program) -> bool:
    """True if optimization changed nothing (no demos, instructions unchanged).

    A rate-limit storm makes MIPROv2 score every optimized trial 0.0 and fall
    back to the seed program: it 'succeeds' and saves an artifact that is
    byte-for-byte the un-optimized baseline. Deploying that would replace the
    hand-written prompt with a weaker one, so we detect and refuse to save it.
    """
    try:
        demos = getattr(compiled.predict, "demos", [])
        compiled_instr = str(compiled.predict.signature.instructions).strip()
        base_instr = str(program.predict.signature.instructions).strip()
        return not demos and compiled_instr == base_instr
    except Exception:
        return False


def _run_compile(dspy, evaluate, program, trainset, optimizer_name, dry_run, auto_budget):
    """Baseline eval → optimize → compiled eval, all on the configured LM.

    Returns (baseline, score, compiled); (baseline, None, None) for a dry run.
    Provider errors (rate limit / auth) propagate to the caller's handler.
    """
    baseline = evaluate(program)
    logger.info("Baseline (uncompiled) dev accuracy: %s", baseline)
    if dry_run:
        return baseline, None, None

    if optimizer_name == "mipro":
        # auto budget controls search breadth: "light" ≈ 3 instruction / 6
        # few-shot candidates, "heavy" ≈ 18 / 18 — far more combinations tried.
        optimizer = dspy.MIPROv2(metric=_intent_match, auto=auto_budget)
        compiled = optimizer.compile(program, trainset=trainset, requires_permission_to_run=False)
    else:
        optimizer = dspy.BootstrapFewShot(
            metric=_intent_match,
            max_bootstrapped_demos=8,
            max_labeled_demos=8,
        )
        compiled = optimizer.compile(program, trainset=trainset)

    score = evaluate(compiled)
    return baseline, score, compiled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--llm",
        choices=["gpt", "haiku"],
        default="gpt",
        help=(
            "Model for the ENTIRE run — baseline eval, optimization, and final "
            "eval all use this one model (a DSPy-compiled prompt is tuned for a "
            "specific LM, so the whole before/after must be single-model). "
            "'gpt' = GitHub Models (config.yaml llm.model, e.g. gpt-4o-mini); "
            "'haiku' = Claude Haiku 4.5 (needs ANTHROPIC_API_KEY). Serve the "
            "compiled artifact on the SAME model you compile with."
        ),
    )
    parser.add_argument(
        "--optimizer",
        choices=["mipro", "bootstrap"],
        default="mipro",
        help="mipro = MIPROv2 (instruction + demo search); bootstrap = BootstrapFewShot (cheaper)",
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="heavy",
        help=(
            "MIPROv2 search budget: light ≈ 3 instruction / 6 few-shot "
            "candidates, heavy ≈ 18 / 18 (many more combinations, more LLM "
            "calls, longer). Ignored for --optimizer bootstrap."
        ),
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
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help=(
            "Concurrent evaluation threads. Keep at 1 for rate-limited free "
            "tiers (e.g. GitHub Models 24 req/60s); raise for higher-limit "
            "providers to speed the compile up."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")

    try:
        import dspy
        from dspy.evaluate import Evaluate
    except ImportError:
        print("dspy is not installed. Run:  pip install -e '.[dspy]'", file=sys.stderr)
        return 1

    from app.agents.dspy_classifier import ARTIFACT_PATH, build_program

    # Single model for the ENTIRE run, chosen up front via --llm. A DSPy
    # optimizer tunes the prompt for one specific LM, so baseline eval,
    # optimization, and final eval must all use the same model for the
    # before/after delta to mean anything. Serve the artifact on this model too.
    resolved = _resolve_llm(args.llm)
    if resolved is None:
        print(
            "--llm haiku needs an Anthropic key (ANTHROPIC_API_KEY in .env or "
            "typed at the prompt). Aborting.",
            file=sys.stderr,
        )
        return 3
    active_lm, provider, is_github = resolved

    # Flag an unsustainable free-tier rate limit up front (seconds, not minutes)
    # and stop — the fix is to re-run with `--llm haiku`, not to swap mid-run.
    # Only meaningful for the GitHub Models tier; Haiku is not the 24/60s tier.
    if is_github and not args.skip_auth_check:
        try:
            _preflight_llm_auth()
        except Exception as exc:
            if _is_auth_error(exc):
                print(
                    "Authentication to GitHub Models failed (401). Check GITHUB_TOKEN.",
                    file=sys.stderr,
                )
                return 2
            if not _is_rate_limit_error(exc):
                print(f"LLM preflight check failed: {exc}", file=sys.stderr)
                return 1
            print(_RATE_LIMIT_BANNER, file=sys.stderr)
            return 3
        logger.info("Probing sustained request rate on %s …", provider)
        if _burst_probe():
            print(_RATE_LIMIT_BANNER, file=sys.stderr)
            return 3

    trainset, devset = _load_examples()
    logger.info("Dataset: %d train / %d dev", len(trainset), len(devset))

    dspy.configure(lm=active_lm)
    logger.info(
        "Compiling on %s (optimizer=%s, auto=%s, num_threads=%d)",
        provider,
        args.optimizer,
        args.auto if args.optimizer == "mipro" else "n/a",
        args.num_threads,
    )
    program = build_program()
    evaluate = Evaluate(
        devset=devset,
        metric=_intent_match,
        display_progress=True,
        num_threads=args.num_threads,
        max_errors=3,
    )
    try:
        baseline, score, compiled = _run_compile(
            dspy, evaluate, program, trainset, args.optimizer, args.dry_run, args.auto
        )
    except Exception as exc:
        if _is_auth_error(exc):
            print("Authentication failed during compile. Check credentials.", file=sys.stderr)
            return 2
        if _is_rate_limit_error(exc):
            if is_github:
                print(_RATE_LIMIT_BANNER, file=sys.stderr)
            else:
                print(
                    "Claude Haiku is also rate-limited — retry later or lower " "--num-threads.",
                    file=sys.stderr,
                )
            return 3
        raise

    if args.dry_run:
        logger.info(
            "Baseline dev accuracy: %.1f%% on %s (n=%d dev, --dry-run, no compile)",
            _score_value(baseline),
            provider,
            len(devset),
        )
        return 0

    baseline_pct = _score_value(baseline)
    compiled_pct = _score_value(score)
    print(
        "\n" + "=" * 72 + f"\n  DSPy compile result — {provider}\n"
        f"    baseline (uncompiled) : {baseline_pct:.1f}%\n"
        f"    compiled              : {compiled_pct:.1f}%\n"
        f"    delta                 : {compiled_pct - baseline_pct:+.1f} pp "
        f"  (held-out dev split, n={len(devset)})\n" + "=" * 72
    )

    # A no-op compile (empty demos, unchanged instruction) means the optimizer
    # returned the seed program. Throttling is already ruled out upstream by the
    # burst probe, so on a clean run this is a legitimate "no improvement found"
    # outcome — not corruption. Either way, saving a bare-signature artifact
    # gives the runtime no gain and would shadow the hand-written prompt, so we
    # skip the save and report honestly.
    if _is_noop_compile(compiled, program):
        print(
            f"\nMIPROv2 found no candidate that beat the baseline "
            f"({baseline_pct:.1f}%) — it returned the un-optimized program, so "
            "there is nothing to save (a no-op artifact would not improve the "
            "runtime classifier). The baseline is already strong; to push "
            "further, raise the optimizer budget (auto='light' → 'medium' in "
            "this script) or add more/harder labelled examples.",
            file=sys.stderr,
        )
        return 0

    compiled.save(str(ARTIFACT_PATH))
    logger.info("Saved compiled program → %s", ARTIFACT_PATH)
    logger.info(
        "Compiled with %s — serve on the SAME model for the artifact to behave "
        "as optimized. The orchestrator loads it automatically; verify with:  "
        "pytest backend/tests/integration/test_intent_eval_live.py -v -s",
        provider,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
