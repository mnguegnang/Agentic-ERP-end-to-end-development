# Improvements Guide — Usage & Testing

Eight improvements were implemented across the agentic pipeline, motivated by
*AI Engineering* (Chip Huyen) and the DSPy programming model. This guide shows,
for each one: what changed, how to use it, and how to test it.

---

## Prerequisites (once)

```bash
cd Agentic-ERP-SupplyChain-Copilot
source agentic-erp-dev/bin/activate
pip install -e ".[dev]"            # runtime + dev deps
```

For **manual / end-to-end** testing you also need the live stack and seeded data:

```bash
cd docker && docker compose up -d && cd ..
python backend/scripts/seed_adventureworks.py
python -m backend.scripts.seed_neo4j   # must run as a module (see note below)
python backend/scripts/seed_contracts.py

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000   # backend (from repo root)
cd frontend && npm ci && npm run dev                        # UI on :3000
```

> **Note on `seed_neo4j.py`**: unlike the other two seed scripts, it imports
> `backend.scripts.seed_adventureworks` — a cross-module import that only
> resolves when the repo root is on `sys.path`. Running it as a direct file
> path (`python backend/scripts/seed_neo4j.py`) puts `backend/scripts/` on
> `sys.path[0]` instead and fails with `ModuleNotFoundError: No module named
> 'backend'`. Always invoke it as `python -m backend.scripts.seed_neo4j` from
> the repo root, exactly as its own docstring says.

Everything marked **Automated test** below runs *without* any live service.

Full regression sweep (run this after any change):

```bash
pytest backend/tests/unit/ backend/tests/red_team/ -q
MOCK_LLM=true MOCK_NEO4J=true pytest backend/tests/integration/ -q -m "not slow"
ruff check backend/ fine_tune/ && black --check backend/ fine_tune/
```

---

## 1. Semantic cache — wired in, and actually semantic

**What changed** — `SemanticCache` was dead code and only did exact SHA-256
matching. It is now checked in `routes_chat.py` *before* the LangGraph run,
with two levels: exact match on the normalized query, then cosine ≥ 0.95
between BGE query embeddings (reusing the CRAG embedding model). Only
low-stakes responses are cached — approval-gated, error/degraded, and
`unclear`-intent responses never are. Redis failures degrade to a cache miss.

**Lexical guard (critical for a parametric domain)** — a cosine match is
accepted only if the two queries share the same *discriminating tokens*: the
multiset of numbers and the set of entity codes (hyphenated/alphanumeric IDs
like `TQ-Electronics`, `C-007`). Without this, "Allocate **400** units…" and
"Allocate **1000** units…" embed >0.95 alike and the cache would answer one
from the other's result (likewise "supplier **4**" vs "**5**"). Pure-language
paraphrases with the same numbers/codes still hit; parameter changes never do.

Files: `backend/app/cache/semantic_cache.py`, `backend/app/api/routes_chat.py`

**How to use** — nothing to configure; TTL comes from `config.yaml`
(`cache.ttl_seconds: 3600`). Tune the similarity threshold via the
`similarity_threshold` argument of `SemanticCache` (default 0.95 —
raise it if you ever observe a wrong answer served from cache).

**Automated test**

```bash
pytest backend/tests/unit/test_semantic_cache.py -v
```

**Manual test** (stack running)

1. In the UI ask: `Show me the supply network for TQ-Electronics` → note the latency.
2. Ask the identical question again → answer should return near-instantly (exact hit).
3. Ask a paraphrase: `Display the supplier network of TQ-Electronics` → also fast (semantic hit).
4. Verify in Redis: `docker exec -it <redis-container> redis-cli KEYS 'cache:*'` —
   you should see `cache:resp:<sha>` and `cache:vec:<sha>` pairs.
5. Ask something that triggers the approval gate (see §6) — confirm **no**
   new cache key appears (approval flows are never cached).

---

## 2. BM25 index caching (RAG latency)

**What changed** — the BM25 sparse index was rebuilt from a full-table
`SELECT` on *every* contract query. It is now cached in-process per
`supplier_id` with a 10-minute TTL; `invalidate_bm25_cache()` forces a refresh
after in-process ingestion.

Files: `backend/app/rag/retriever.py`

**How to use** — automatic. After re-running `seed_contracts.py` (a separate
process), either wait ≤ 10 min or restart the API for the new corpus to be
picked up.

**Automated test**

```bash
pytest backend/tests/unit/test_crag_retriever.py -v
```

**Manual test** — ask two different contract questions back-to-back
(e.g. `What does the force majeure clause say?` then
`What are the payment terms for supplier 3?`). The second query skips the
corpus load; with `--log-level debug` you'll see no second
"BM25 corpus" DB round-trip.

---

## 3. Live intent-classification eval (evaluates the model, not the harness)

**What changed** — the existing 100-query gate (`test_agent_eval.py`) patches
the classifier, so it only verifies routing logic. A new harness runs the same
100 labelled queries through the **real** LLM classifier via
`llm_classify_intent()` (refactored out so the keyword fallback can't mask
failures), with retry/backoff so transient 429s don't count as
misclassifications. Gate: ≥ 90 % accuracy, with a per-intent breakdown.

Files: `backend/tests/integration/test_intent_eval_live.py`,
`backend/app/agents/orchestrator.py` (`llm_classify_intent`)

**How to use / test** — needs a real `GITHUB_TOKEN` in `.env`; makes ~100 LLM
calls, so treat it as a nightly / pre-release gate, not per-PR:

```bash
pytest backend/tests/integration/test_intent_eval_live.py -v -s
```

The `-s` flag prints the per-intent accuracy table. If the >10% error-rate
guard trips (API outage / quota exhaustion), the summary also prints a
self-diagnosing error breakdown — e.g. `100x  429 RateLimitReached` or
`3x  401 unauthorized` — so the cause is visible in the test output itself,
no manual reproduction needed. It auto-skips when
`MOCK_LLM=true` or `GITHUB_TOKEN` is missing, and is excluded from the
per-PR CI job by the `slow` marker.

---

## 4. Parameter extraction for all seven solvers

**What changed** — previously only MCNF had parameter extraction; the other
six solvers were called with empty inputs and could never produce a real
answer. Every solver intent now has a typed Pydantic input schema
(`SolveJspInput`, `SolveVrpInput`, `SolveRobustInput`, `SolveMeioInput`,
`AnalyzeBullwhipInput`, `SolveDisruptionInput`) extracted from the query via
structured LLM output. If extraction fails, the solver runs with empty
defaults (trivial OPTIMAL) instead of crashing.

Files: `backend/app/api/schemas.py`, `backend/app/agents/orchestrator.py`

**Automated test**

```bash
pytest backend/tests/unit/test_solver_dispatch.py -v
```

**Manual test** — try each solver intent in the UI and check the solver panel
shows a non-trivial result:

| Intent | Example query |
|---|---|
| `jsp_schedule` | `Schedule 2 jobs: job 1 has ops on machine 0 for 3h then machine 1 for 2h; job 2 has ops on machine 1 for 4h then machine 0 for 1h. Minimise makespan.` |
| `vrp_route` | `Route 2 vehicles of capacity 100 from depot at (0,0) to customers at (10,0) demand 40, (0,10) demand 50, (10,10) demand 60.` |
| `robust_allocate` | `Allocate 400 units across supplier A (cost 10±2, capacity 300) and supplier B (cost 12±0.5, capacity 300) robustly with omega 1.` |
| `meio_optimize` | `Optimise safety stock for 2 stages: stage 0 holding cost 2, demand std 10, lead time 3; stage 1 (feeds stage 0) holding cost 1, demand std 10, lead time 5. Service level 95%.` |
| `bullwhip_analyze` | `Analyse bullwhip for demand series 100, 110, 95, 105, 120, 90, 100, 115 with lead time 2, forecast window 3, 3 echelons.` |
| `disruption_resource` | `Component C-101 is disrupted, demand 500 units. Alternatives: supplier S1 cost 8 capacity 300, supplier S2 cost 9 capacity 400. Reallocate.` |

---

## 5. CRAG: per-document relevance + corrective query rewrite

**What changed** — the old pipeline judged only the top-1 chunk and applied
that verdict to all five (one bad top hit discarded four good chunks; one good
top hit let four irrelevant ones through). Now **every** retrieved chunk is
labelled in a single batched LLM call and irrelevant chunks are dropped
individually. If *all* chunks are irrelevant, the query is rewritten once
(`rewrite_query`) and retrieval retried — the corrective action from the CRAG
paper — before returning `no_answer`. A recovered retry is tagged
`fallback="query_rewritten"`.

Files: `backend/app/rag/evaluator.py` (`evaluate_relevance_batch`,
`rewrite_query`), `backend/app/rag/retriever.py`

**Automated tests**

```bash
pytest backend/tests/unit/test_crag_retriever.py -v
MOCK_NEO4J=true pytest backend/tests/integration/test_crag_recall.py -v
```

**Manual test** — with seeded contracts, ask a vague/abbreviated contract
question (e.g. `whats the FM situation in our supplier agreements?`). Watch
the API logs for `CRAG corrective rewrite: ... → ...` when the first pass
fails, and confirm the answer cites contract text.

---

## 6. Human-in-the-Loop gate that actually pauses

**What changed** — the graph previously synthesized and delivered the full
answer *before* anyone approved anything. Now the graph compiles with a
`MemorySaver` checkpointer and `human_approval_gate` calls `interrupt()`:
a high-cost run genuinely stops, the user gets only a pending banner, and
`POST /api/approve/{decision_id}` resumes the checkpointed run with the
manager's decision. The synthesized answer then reflects approval or
rejection, and the frontend shows it as a new chat message.
`decision_id` equals the LangGraph `thread_id`; the Redis record remains the
audit trail.

**Known limitation** — the checkpointer is in-process: an API restart drops
paused runs (the Redis record survives, but the run can't be resumed; the
approval endpoint records the decision and returns `final_response: null`).

Files: `backend/app/agents/orchestrator.py`, `backend/app/api/routes_approve.py`,
`backend/app/agents/graph_state.py`, `frontend/src/App.tsx`

**Automated tests**

```bash
pytest backend/tests/integration/test_orchestrator_graph.py -v
```

(`test_run_orchestrator_pauses_on_high_cost_and_resumes_on_approval` asserts
the synthesis LLM is **never invoked** while the decision is pending.)

**Manager password** — approving OR rejecting requires the manager secret
(`MANAGER_APPROVAL_PASSWORD` in `.env`, passed to the api container by
compose). Wrong password → HTTP 403 and the decision stays pending; unset
password → HTTP 503 (approvals are locked, never open by default). The check
is constant-time (`hmac.compare_digest`) and runs before the decision record
is even loaded.

**Manual test** (stack running)

1. Ask: `Route 10000 units from factory (A) to warehouse (B). Arc capacity 20000, cost_per_unit=5. Demand at B is 10000.`
   (total cost $50,000 > $10k threshold)
2. You should get **only** the ⚠️ approval banner — no routing plan yet.
3. Type a **wrong** password in the card's password field, click **Approve** →
   inline error "Incorrect manager password", decision stays pending.
4. Type the correct `MANAGER_APPROVAL_PASSWORD`, click **Approve** → a new
   assistant message appears with the full plan, prefixed by an approval note.
   (Or **Reject** → a rejection summary with alternatives instead.)
5. Or via curl:
   `curl -X POST localhost:8000/api/approve/<decision_id> -H 'Content-Type: application/json' -d '{"approved": true, "approved_by": "you", "password": "<manager password>"}'`
   — the JSON response contains `final_response`; wrong password → 403.

**Automated test (password gate)**

```bash
pytest backend/tests/unit/test_approval_password.py -v
```

---

## 7. DSPy-compiled intent classifier (programming, not prompting)

**What changed** — the five hand-written prompts are DSPy-shaped: typed input
→ typed output. The intent classifier is the first one migrated: a DSPy
`ChainOfThought` program is compiled (MIPROv2 or BootstrapFewShot) against the
100-query labelled dataset — the same data that previously only gated tests —
and saved as `backend/app/agents/compiled_intent_classifier.json`. When that
artifact exists, `llm_classify_intent` uses it automatically; when the `dspy`
package or the artifact is absent, everything falls back to the existing
zero-shot prompt. Strictly opt-in.

Files: `backend/app/agents/dspy_classifier.py`,
`backend/scripts/compile_intent_classifier.py`, `pyproject.toml` (`[dspy]` extra)

**How to use**

```bash
pip install -e ".[dspy]"                                     # one-time
python backend/scripts/compile_intent_classifier.py --dry-run   # baseline dev accuracy, no compile
python backend/scripts/compile_intent_classifier.py             # MIPROv2 (a few hundred LLM calls)
python backend/scripts/compile_intent_classifier.py --optimizer bootstrap   # cheaper alternative
```

The script prints baseline vs. compiled dev-set accuracy and saves the
artifact. Restart the API; the compiled classifier is picked up on the first
query. To go back to the zero-shot prompt, delete the artifact JSON.

**Quota fallback (§9) applies here too, but differently.** DSPy wraps litellm,
a different stack from the request-path LangChain fallback — so the compile
job can't fall back per-call. Instead it selects the provider up front: a fast
preflight hits GitHub Models, and on a 429 (quota exhausted) the *entire*
compile runs on Claude Haiku 4.5 instead (requires `ANTHROPIC_API_KEY`; you'll
see `GitHub Models is out of quota (429) — compiling with the Anthropic
fallback model` in the logs). Without an Anthropic key, it exits with a clear
message telling you to set one or wait for the quota reset.

**Automated tests**

```bash
pytest backend/tests/unit/test_dspy_classifier.py -v     # fallback contract (no dspy needed)
pytest backend/tests/integration/test_intent_eval_live.py -v -s   # measure the compiled model (§3)
```

The intended loop: compile (§7) → measure with the live eval (§3) → keep the
artifact only if accuracy improved.

---

## 8. MCP servers — honest, runnable external interface

**What changed** — the six FastMCP servers were documented as "how the agent
gets tools" but nothing ever invoked them (the orchestrator calls solvers
directly — correct for latency). They are now explicitly the **external** tool
interface: each server has a stdio entry point, tests pin that the tools
delegate to the *same* deterministic solver functions the orchestrator uses,
and CLAUDE.md describes the boundary truthfully.

Files: `backend/app/mcp/server_*.py`, `backend/tests/unit/test_mcp_servers.py`, `CLAUDE.md`

**How to use** — run any server standalone over stdio (from `backend/`, venv active):

```bash
cd backend && python -m app.mcp.server_ortools
```

Plug into Claude Code so Claude can call your solvers directly:

```bash
claude mcp add erp-solvers -- bash -c "cd $(pwd)/backend && $(pwd)/agentic-erp-dev/bin/python -m app.mcp.server_ortools"
```

Or inspect interactively: `npx @modelcontextprotocol/inspector python -m app.mcp.server_ortools`

**Automated test**

```bash
pytest backend/tests/unit/test_mcp_servers.py -v
```

---

## 9. Quota-exhaustion LLM fallback — Claude Haiku 4.5

**What changed** — GitHub Models' free tier is small (50 req/day for gpt-4o,
150/day for gpt-4o-mini) and shared across every LLM call site (intent
classification, KG entity/relation extraction, CRAG per-document evaluation,
query rewrite, response synthesis). Once exhausted, every call fails with
HTTP 429 until the next day. Every call site now wraps its primary runnable
with `with_quota_fallback()`, which retries — **specifically and only** on
`openai.RateLimitError` — against Claude Haiku 4.5 via Anthropic. Any other
exception (validation, timeout, genuine auth failure) is untouched and
propagates exactly as before this existed.

Files: `backend/app/agents/llm_fallback.py` (the wrapper), applied in
`orchestrator.py` (4 call sites), `kg_agent.py` (2), `rag/evaluator.py` (3).
This covers the whole request path. The one code path it does NOT cover is the
offline DSPy compile script — DSPy wraps litellm, not LangChain — so that has
its own up-front provider selection (`make_fallback_lm` in `dspy_classifier.py`,
wired in `compile_intent_classifier.py`); see §7.

**How to use** — set `ANTHROPIC_API_KEY` in `.env` (get one at
https://console.anthropic.com/settings/keys). Leave it unset and the
fallback is a total no-op — identical to before this feature existed. The
fallback model is `claude-haiku-4-5-20251001`, configured in `config.yaml`
under `llm.fallback_model`.

```bash
docker compose up -d --build api   # picks up ANTHROPIC_API_KEY from .env
```

**Automated test**

```bash
pytest backend/tests/unit/test_llm_fallback.py -v
```

Covers: no-op without a key, real fallback trigger on `RateLimitError`
(using the actual `openai.RateLimitError` class, not a string match),
non-rate-limit errors propagate untouched, fallback never invoked when the
primary succeeds, and a non-`Runnable` primary (bare mock, as used
throughout this test suite) is never wrapped.

**Manual test** — hardest part to trigger live is a *real* 429 without
actually burning your quota. Two options:

1. **Cheapest**: monkeypatch the primary to fail once, live, via a Python
   REPL — see the reproduction script in `test_llm_fallback.py`
   (`test_rate_limit_error_triggers_fallback`) for the exact pattern using a
   real `openai.RateLimitError` and a real `ChatAnthropic` call.
2. **Most realistic**: exhaust gpt-4o-mini's real daily quota (150 rapid
   requests), then ask a question in the UI — the answer should still come
   back correctly (via Haiku) instead of degrading to the keyword/template
   fallback. Check the API logs for confirmation:
   ```bash
   docker logs docker-api-1 --since 5m | grep -i "claude-haiku\|anthropic"
   ```

---

## Quick reference — all test commands

```bash
# Fast, hermetic (no services, no LLM):
pytest backend/tests/unit/ -q                                     # 113 tests, all subsystems
pytest backend/tests/red_team/ -q                                 # injection suite
MOCK_LLM=true MOCK_NEO4J=true pytest backend/tests/integration/ -q -m "not slow"

# Live-model gates (need GITHUB_TOKEN, cost LLM calls):
pytest backend/tests/integration/test_intent_eval_live.py -v -s

# Quality gates:
ruff check backend/ fine_tune/
black --check backend/ fine_tune/
mypy backend/app --ignore-missing-imports
bandit -r backend/app -ll
cd frontend && npx tsc --noEmit
```

Pre-existing (not from these changes): 9 ruff `N806` naming warnings and
3 mypy errors in the solver math code.
