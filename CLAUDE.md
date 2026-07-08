# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agentic ERP Supply Chain Copilot — a multi-agent system that answers natural-language supply-chain queries using a LangGraph orchestrator, OR solvers, CRAG (RAG with relevance evaluation), and a Neo4j knowledge graph. A Human-in-the-Loop gate fires when solver costs exceed $10,000.

## Environment Setup

Python venv is at `agentic-erp-dev/` in the repo root. Always activate it before running Python commands:

```bash
source agentic-erp-dev/bin/activate
```

Python requirement: `>=3.11,<3.13`. The `pyproject.toml` lives at the repo root (not inside `backend/`) so that `pip install -e ".[dev]"` works from CI and sets `PYTHONPATH=backend` via pytest config.

Copy `.env.example` → `.env` and fill in `GITHUB_TOKEN`, `PG_PASSWORD`, `NEO4J_PASSWORD`, `JWT_SECRET_KEY`, and optionally `LANGSMITH_API_KEY`. Non-secret runtime config is in `config.yaml` (committed).

## Commands

### Backend

```bash
pip install -e ".[dev]"                          # install all deps (runtime + dev)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000   # run API (from repo root, venv active)
```

### Frontend

```bash
cd frontend
npm ci           # install (use ci, not install)
npm run dev      # Vite dev server on port 3000
npm run build    # tsc && vite build
npm run lint     # tsc --noEmit (TypeScript type-check only; no ESLint)
```

### Docker (full local stack)

```bash
cd docker
docker compose up -d      # postgres:5432, neo4j:7474/7687, redis:6379, api:8000, frontend:3000
docker compose logs -f api
```

After services are healthy, seed all three databases:

```bash
python backend/scripts/seed_adventureworks.py
python -m backend.scripts.seed_neo4j   # must run as a module: it imports backend.scripts.seed_adventureworks,
                                        # which only resolves when the repo root is on sys.path
python backend/scripts/seed_contracts.py
```

### Tests

```bash
# Unit tests (no external services needed)
pytest backend/tests/unit/ -v --tb=short

# Single test file or case
pytest backend/tests/unit/test_mcnf.py::test_solve_mcnf_basic -v

# Integration tests (requires live Postgres + Neo4j + Redis)
pytest backend/tests/integration/ -v --tb=short -m "not slow"

# Agent eval harness (100-query routing gate; fully mocked, no LLM cost)
MOCK_LLM=true pytest backend/tests/integration/test_agent_eval.py -v

# LIVE intent-classification eval (hits the real LLM; nightly/pre-release, needs GITHUB_TOKEN)
pytest backend/tests/integration/test_intent_eval_live.py -v -s

# CRAG Recall@5 gate; use MOCK_NEO4J=true to skip live graph
MOCK_NEO4J=true pytest backend/tests/integration/test_crag_recall.py -v

# Red-team injection tests
pytest backend/tests/red_team/ -v --tb=short

# promptfoo adversarial suite (needs: npm install -g promptfoo@0.100.0)
cd backend/tests/red_team && promptfoo eval --config promptfoo.yaml --output promptfoo-results.json --no-cache
```

### Lint / Format / Type-check

```bash
ruff check backend/ fine_tune/           # lint
ruff check --fix backend/ fine_tune/     # lint + auto-fix
black --check backend/ fine_tune/        # format check
black backend/ fine_tune/                # format
mypy backend/app --ignore-missing-imports
bandit -r backend/app -ll                # security scan (skips B101)
```

## Architecture

### Request Flow

```
Browser (WebSocket /ws/chat)
    → FastAPI (routes_chat.py) → semantic cache check (exact + embedding cosine ≥0.95)
    → LangGraph Orchestrator (agents/orchestrator.py), thread_id per query
         ├── classify_intent  → structured LLM call → one of 10 intents
         │                      fallback: keyword rules on HTTP 429
         ├── route_by_intent (conditional edge)
         │    ├── "contract_query"  → contract_agent_node  (CRAG pipeline)
         │    ├── solver intents    → solver_dispatch_node  (deterministic OR)
         │    └── everything else   → kg_agent_node → solver_dispatch_node
         ├── solver_dispatch_node — structured LLM param extraction per solver
         │    └── total_cost > $10k → decision_id (= thread_id) stored in Redis
         ├── check_impact (conditional edge)
         │    ├── high_impact → human_approval_gate — interrupt() PAUSES the graph;
         │    │                 POST /api/approve/{id} resumes it with the decision
         │    └── low_impact  → synthesize_response
         └── synthesize_response → WsResponse back to browser (cached if low-stakes)
```

### Key Subsystems

**LangGraph state** — `agents/graph_state.py` defines `GraphState` (TypedDict). Every node reads/writes this dict; the conditional edges branch on its fields.

**Intent classification** — 10 intents: `mcnf_solve`, `jsp_schedule`, `vrp_route`, `robust_allocate`, `meio_optimize`, `bullwhip_analyze`, `disruption_resource`, `kg_query`, `contract_query`, `multi_step`. Confidence threshold is `0.7` (from `config.yaml`); below it the orchestrator asks the user to clarify. Two classifier paths: if `agents/compiled_intent_classifier.json` exists (produced by `python backend/scripts/compile_intent_classifier.py`, needs `pip install -e ".[dspy]"`), the DSPy-compiled program is used; otherwise the zero-shot structured-output prompt. The DSPy path is strict opt-in — missing package or artifact falls through silently.

**CRAG pipeline** (`agents/contract_agent.py`) — BGE-large-en-v1.5 embed → pgvector cosine + BM25 (index cached in-process, 10 min TTL) → RRF@60 fusion → CrossEncoder rerank (ms-marco-MiniLM-L-12-v2) → per-document LLM relevance labels in one batched call. Irrelevant chunks are dropped individually; if every chunk is irrelevant the query is rewritten once (`rag/evaluator.rewrite_query`) and retrieval retried before returning `no_answer`.

**KG Think-on-Graph** (`agents/kg_agent.py`) — extracts entities via LLM, selects from a hardcoded relation whitelist, issues parameterized Cypher against Neo4j (never raw string interpolation), returns structured triples.

**OR solvers** — seven deterministic solvers dispatched by intent:
| Solver | Library | Intent |
|---|---|---|
| MCNF | OR-Tools | `mcnf_solve` |
| JSP | OR-Tools CP-SAT | `jsp_schedule` |
| VRP | OR-Tools | `vrp_route` |
| Disruption | OR-Tools | `disruption_resource` |
| Robust min-max | CVXPY | `robust_allocate` |
| MEIO/GSM | CVXPY | `meio_optimize` |
| Bullwhip | SciPy | `bullwhip_analyze` |

**Human-in-the-Loop** — the graph is compiled with a `MemorySaver` checkpointer; `human_approval_gate` calls `interrupt()`, so high-cost runs genuinely pause before any answer is synthesized. The `decision_id` equals the LangGraph `thread_id` and is stored in Redis (24 h TTL) for audit. `POST /api/approve/{id}` requires the manager password (`MANAGER_APPROVAL_PASSWORD` in `.env`; wrong → 403 and the decision stays pending, unset → 503 locked) and then calls `resume_orchestrator()`, which resumes the paused run with the manager's decision and returns the synthesized `final_response` in the approval record. Caveat: the checkpointer is in-process — an API restart drops paused runs (the Redis record survives, but the run cannot be resumed).

**Semantic cache** (`cache/semantic_cache.py`) — Redis-backed, TTL 3600 s, checked in `routes_chat.py` before the graph runs. Two levels: exact SHA-256 match on the normalized query, then cosine ≥ 0.95 between BGE query embeddings. A semantic hit additionally requires a **lexical guard** to pass — the two queries must share the same numbers and entity codes — so parametric queries ("Allocate 400 units" vs "1000 units", "supplier 4" vs "5") are never answered from each other's cached result. Only low-stakes responses are cached (never approval-gated, error/degraded, or `unclear`-intent responses).

**MCP tool servers** (`mcp/`) — six FastMCP servers form the EXTERNAL tool interface (the in-process orchestrator calls solver functions directly; the MCP tools delegate to those same functions, verified by `tests/unit/test_mcp_servers.py`): `server_erp.py` (ORM queries), `server_kg.py` (Neo4j Cypher, whitelisted), `server_crag.py` (contract search), `server_ortools.py` (MCNF/JSP/VRP/Disruption), `server_cvxpy.py` (Robust/MEIO), `server_scipy.py` (Bullwhip). Run one standalone over stdio with e.g. `python -m app.mcp.server_ortools` (from `backend/`, venv active) to plug it into Claude Code or the MCP inspector.

### Databases

| Store | Purpose |
|---|---|
| PostgreSQL 16 + pgvector | AdventureWorks ERP schema + contract embeddings |
| Neo4j 5.27 | Supply-chain knowledge graph |
| Redis 7.4 | Semantic cache + HiTL decision store |

### LLM Provider

GitHub Models API (GPT-4o via `https://models.inference.ai.azure.com`). Requires `GITHUB_TOKEN` in `.env`. Configured in `config.yaml` under `llm.*`.

## CI Pipeline

Five jobs in `.github/workflows/ci.yml` (trigger: push/PR to `master`):

1. `backend-quality` — ruff + black + mypy + bandit + unit tests
2. `frontend-quality` — `tsc --noEmit`
3. `integration-tests` — needs job 1; uses live service containers
4. `red-team` — needs job 1; pytest red_team/ + promptfoo
5. `build-and-push` — needs all four; master only; builds Docker images, pushes to Azure Container Registry (OIDC auth), then triggers deployment via `repository_dispatch` to the `Agentic-ERP-Deploy` repo

Required secrets: `AZURE_CREDENTIALS`, `ACR_NAME`, `DEPLOY_REPO_PAT`.

## Fine-tuning

Scripts under `fine_tune/` run on Lightning AI L4 (not locally). Flow: LangSmith trace curation → `prepare_dataset.py` → `train_dpo.py` (DPO + QLoRA, LoRA r=16 alpha=32, beta=0.1) → `eval_tool_accuracy.py`. Not part of the normal dev loop.


# Project Context when working on web (claude.ai/code)
## Repo Structure
This project depends on two external repos:
- **Agents repo**: `../agents-repo/` — contains custom subagent .md files
- **Prompts repo**: `../prompts-repo/` — contains task-specific prompts invoked by agents

## Custom Agents
Agents are located in `.claude/agents/` (symlinked or copied from the agents repo).
Always check the agents repo before implementing any fix — there may be a specialized
agent (e.g. code-review-improve, ai-research-scientist) that should handle the task.

## Workflow
1. First diagnose the CI failure 
2. Identify which agent is appropriate for the fix
3. Invoke that agent explicitly using @agent-name
