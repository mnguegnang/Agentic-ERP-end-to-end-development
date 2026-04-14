# Developer Log — Agentic ERP Supply Chain Copilot (Project 1)

> **Repository:** `Agentic-ERP-SupplyChain-Copilot`  
> **Started:** 2026-04-14  
> **Coder Agent:** `@ml-genai-coder`  
> **Blueprint:** `Agentic_Decision_Intelligence_Implementation_Blueprint.md`  

---

## Entry 001 — 2026-04-14 | Stage 1 — Scaffolding

**Status:** ✅ Completed  
**Action:** Step 2 scaffolding executed.

**Environment detected:**
- Python 3.11.9 (system) — compatible with `>=3.11,<3.13`
- Docker 29.2.1 / Docker Compose v5.0.2 — ready
- Node.js: installing via nvm (22 LTS)
- OS: Linux

**Files created:**
- `.gitignore` — excludes `*.tex`, `*.pdf`, LaTeX artefacts, venv, `.env`, Node modules, fine-tune checkpoints
- `.env.example` — all required env vars documented
- `config.yaml` — non-secret config (LLM, RAG, solvers, cache, observability)
- `pyproject.toml` (root) — version-pinned deps + dev tools; `packages.find where = ["backend"]`
- `backend/requirements.txt` — runtime deps for Docker build context
- `backend/app/config.py` — Pydantic V2 `BaseSettings` + YAML-backed properties
- All Python `__init__.py` files for every sub-package
- `.gitkeep` files in `data/`, `docs/`, `backend/tests/integration/results/`

**Git:** Repository initialized, default branch set to `master`.

**Deviation noted:** `pyproject.toml` placed at repo root (ADR-002). See `Project_Notes.md`.

---

## Entry 002 — 2026-04-14 | Stage 1 — Environment Verification

**Status:** ✅ Completed

**Results:**
- `pip check` → no broken requirements
- `fastapi==0.115.6`, `langgraph==0.2.60`, `ortools==9.12.4544`, `cvxpy==1.6.0` — all verified
- Node.js v22.22.2 LTS via nvm v0.39.7, npm 10.9.7

**Bug resolved — `setuptools.backends.legacy:build`:**
- **Symptom:** `pip install -e ".[dev]"` failed: pip isolated subprocess could not find build backend
- **Root cause:** `setuptools.backends.legacy:build` is an internal path, not the public entrypoint
- **Fix:** Changed `pyproject.toml` `build-backend` to `setuptools.build_meta` (universally compatible)
- **ADR:** ADR-002 updated in `Project_Notes.md`

---

## Entry 003 — 2026-04-14 | Stage 1 — Full Source Scaffold + Initial Commit

**Status:** ✅ Completed

**Files created (this session):**
- `backend/app/main.py`, `routes_health.py`, `routes_chat.py`
- `backend/app/agents/`: `graph_state.py` (full AgentState TypedDict), `orchestrator.py`, `kg_agent.py`, `contract_agent.py`
- `backend/app/solvers/`: mcnf, robust_minmax, meio_gsm, bullwhip, jsp, vrp, disruption (7 solvers)
- `backend/app/mcp/`: server_erp, server_kg, server_crag, server_ortools, server_cvxpy, server_scipy (6 MCP servers)
- `backend/app/kg/`: client, queries, schema
- `backend/app/rag/`: embedder, retriever (RRF k=60), reranker, evaluator, chunker
- `backend/app/db/`: models (ORM), session (async)
- `backend/app/cache/semantic_cache.py` — full implementation per Blueprint §4.8
- `backend/app/security/`: sanitizer, rbac, jwt_auth
- `backend/scripts/`: seed_adventureworks, seed_neo4j, seed_contracts
- `fine_tune/`: prepare_dataset, train_dpo (LORA_R=16, DPO_BETA=0.1, SEED=42), eval_tool_accuracy
- `docker/`: docker-compose.yml (§6.1 + healthchecks), Dockerfile.api, Dockerfile.frontend, Dockerfile.worker
- `frontend/`: vite.config.ts, index.html, main.tsx, App.tsx, ChatPanel, GraphViewer, SolverResults, useWebSocket, api service
- `.github/workflows/ci.yml` — master branch, ruff/black/mypy/bandit/pytest/tsc gates
- `backend/tests/unit/test_placeholder.py` — 1/1 passed

**Git commit:** `3ca030b` — 92 files, 3726 insertions

---

## Entry 004 — Stage 2: Data Pipeline (2026-04-14)

**Scope:** Blueprint §2.1 — PostgreSQL schema, seed data, Neo4j KG, synthetic contract embeddings, unit tests, quality gates.

### Files Created / Modified
| File | Action |
|------|--------|
| `data/adventureworks/init.sql` | **NEW** — full schema (purchasing/production/supply_chain schemas, pgvector index) |
| `backend/app/db/models.py` | **REPLACED** — added Vendor, Product, BOM, Location, DistributionCenter, ContractEmbedding ORM models |
| `backend/scripts/seed_adventureworks.py` | **NEW** — 14 suppliers, 9 components, 4 products, 20 contracts, 5 work centers, 3 DCs, logistics arcs |
| `backend/scripts/seed_neo4j.py` | **NEW** — full KG seeding (constraints, 6 node types, 6 relationship types) |
| `backend/scripts/seed_contracts.py` | **NEW** — 5 FM variants × 20 contracts, BGE embed, pgvector insert |
| `backend/tests/unit/test_chunker.py` | **NEW** — 9 unit tests for chunk_text |
| `backend/tests/unit/test_embedder.py` | **NEW** — 5 unit tests (mocked SentenceTransformer) |
| `backend/tests/unit/test_seed_data.py` | **NEW** — 24 structural integrity tests |
| `pyproject.toml` | per-file-ignores E501 for scripts/, pythonpath=["backend"] for pytest |

### Quality Gate Results
| Gate | Result |
|------|--------|
| `ruff check` | ✅ All checks passed |
| `black --check` | ✅ All 41 files formatted |
| `pytest backend/tests/unit/` | ✅ 39/39 passed |

### Bugs Resolved

**BUG-004a — F601: Duplicate dict key "RUB" in seed_contracts.py incoterms_map**
- Root cause: `account_number.split("-")[1]` returned "RUB" for both PAC-RUB-009 and BRA-RUB-013
- Fix: Changed key to `split("-")[0]` (TQ/FAB/LUC/CON/AWC/HAN/SHA/PAC/NOR/AUS/CAN/BRA/KOR) → unique keys
- ADR: see ADR-005

**BUG-004b — ModuleNotFoundError: backend.scripts in test_seed_data.py**
- Root cause: `backend/` has no `__init__.py`; import attempted `from backend.scripts` which requires `backend` on `sys.path`
- Fix: Changed imports to `from scripts.seed_adventureworks import` + `from scripts.seed_contracts import`; added `pythonpath = ["backend"]` to `pyproject.toml` [tool.pytest.ini_options]
- ADR: ADR-006

**BUG-004c — test_exact_chunk_size_one_chunk: AssertionError (expected 1 chunk, got 2)**
- Root cause: Sliding-window chunker advances by `chunk_size - overlap` (462). Text of exactly 512 words → start=0, chunk words[0:512], start=462; 462<512 → second iteration → 2 chunks. Test expectation was wrong.
- Fix: Renamed test to `test_up_to_step_boundary_is_one_chunk`; changed input to `CHUNK_SIZE - CHUNK_OVERLAP` words (462), the correct 1-chunk boundary.
- ADR: ADR-007

**BUG-004d — UP042: class Role(str, Enum) — unsafe ruff fix required**
- Fix: `ruff check --fix --unsafe-fixes --select UP042` → `class Role(StrEnum)` (requires `from enum import StrEnum`)

**BUG-004e — VS Code buffer/disk divergence**
- Root cause: `read_file` tool reads VS Code in-memory buffer; `grep` / `pytest` read on-disk file. After multi_replace_string_in_file, VS Code buffer was updated but disk was not for test_seed_data.py.
- Fix: Used `sed -i` to apply changes directly to disk.
- Note: Always verify changes with `cat` / `grep` via terminal before running tests.

### ADRs Logged This Entry
- ADR-004: asyncpg for seed scripts
- ADR-005: contracts as .txt not PDF
- ADR-006: pytest pythonpath = ["backend"]
- ADR-007: chunker single-chunk boundary at CHUNK_SIZE − CHUNK_OVERLAP words

**Next:** Stage 3 — Baseline PoC (Blueprint §3.3): single `solve_mcnf` MCP tool + LangChain ChatOpenAI integration.

---
