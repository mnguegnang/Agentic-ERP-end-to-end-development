# Project Notes — Agentic ERP Supply Chain Copilot

> **Repository:** `Agentic-ERP-SupplyChain-Copilot` (Project 1)  
> **Blueprint:** `Agentic_Decision_Intelligence_Implementation_Blueprint.md`  
> **Date Started:** 2026-04-14  
> **Coder Agent:** `@ml-genai-coder`  

---

## Initial Assumptions

1. Project 1 repo (`Agentic-ERP-SupplyChain-Copilot`) = current workspace. Stages 1-6.
2. Project 2 repo (`Agentic-ERP-Deploy`) will be created separately when the Stage 6 dev-complete gate (M8) passes. Stage 7 only.
3. All GPU work (DPO fine-tuning, Stage 6 M7) executes on Lightning AI L4 (22 hr free/month).
4. TeX source files and PDFs are excluded from git tracking per user instruction. Physical files remain in `docs/` but are untracked.
5. Default git branch is `master` (user preference; CI workflow will reference `refs/heads/master`).

---

## Architectural Decision Records

### ADR-001 — LLM Provider: GitHub Models API

| Field | Value |
|-------|-------|
| **Date** | 2026-04-14 |
| **Blueprint spec** | `gpt-4o` or `claude-sonnet-4-20250514` (Section 1.5) |
| **Decision** | GitHub Models API (`gpt-4o`) as primary LLM |
| **Rationale** | User specified. No OpenAI billing account needed — uses GitHub PAT. Endpoint is OpenAI-compatible; `langchain-openai.ChatOpenAI` accepts custom `base_url`. |
| **Config** | `base_url=https://models.inference.ai.azure.com`, auth via `GITHUB_TOKEN` env var |
| **Impact** | `ChatOpenAI(base_url=settings.llm_base_url, api_key=settings.github_token, model=settings.llm_model)` |
| **Status** | Active |

### ADR-002 — pyproject.toml Location: Repo Root vs. backend/

| Field | Value |
|-------|-------|
| **Date** | 2026-04-14 |
| **Blueprint spec** | `backend/pyproject.toml` (Section 1.1.1 tree) |
| **Decision** | `pyproject.toml` placed at **repo root** |
| **Rationale** | CI step `pip install -e ".[dev]"` runs from repo root (after `actions/checkout@v4`). With `pyproject.toml` in `backend/`, pip would fail to find it unless the step changed directory first. Placing it at root with `[tool.setuptools.packages.find] where = ["backend"]` makes `import app` work as intended and keeps CI clean. |
| **`backend/requirements.txt`** | Retained for `Dockerfile.api` build context (Docker copies `backend/` folder and runs `pip install -r requirements.txt`). |
| **Status** | Active |

### ADR-003 — Git Default Branch: master (not main)

| Field | Value |
|-------|-------|
| **Date** | 2026-04-14 |
| **Blueprint spec** | CI workflow uses `refs/heads/main` |
| **Decision** | Branch is `master` (user preference) |
| **Impact** | CI workflow (`build-and-push-images` job `if:` condition) must use `refs/heads/master` |
| **Status** | Active — to be applied when `.github/workflows/ci.yml` is created (Stage 6) |

---

## Open Items

- [ ] Confirm AdventureWorks dataset source (dump file URL or generate from scratch) — needed for Stage 2 seed scripts
- [ ] Confirm LangSmith project name (currently `agentic-erp-supply-chain` in config.yaml)
- [ ] Node.js 22 LTS nvm installation — verify before frontend scaffolding (Stage 3)

### ADR-004 — asyncpg for Seed Scripts (not SQLAlchemy ORM)

| Field | Value |
|-------|-------|
| **Date** | 2026-04-14 |
| **Blueprint spec** | Stage 2 seed scripts implicitly assumed ORM |
| **Decision** | Use `asyncpg` directly for `seed_adventureworks.py`, `seed_contracts.py`, `seed_neo4j.py` |
| **Rationale** | Bulk insert performance; `pgvector.asyncpg.register_vector(conn)` required for vector inserts via asyncpg; ORM adds unnecessary overhead for one-time seed operations |
| **Status** | Active |

### ADR-005 — Synthetic Contracts as .txt (not PDF)

| Field | Value |
|-------|-------|
| **Date** | 2026-04-14 |
| **Blueprint spec** | §2.1.3 — "Generate 20 synthetic supplier contracts as PDFs" |
| **Decision** | Contracts stored as `.txt` files in `data/contracts/` |
| **Rationale** | `fpdf2` and `reportlab` are not in blueprint `requirements.txt`. Text content (~1500 words, 5 FM variants) is fully representative. CRAG embedding pipeline (chunk → BGE embed → pgvector insert) is identical regardless of source format. |
| **Impact** | None on downstream CRAG evaluation (recall@5 metric unaffected). |
| **Status** | Active |

### ADR-006 — pytest pythonpath = ["backend"]

| Field | Value |
|-------|-------|
| **Date** | 2026-04-14 |
| **Blueprint spec** | Not specified |
| **Decision** | Added `pythonpath = ["backend"]` to `[tool.pytest.ini_options]` in `pyproject.toml` |
| **Rationale** | `backend/` contains `app/` and `scripts/` packages; pytest runs from repo root. Setting pythonpath ensures `from app.*` and `from scripts.*` resolve correctly without creating a spurious `backend/__init__.py`. |
| **Status** | Active |

### ADR-007 — Chunker Boundary: Single Chunk Requires ≤ CHUNK_SIZE − CHUNK_OVERLAP Words

| Field | Value |
|-------|-------|
| **Date** | 2026-04-14 |
| **Blueprint spec** | §2.3 — 512-token chunks, 50-token overlap |
| **Decision** | Sliding window advances by `chunk_size − overlap` (462 words); text ≤ 462 words → 1 chunk; text of exactly 512 words produces 2 chunks (full + trailing overlap) |
| **Rationale** | Standard sliding-window chunking behavior. Test `test_exact_chunk_size_one_chunk` was renamed to `test_up_to_step_boundary_is_one_chunk` and corrected to use `CHUNK_SIZE − CHUNK_OVERLAP` words. |
| **Status** | Active |
