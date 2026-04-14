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
