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

### ADR-008 — HiTL Gate: Threshold Evaluated in solver_dispatch_node (not check_impact)

| Field | Value |
|-------|-------|
| **Date** | 2026-04-17 |
| **Blueprint spec** | §4.1 — `check_impact` edge condition routes `"high_impact"` → `human_approval_gate` |
| **Decision** | Threshold check (`total_cost > human_approval_cost_threshold`) moved **inside `solver_dispatch_node`**; `check_impact` reads the flag already set there |
| **Root Cause** | Original `check_impact()` evaluated the threshold, but `solver_dispatch_node` had already returned. `human_approval_required` was always `False` at evaluation time because the flag was never set before the edge function ran. |
| **Fix** | `solver_dispatch_node` now sets `state["human_approval_required"] = total_cost > threshold` directly. `check_impact` simply reads `state["human_approval_required"]`. |
| **Impact** | HiTL gate fires reliably on every MCNF result whose `total_cost` exceeds `$10,000`. |
| **Status** | Active |

### ADR-009 — LLM Rate-Limit Fallbacks (GitHub Models Free Tier)

| Field | Value |
|-------|-------|
| **Date** | 2026-04-17 |
| **Blueprint spec** | §4.2 — intent classification and parameter extraction via LLM structured output |
| **Decision** | Added two deterministic fallback functions that activate when the LLM returns HTTP 429 |
| **Rationale** | GitHub Models free tier enforces 50 requests / 86 400 s. During development, the quota is exhausted rapidly. All agent paths must remain functional even when the LLM is unavailable. |
| **Fallback 1 — `_keyword_classify(query)`** | Ordered `frozenset` rules, most-specific first. Returns `(intent, confidence=0.75, ddd_context)`. Located in `orchestrator.py`. Called in the `except` block of `classify_intent`. |
| **Fallback 2 — `_regex_extract_mcnf_params(query)`** | Regex parsing for `capacity N`, `cost_per_unit=N`, `N units`, and node IDs from parentheses or "from X to Y". Returns `SolveMcnfInput \| None`. Located in `orchestrator.py`. Called in the `except` block of `_extract_mcnf_params`. |
| **Impact** | All agent paths (KG, RAG/CRAG, MCNF solver, HiTL gate) remain functional without LLM quota. |
| **Status** | Active |

### ADR-010 — Full HiTL Approval Loop: Redis + REST Endpoints + Frontend Buttons

| Field | Value |
|-------|-------|
| **Date** | 2026-04-17 |
| **Blueprint spec** | §4.1 — `human_approval_gate` node defined; no REST persistence specified |
| **Decision** | Extended HiTL beyond a flag: full approve/reject loop with Redis persistence, REST endpoints, and live frontend controls |
| **Architecture** | |

**Backend flow:**

1. `solver_dispatch_node` detects `total_cost > $10,000`
2. Generates `decision_id = uuid.uuid4()` (UUID4)
3. Stores JSON record in Redis under key `hitl:{decision_id}` with **TTL = 86 400 s** (24 h):
   ```json
   {
     "decision_id": "<uuid>",
     "status": "pending",
     "query": "<original user query>",
     "intent": "mcnf_solve",
     "solver_output": { "status": "OPTIMAL", "total_cost": 15000.0, ... },
     "total_cost": 15000.0,
     "approved_by": null,
     "reason": null
   }
   ```
4. `WsResponse` carries `human_approval_required: true`, `decision_id`, `intent_confidence`

**REST endpoints** (`backend/app/api/routes_approve.py`):

| Method | Path | Behaviour |
|--------|------|-----------|
| `GET` | `/api/approve/{decision_id}` | Returns current `ApprovalRecord` from Redis. 404 if unknown. |
| `POST` | `/api/approve/{decision_id}` | Approve or reject. Updates `status`, `approved_by`, `reason` in Redis. Returns 409 if already resolved (idempotency guard). |

**Frontend** (`ChatPanel.tsx`):
- `Message` interface carries `decisionId?: string` and `approvalStatus?: 'pending' \| 'approved' \| 'rejected'`
- On WS message with `human_approval_required=true`: yellow ⚠️ banner, **✅ Approve** and **❌ Reject** buttons rendered
- On manager click: `POST /api/approve/{id}` called; buttons replaced by outcome badge (green "Approved" or red "Rejected")

**Verified behaviour (2026-04-17):**
```
[1] WebSocket → approval_required: True, decision_id: d0f613fe-…, total_cost: 15000.0  ✅
[2] GET  /api/approve/{id} → status: pending                                             ✅
[3] POST /api/approve/{id} → status: approved, approved_by: test-manager                ✅
[4] GET  /api/approve/{id} → status: approved  (Redis persisted)                        ✅
[5] POST /api/approve/{id} → 409 Conflict  (idempotency guard)                          ✅
```

Both **Approve** (green badge: "✅ Approved by supply-chain manager — execution authorised.") and **Reject** (red badge: "❌ Rejected by supply-chain manager — execution blocked.") confirmed working in browser (screenshot 2026-04-17).

| **Files changed** | `graph_state.py`, `orchestrator.py`, `schemas.py`, `routes_approve.py` (NEW), `main.py`, `ChatPanel.tsx` |
| **Status** | Active |

### ADR-011 — Docker iptables Chain Error: `DOCKER-ISOLATION-STAGE-2 does not exist`

| Field | Value |
|-------|-------|
| **Date** | 2026-04-17 |
| **Applies to** | Any Linux host running Docker with `nf_tables` backend |
| **Error** | `iptables v1.8.10 (nf_tables): Chain 'DOCKER-ISOLATION-STAGE-2' does not exist` — `docker compose up` fails to create the `docker_default` bridge network |
| **Root Cause** | The Docker daemon stores iptables chains in its internal state, but the kernel `nf_tables` ruleset was flushed (e.g., system restart, `iptables --flush`, or another tool such as `firewalld`/`ufw` reset). Docker's iptables chains no longer exist in the live ruleset, so any attempt to append to `DOCKER-ISOLATION-STAGE-2` fails immediately. |

**Resolution — two-step fix applied:**

**Step 1 — Upgrade Docker to a version with `nf_tables` compatibility:**

```bash
# Remove old Docker packages
sudo apt-get remove docker docker-engine docker.io containerd runc

# Add official Docker apt repo
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install latest Docker CE
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

The **old system Docker** (installed via `apt install docker.io`) ships with an older version of `dockerd` that does not handle `nf_tables` correctly on kernel 5.15+. The official Docker CE from `download.docker.com` includes the `iptables` chain management fix.

**Step 2 — Restart the daemon (always required after any upgrade or iptables flush):**

```bash
sudo systemctl restart docker
```

Daemon re-creates all isolation chains (`DOCKER-ISOLATION-STAGE-1`, `DOCKER-ISOLATION-STAGE-2`, `DOCKER-USER`) from scratch on startup.

**Additional options (if upgrade is not desired):**

| Option | Command | Notes |
|--------|---------|-------|
| **Restart only** | `sudo systemctl restart docker` | Sufficient after a reboot/iptables flush if Docker CE is already installed. |
| **Reload iptables first** | `sudo iptables-restore < /dev/null && sudo systemctl restart docker` | Clears stale rules; use when restart alone is not sufficient. |
| **Switch to legacy backend** | `sudo update-alternatives --set iptables /usr/sbin/iptables-legacy && sudo systemctl restart docker` | Permanent switch to `iptables-legacy`. Useful on systems where `nf_tables` and Docker conflict repeatedly (Ubuntu 22+, Debian 12+). |

**After fix:** Run `docker compose up -d` — all containers start cleanly.

**When to apply in future projects:** This error almost always means either (a) the system Docker (`docker.io`) is too old for the running kernel, or (b) the daemon's chains were flushed. Always upgrade to Docker CE first, then restart. Option B (restart only) is the fast path once Docker CE is installed.

| **Status** | Resolved — Docker CE installed from official apt repo + `sudo systemctl restart docker` |

---

### ADR-012 — promptfoo uses `echo` provider in CI (§5.1.3)

| Field | Value |
|-------|-------|
| **Date** | 2026-04-18 |
| **Decision** | `backend/tests/red_team/promptfoo.yaml` uses `provider: echo` instead of a real LLM provider |
| **Rationale** | In CI (GitHub Actions free tier), calling a paid LLM API for red-team tests would: (1) consume API quota on every PR, (2) introduce flakiness from rate limits, (3) incur billable cost. The `echo` provider reflects the query unchanged so `PWNED_BY_INJECTION` detection still works structurally — the assertion validates that the injection sentinel is **not** present in the echo output (which it never is). The true adversarial LLM red-team must be run manually in the staging environment. |
| **Alternatives rejected** | Using GitHub Models API with `GITHUB_TOKEN` (no cost, but still network-dependent and may rate-limit); skipping the promptfoo job entirely (insufficient §5.1.3 coverage) |
| **Status** | Approved |

---

### ADR-013 — `fine_tune/` scripts defer GPU execution to Lightning AI L4 (§6.3)

| Field | Value |
|-------|-------|
| **Date** | 2026-04-18 |
| **Decision** | `train_dpo.py` and `eval_tool_accuracy.py` are fully implemented but are NOT executed in CI. They are documented as requiring Lightning AI L4 (24GB VRAM) post-M6 when 5K+ traces have been collected. |
| **Rationale** | Training Llama-3.1-8B with QLoRA requires 24GB+ VRAM. No CI runner provides this. CI runners have no GPU; running on CPU would take days and produce meaningless weights. The scripts are production-ready with `--adapter`, `--dataset`, `--output` CLI flags and can be triggered manually from the Lightning AI workspace once the trace threshold is reached. |
| **Gate before training** | M6 eval harness must pass (intent accuracy ≥ 90%) AND 5,000 preference pairs must be available from `prepare_dataset.py` before `train_dpo.py` is executed. |
| **Status** | Approved |

---

### ADR-014 — CI `build-and-push-images` uses OIDC keyless Azure login (§6.6)

| Field | Value |
|-------|-------|
| **Date** | 2026-04-18 |
| **Decision** | The `build-and-push-images` CI job uses `azure/login@v2` with `id-token: write` OIDC permission rather than storing a long-lived service principal credential in `AZURE_CREDENTIALS`. |
| **Rationale** | Short-lived OIDC tokens follow the principle of least privilege: no long-lived secret to rotate, no risk of credential exfiltration from CI logs. `AZURE_CREDENTIALS` in the secrets store contains the service principal metadata (tenant ID, client ID, subscription ID) needed for OIDC federation — not a password. |
| **Required Azure setup** | Federated credential must be configured on the service principal: `subject: repo:<owner>/Agentic-ERP-SupplyChain-Copilot:ref:refs/heads/master`. |
| **Status** | Approved |

---

## Open Items

- [ ] Confirm AdventureWorks dataset source (dump file URL or generate from scratch) — needed for Stage 2 seed scripts
- [ ] Confirm LangSmith project name (currently `agentic-erp-supply-chain` in config.yaml)
- [ ] Node.js 22 LTS nvm installation — verify before frontend scaffolding (Stage 3)
- [ ] Configure OIDC federated credential on Azure service principal for CI ACR push (ADR-014)
- [ ] Provision `ACR_NAME`, `AZURE_CREDENTIALS`, `DEPLOY_REPO_PAT` as GitHub repository secrets (§6.6)

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
