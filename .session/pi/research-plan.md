# Phase 1 — Research Plan (Finalized)

## Project Name
Agentic Decision Intelligence System (Agentic ERP Supply Chain Copilot)

## Topic
Agentic LLM Orchestration × Deterministic Operations Research for Enterprise Supply Chain Management

## Learning Objective
Full professor-level lecture: formal problem formulations, complexity analysis, duality theory, convergence guarantees, neuro-symbolic integration

## User Profile
- PhD Applied Mathematics (convergence analysis of DNN training)
- Proficient in: Python, FastAPI, React/TypeScript, Azure, K8s, OR, ML

## Dataset
- Primary: Microsoft AdventureWorks (https://github.com/microsoft/sql-server-samples)
- Augmentation: Deterministic Extension Schema for multi-tier suppliers, logistics, synthetic contracts

## DDD Bounded Contexts
- Domain A (N-Tier Visibility): KG Agent + MCNF + Disruption Re-Sourcing
- Domain B (Inventory vs. Capital): MEIO + Bullwhip + JSP
- Domain C (Compliance & Risk): Contract CRAG + Robust Min-Max

## Architecture
- Frontend: React/TypeScript (chat + graphs + charts)
- Backend: FastAPI (Python) — REST/WS for UI, LangGraph orchestration, MCP hosting
- Data: PostgreSQL (ERP) + Neo4j (KG) + pgvector (contract embeddings)
- Caching: Redis (semantic cache)
- Events: Redis Streams (local) → Azure Service Bus (migration)
- Observability: OpenTelemetry + LangSmith
- CI/CD: GitHub Actions + red-team security eval (Promptfoo/Giskard)

## OR Solver Stack
1. Multi-Tier Supplier Disruption → OR-Tools CP-SAT
2. Job-Shop Scheduling → OR-Tools CP-SAT
3. Bullwhip / Inventory Optimization → SciPy
4. Robust Min-Max (Supplier Risk) → CVXPY
5. MCNF → OR-Tools / PuLP (LP)
6. MEIO → CVXPY (convex programming)
7. VRP → OR-Tools RoutingModel

## Agent Framework
- LangGraph (stateful, human-in-the-loop, tool-augmented)
- CRAG for contract retrieval (dense + BM25 + cross-encoder)
- MCP servers for all DB + solver tool exposure

## Compute
- Local: Docker Compose ($0)
- Azure: AKS (B2s), PostgreSQL Flex (B1ms), Cosmos Gremlin (free tier), AI Search (free tier)
- Budget: $170 Azure, phased deployment

## Phase Enhancement
- SLM fine-tuning: Llama 3 8B via QLoRA + DPO (post-pipeline validation)
- Training data: curated tool-call pairs from LangSmith traces

## Deliverables
- Lecture PDF (LaTeX, no page limit)
- Implementation Blueprint (Markdown)
- Video Demo (5-10 min)

## Research Scope (Phase 2)
Must cover all of:
1. Multi-agent LLM orchestration (LangGraph, tool-use patterns)
2. Knowledge Graphs + LLMs (neuro-symbolic integration)
3. CRAG / advanced RAG architectures
4. All 7 OR formulations (mathematical foundations)
5. MCP protocol for enterprise tool integration
6. DPO / SLM fine-tuning for structured output
7. Enterprise LLMOps (guardrails, caching, observability)
