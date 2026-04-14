# Phase 2 — Research Findings

## Date: 2026-04-14
## Topic: Agentic Decision Intelligence System (ERP Supply Chain Copilot)

## Seed Papers (Tier 1, ≥14/20)

### OR + LLM Integration
- S1: Li et al. "LLMs for Supply Chain Optimization" (2023, Microsoft, 91 cit) — foundational
- S2: Zhang & Luo "OR-LLM-Agent" (2025, 19 cit) — automates OR modeling with LLMs
- S3: Venkatachalam "Integrating LLMs with Network Optimization for SC Planning" (2025) — exact architecture match
- S4: Wasserkrug et al. "Enhancing Decision Making: LLMs + OR" (AAAI 2025, 5 cit)

### KG + LLM (Neuro-Symbolic)
- S5: Sun et al. "Think-on-Graph" (2023) — LLM×KG paradigm
- S6: Luo et al. "Reasoning on Graphs" (2023) — faithful KG reasoning
- S7: Luo et al. "Graph-constrained Reasoning" (2024) — GCR
- S8: Liu et al. "SymAgent" (2025) — neuro-symbolic self-learning agent
- S9: Chen et al. "Plan-on-Graph" (2024) — adaptive KG planning

### CRAG / RAG
- S10: Lin "Higress-RAG" (2025) — enterprise CRAG with dual hybrid retrieval
- S11: Aghajani "FAIR-RAG" (2025) — iterative refinement for multi-hop
- S12: Sawarkar et al. "Blended RAG" (2024) — hybrid dense+sparse
- S13: Wampler et al. "Engineering the RAG Stack" (2025) — comprehensive review

### MCP Protocol
- S14: Narajala & Habler "Enterprise-Grade Security for MCP" (2025)
- S15: Luo et al. "MCP-Universe" (2025) — benchmark
- S16: Radosevich & Halloran "MCP Safety Audit" (2025) — security exploits
- S17: Bhatt et al. "ETDI" (2025) — OAuth-enhanced tool definitions

### DPO / SLM
- S18: Dettmers et al. "QLoRA" (NeurIPS 2023, 4251 cit) — quantized fine-tuning
- S19: Jung et al. "DiaTool-DPO" (2025) — DPO for tool-augmented LLMs
- S20: Chen et al. "SecAlign" (2024) — DPO for prompt injection defense
- S21: Shen et al. "Small LLMs Are Weak Tool Learners" (2024) — multi-LLM agent

### Supply Chain Specific
- S22: Guan et al. "SupChain-Bench" (2026) — LLM supply chain benchmark
- S25: Yü et al. "Bullwhip Effect of Supply Networks" (2022)
- T6: Ma & Zhao "Confidence Scoring for LLM-Generated SQL in SC" (2025)
- T7: Zhang et al. "Multi-Agent RAG for Supply Chain Knowledge Bases" (2025)

### OR Formulations
- S23: Achkar et al. "Extensions to Guaranteed Service Model for MEIO" (2023)
- S24: Lan & Berkhout "PyJobShop: OR-Tools CP-SAT" (2025)
- S26: Zhen, Kuhn & Wiesemann "Unified Theory of Robust Optimization" (2021)
- S27: Mirzaee et al. "Robust Optimization for Supplier Selection" (2022)

### Security / LLMOps
- S28: Yin et al. "PISmith: RL-based Red Teaming" (2026)
- S29: Ko et al. "7 Security Challenges in Cross-domain Multi-agent LLM" (2025, 11 cit)
- S30: Shu et al. "GenAI Multi-Agent Collaboration for Enterprise" (2024)

## Contradictions Flagged
1. Single-agent vs multi-agent for SC tasks (Xu 2026 vs Drammeh 2025) — resolved: heterogeneous tool-specialized agents justify MAS
2. MCP security (enabler vs liability) — resolved: adopt with ETDI-style extensions

## Open Questions / Research Gaps
1. No unified multi-solver dispatch under single agentic orchestrator
2. KG × OR solver integration (neo4j → adjacency matrix → linprog) unexplored
3. DPO for OR tool calling (constraint extraction from NL) unstudied
4. LLM-driven bullwhip simulation absent
5. MCP SQL/Cypher injection through tool calls under-studied

## Architectural Direction
Neuro-Symbolic Decision Intelligence: LLM handles NLU + intent + synthesis; OR solvers handle deterministic optimization; KG handles structured graph queries; MCP protocol is the typed interface layer.

## Foundational References (Pre-2023)
- Rafailov et al. "DPO" (NeurIPS 2023, ~3K+ cit)
- Lee, Padmanabhan & Whang "Bullwhip Effect" (Management Science 1997, ~8K+ cit)
- Clark & Scarf "Multi-Echelon Inventory" (1960)
- Ahuja, Magnanti & Orlin "Network Flows" (1993, textbook)
- Garey & Johnson "Computers and Intractability" (1979)
- Ben-Tal, El Ghaoui & Nemirovski "Robust Optimization" (2009)
- Toth & Vigo "The Vehicle Routing Problem" (2002/2014)
