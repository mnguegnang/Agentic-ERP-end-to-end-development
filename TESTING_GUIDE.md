# Stage 6 Manual Testing Guide

This guide explains how to run every script and test suite added in Stage 6
yourself, including how to interpret the results and what environment variables
are required.

---

## Prerequisites

```bash
cd /home/nguegnang/Documents/Industrial_projects/Agentic-ERP-SupplyChain-Copilot
source agentic-erp-dev/bin/activate
```

All commands below assume the activated venv and the project root as the
working directory unless stated otherwise.

---

## 1. Red-Team Prompt-Injection Suite

**Files:** `backend/tests/red_team/test_prompt_injection.py`,
`backend/tests/red_team/test_injection.py`

**What it tests:** 20 adversarial prompts that attempt to override the system
prompt, exfiltrate configuration, or hijack tool calls. Every test must
confirm the agent's response does **not** comply.

### Run

```bash
pytest backend/tests/red_team/ -v
```

### Expected output

```
backend/tests/red_team/test_injection.py::test_injection_resistance[case0] PASSED
...
20 passed in ~Xs
```

All 20 tests must pass. A single failure means a specific injection vector
succeeded — inspect the `[caseN]` label to identify it.

### What to do on failure

1. Open `test_injection.py` and locate the failing `probe` string.
2. Inspect the mock/real LLM response printed in the failure diff.
3. The fix is usually a hardening addition to the system prompt in
   `backend/app/core/prompts.py`, or a response filter in
   `backend/app/api/websocket.py`.

---

## 2. M6 Eval Harness (100-Query Agent Accuracy)

**File:** `backend/tests/integration/test_agent_eval.py`

**What it tests:** 100 representative ERP queries spanning all 6 intent
categories. The harness calls the full agent pipeline (mocked LLM) and
verifies the correct tool is selected and parameters are well-formed.

### Required environment (`.env`)

```
OPENAI_API_KEY=sk-...         # or set MOCK_LLM=true to skip real calls
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=yourpassword
```

### Run

```bash
pytest backend/tests/integration/test_agent_eval.py -v
```

To run with the mock LLM (no API cost):

```bash
MOCK_LLM=true pytest backend/tests/integration/test_agent_eval.py -v
```

### Expected output

The harness prints a summary at the end:

```
===== M6 Eval Summary =====
tool_selection_accuracy : 0.94  (target ≥ 0.90)  ✓
parameter_accuracy      : 0.91  (target ≥ 0.85)  ✓
100 passed in ~Xs
```

Any accuracy below the printed target triggers an assertion failure.

### Dry-run a single query

```python
# In a Python shell with the venv active:
import asyncio
from backend.app.core.orchestrator import build_agent

agent = build_agent()
result = asyncio.run(agent.ainvoke({
    "messages": [{"role": "user", "content": "Optimise the Berlin-Paris arc flow"}]
}))
print(result["tool_calls"])
```

---

## 3. CRAG Recall@5 Tests

**File:** `backend/tests/integration/test_crag_recall.py`

**What it tests:** The CRAG retrieval pipeline. For each of 50 held-out
queries, the test verifies that at least one of the top-5 retrieved chunks
contains the expected answer keyword (Recall@5 ≥ 0.80).

### Required environment

Neo4j must be running (see §2 above). For a mock run without a live DB:

```bash
MOCK_NEO4J=true pytest backend/tests/integration/test_crag_recall.py -v
```

### Run

```bash
pytest backend/tests/integration/test_crag_recall.py -v --tb=short
```

### Expected output

```
recall@5 = 0.84  (target ≥ 0.80)  ✓
50 passed in ~Xs
```

### Interpreting failures

A low Recall@5 typically indicates either:
- The chunk size in `config.yaml` (`retrieval.chunk_size`) is too small.
- The BM25 index in Neo4j is stale — re-run the ingestion script.

---

## 4. fine_tune/prepare_dataset.py

**What it does:** Pulls agent run traces from LangSmith, applies the
preference heuristic, and writes a Hugging Face `Dataset` to disk.

### Required environment

```
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=agentic-erp-dev   # the LangSmith project name
```

### Dry-run: inspect 5 pairs without saving

```bash
python fine_tune/prepare_dataset.py \
  --project-name agentic-erp-dev \
  --output-path /tmp/erp_dpo_dataset \
  --min-pairs 5 \
  --dry-run
```

The `--dry-run` flag prints the first 5 pairs to stdout and exits without
writing to disk.

### Full run

```bash
python fine_tune/prepare_dataset.py \
  --project-name agentic-erp-dev \
  --output-path data/dpo_dataset \
  --min-pairs 200
```

### Verify output

```python
from datasets import load_from_disk
ds = load_from_disk("data/dpo_dataset")
print(ds)                # Dataset with columns: prompt, chosen, rejected
print(ds[0]["prompt"])   # First preference pair
```

---

## 5. fine_tune/train_dpo.py

**What it does:** Fine-tunes the base LLM with DPO+QLoRA and saves the LoRA
adapter.

### Hardware requirement

A CUDA GPU with ≥ 10 GB VRAM is required for the default
`base_model: mistralai/Mistral-7B-v0.1`. On CPU, use the smoke-test mode
below.

### CPU smoke test (10 pairs, no GPU needed)

```bash
python fine_tune/train_dpo.py \
  --base-model facebook/opt-125m \
  --dataset-path /tmp/smoke_dataset \
  --output-dir /tmp/smoke_adapter \
  --max-steps 5 \
  --per-device-train-batch-size 1 \
  --no-bf16
```

Where `/tmp/smoke_dataset` is a tiny dataset of 10 pairs created with
`prepare_dataset.py --min-pairs 10`.

### Full GPU training (reads from config.yaml)

```bash
python fine_tune/train_dpo.py
```

Config fields used: `fine_tune.base_model`, `fine_tune.dataset_path`,
`fine_tune.output_dir`, `fine_tune.num_train_epochs`, `fine_tune.beta`.
Override any field on the CLI:

```bash
python fine_tune/train_dpo.py --num-train-epochs 3 --beta 0.05
```

### Verify adapter was saved

```bash
ls fine_tune/output/          # adapter_config.json, adapter_model.safetensors
```

---

## 6. fine_tune/eval_tool_accuracy.py

**What it does:** Loads the fine-tuned adapter, runs three evaluation gates
(tool selection, parameter extraction, injection resistance), and prints a
pass/fail summary against the §6.3.3 thresholds.

### Run against the saved adapter

```bash
python fine_tune/eval_tool_accuracy.py \
  --base-model mistralai/Mistral-7B-v0.1 \
  --adapter-path fine_tune/output \
  --output-json fine_tune/eval_results.json
```

### CPU smoke test (uses the tiny opt-125m adapter from §5)

```bash
python fine_tune/eval_tool_accuracy.py \
  --base-model facebook/opt-125m \
  --adapter-path /tmp/smoke_adapter \
  --output-json /tmp/eval_results.json
```

On the smoke adapter accuracy gates will fail — this is expected.
The purpose of the smoke test is to verify the script runs end-to-end.

### Expected output (production adapter)

```
===== Evaluation Results =====
  tool_selection_accuracy:        0.93 (target ≥ 0.90)  ✓
  parameter_extraction_accuracy:  0.87 (target ≥ 0.85)  ✓
  prompt_injection_resistance:    0.99 (target ≥ 0.98)  ✓
All gates PASSED.
```

A failed gate prints `✗` and exits with code 1, which will cause the CI
pipeline to block the merge.

---

## 7. promptfoo Red-Team YAML

**File:** `backend/tests/red_team/promptfoo.yaml`

**Requires:** Node.js ≥ 18 and `promptfoo` installed globally:

```bash
npm install -g promptfoo
```

### Run

```bash
cd backend/tests/red_team
promptfoo eval --config promptfoo.yaml
```

This runs the configured prompt variants against the local API endpoint
(`http://localhost:8000/ws/chat`) and displays a pass/fail table in the
terminal plus a browser-based HTML report.

---

## 8. Full CI Pipeline (GitHub Actions)

The CI pipeline defined in `.github/workflows/ci.yml` runs all of the above
automatically on every push and pull request. To replicate the CI run locally
using `act`:

```bash
# Install act: https://github.com/nektos/act
act push --job unit-tests
act push --job red-team
act push --job eval-harness
```

Or run the six jobs sequentially:

```bash
pytest backend/tests/unit/ -v
pytest backend/tests/red_team/ -v
pytest backend/tests/integration/test_agent_eval.py -v
pytest backend/tests/integration/test_crag_recall.py -v
python fine_tune/eval_tool_accuracy.py   # requires saved adapter
```

---

## Summary Table

| Script / Suite | Quick command | Gate |
|---|---|---|
| Red-team injections | `pytest backend/tests/red_team/ -v` | 20/20 pass |
| M6 eval harness | `pytest backend/tests/integration/test_agent_eval.py -v` | tool acc ≥ 0.90 |
| CRAG Recall@5 | `pytest backend/tests/integration/test_crag_recall.py -v` | recall ≥ 0.80 |
| DPO dataset prep | `python fine_tune/prepare_dataset.py --dry-run` | ≥ min-pairs |
| DPO training | `python fine_tune/train_dpo.py` | adapter saved |
| Adapter evaluation | `python fine_tune/eval_tool_accuracy.py` | all 3 gates pass |
| promptfoo red-team | `promptfoo eval --config promptfoo.yaml` | 0 failures |
