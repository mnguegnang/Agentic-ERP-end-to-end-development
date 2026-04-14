"""DPO dataset curation from LangSmith traces (Blueprint §6.3.1).

Stage 6 implementation (post M6 — 5K+ traces collected).

Usage (on Lightning AI L4):
    python fine_tune/prepare_dataset.py
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Target: 5,000 preference pairs across 10 intents
# preferred: correct tool call + correct parameters
# dispreferred: hallucinated answer (no tool call) OR wrong tool / wrong params


def main() -> None:
    # TODO Stage 6:
    #   1. Pull runs from LangSmith via langsmith.Client()
    #   2. Filter by correctness label (human feedback or heuristic)
    #   3. Build (prompt, chosen, rejected) triples
    #   4. Save as HuggingFace datasets.Dataset
    logger.info("prepare_dataset: NOT_IMPLEMENTED — Stage 6 task")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
