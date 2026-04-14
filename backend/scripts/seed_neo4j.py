"""Populate Neo4j KG from AdventureWorks ERP (Blueprint §2.1.2).

Stage 2 implementation: run after seed_adventureworks.py.

Usage:
    python -m backend.scripts.seed_neo4j
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def main() -> None:
    """Extract ERP entities → create KG nodes + relationships in Neo4j."""
    # TODO Stage 2:
    #   1. Query AdventureWorks for vendors, components, products, work centres
    #   2. Create (:Supplier), (:Component), (:Product), (:WorkCenter), (:DistributionCenter)
    #   3. Create PROVIDES, USED_IN, PROCESSED_AT, SHIPS_TO, BOUND_BY, SUPPLIED_BY edges
    logger.info("seed_neo4j: NOT_IMPLEMENTED — Stage 2 task")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
