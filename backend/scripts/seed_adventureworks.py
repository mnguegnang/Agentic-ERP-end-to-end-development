"""Seed AdventureWorks OLTP + supply-chain extension tables (Blueprint §2.1.1, §3.3).

Stage 2 implementation: run once after `docker compose up` to populate PostgreSQL.

Usage:
    python -m backend.scripts.seed_adventureworks
"""
from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger(__name__)


async def main() -> None:
    """Load AdventureWorks schema + seed 14 suppliers with logistics arcs."""
    # TODO Stage 2:
    #   1. Connect via asyncpg to DATABASE_URL
    #   2. Execute adventureworks/init.sql (AdventureWorks OLTP schema)
    #   3. CREATE SCHEMA supply_chain; CREATE TABLE supply_chain.supplier_tiers ...
    #   4. Insert 14 synthetic suppliers + logistics arcs
    logger.info("seed_adventureworks: NOT_IMPLEMENTED — Stage 2 task")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
