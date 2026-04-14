"""Populate Neo4j KG from AdventureWorks ERP (Blueprint §2.1.2).

Stage 2 implementation: run after seed_adventureworks.py.

Node types created:
  (:Supplier), (:Component), (:Product), (:WorkCenter),
  (:DistributionCenter), (:Contract)

Relationships created:
  PROVIDES, USED_IN, PROCESSED_AT, SHIPS_TO, BOUND_BY, SUPPLIED_BY

Usage:
    python -m backend.scripts.seed_neo4j
"""

from __future__ import annotations

import asyncio
import logging

from app.config import get_settings
from neo4j import AsyncGraphDatabase

from backend.scripts.seed_adventureworks import (
    BOM_ROWS,
    COMPONENTS,
    CONTRACT_DEFS,
    DISTRIBUTION_CENTERS,
    PRODUCTS,
    SUPPLIER_PROVIDES,
    SUPPLIERS,
    WORK_CENTERS,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node / relationship creation helpers
# ---------------------------------------------------------------------------

_CONSTRAINTS = [
    "CREATE CONSTRAINT supplier_id IF NOT EXISTS FOR (s:Supplier) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT component_id IF NOT EXISTS FOR (c:Component) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT product_id IF NOT EXISTS FOR (p:Product) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT wc_id IF NOT EXISTS FOR (w:WorkCenter) REQUIRE w.id IS UNIQUE",
    "CREATE CONSTRAINT dc_id IF NOT EXISTS FOR (d:DistributionCenter) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT contract_id IF NOT EXISTS FOR (x:Contract) REQUIRE x.id IS UNIQUE",
]


async def _create_constraints(session) -> None:  # type: ignore[type-arg]
    for cypher in _CONSTRAINTS:
        await session.run(cypher)
    logger.info("Constraints ensured.")


async def _seed_suppliers(session, supplier_ids: list[int]) -> None:
    """Merge (:Supplier) nodes and (:Supplier)-[:SUPPLIED_BY]->(:Supplier) arcs."""
    for i, (sup, vid) in enumerate(zip(SUPPLIERS, supplier_ids)):
        await session.run(
            """
            MERGE (s:Supplier {id: $id})
            SET s.name = $name,
                s.country = $country,
                s.tier = $tier,
                s.reliability = $reliability,
                s.account_number = $account_number
            """,
            id=vid,
            name=sup["name"],
            country=sup["country_code"],
            tier=sup["tier_level"],
            reliability=float(sup["reliability_score"]),
            account_number=sup["account_number"],
        )
    # SUPPLIED_BY (multi-tier hierarchy)
    for i, (sup, vid) in enumerate(zip(SUPPLIERS, supplier_ids)):
        if sup["parent_idx"] is not None:
            parent_vid = supplier_ids[sup["parent_idx"]]
            await session.run(
                """
                MATCH (child:Supplier {id: $child_id})
                MATCH (parent:Supplier {id: $parent_id})
                MERGE (child)-[:SUPPLIED_BY {tier_level: $tier}]->(parent)
                """,
                child_id=vid,
                parent_id=parent_vid,
                tier=sup["tier_level"],
            )
    logger.info("Seeded %d Supplier nodes.", len(SUPPLIERS))


async def _seed_components(session, component_ids: list[int]) -> None:
    """Merge (:Component) nodes."""
    for comp, cid in zip(COMPONENTS, component_ids):
        await session.run(
            """
            MERGE (c:Component {id: $id})
            SET c.name = $name,
                c.product_number = $product_number,
                c.unit_cost = $unit_cost
            """,
            id=cid,
            name=comp["name"],
            product_number=comp["product_number"],
            unit_cost=float(comp["standard_cost"]),
        )
    logger.info("Seeded %d Component nodes.", len(COMPONENTS))


async def _seed_products(session, product_ids: list[int]) -> None:
    """Merge (:Product) nodes."""
    for prod, pid in zip(PRODUCTS, product_ids):
        await session.run(
            """
            MERGE (p:Product {id: $id})
            SET p.name = $name,
                p.product_number = $product_number,
                p.list_price = $list_price,
                p.weight = $weight
            """,
            id=pid,
            name=prod["name"],
            product_number=prod["product_number"],
            list_price=float(prod["list_price"]),
            weight=float(prod["weight"]),
        )
    logger.info("Seeded %d Product nodes.", len(PRODUCTS))


async def _seed_work_centers(session) -> None:
    """Merge (:WorkCenter) nodes (use 1-based index as KG id)."""
    for i, wc in enumerate(WORK_CENTERS, start=1):
        await session.run(
            """
            MERGE (w:WorkCenter {id: $id})
            SET w.name = $name,
                w.capacity_hours = $availability
            """,
            id=i,
            name=wc["name"],
            availability=float(wc["availability"]),
        )
    logger.info("Seeded %d WorkCenter nodes.", len(WORK_CENTERS))


async def _seed_distribution_centers(session, dc_ids: list[int]) -> None:
    """Merge (:DistributionCenter) nodes."""
    for dc, dc_id in zip(DISTRIBUTION_CENTERS, dc_ids):
        await session.run(
            """
            MERGE (d:DistributionCenter {id: $id})
            SET d.name = $name,
                d.region = $region,
                d.country = $country
            """,
            id=dc_id,
            name=dc["name"],
            region=dc["region"],
            country=dc["country_code"],
        )
    logger.info("Seeded %d DistributionCenter nodes.", len(DISTRIBUTION_CENTERS))


async def _seed_contracts(
    session, contract_ids: list[int], supplier_ids: list[int]
) -> None:
    """Merge (:Contract) nodes and (:Supplier)-[:BOUND_BY]->(:Contract)."""
    for i, (sup_idx, eff, exp) in enumerate(CONTRACT_DEFS):
        cid = contract_ids[i]
        vid = supplier_ids[sup_idx]
        await session.run(
            """
            MERGE (x:Contract {id: $id})
            SET x.effective_date = $eff,
                x.expiry_date = $exp,
                x.supplier_id = $supplier_id
            """,
            id=cid,
            eff=eff,
            exp=exp,
            supplier_id=vid,
        )
        await session.run(
            """
            MATCH (s:Supplier {id: $supplier_id})
            MATCH (x:Contract {id: $contract_id})
            MERGE (s)-[:BOUND_BY]->(x)
            """,
            supplier_id=vid,
            contract_id=cid,
        )
    logger.info("Seeded %d Contract nodes.", len(CONTRACT_DEFS))


async def _seed_relationships(
    session,
    supplier_ids: list[int],
    component_ids: list[int],
    product_ids: list[int],
) -> None:
    """Create PROVIDES, USED_IN, PROCESSED_AT, SHIPS_TO relationships."""

    # PROVIDES: (:Supplier)-[:PROVIDES]->(:Component)
    for sup_idx, comp_idx in SUPPLIER_PROVIDES:
        vid = supplier_ids[sup_idx]
        cid = component_ids[comp_idx]
        comp = COMPONENTS[comp_idx]
        await session.run(
            """
            MATCH (s:Supplier {id: $sup_id})
            MATCH (c:Component {id: $comp_id})
            MERGE (s)-[r:PROVIDES]->(c)
            SET r.cost = $cost, r.capacity = 10000
            """,
            sup_id=vid,
            comp_id=cid,
            cost=float(comp["standard_cost"]),
        )

    # USED_IN: (:Component)-[:USED_IN {quantity}]->(:Product)
    for prod_idx, comp_idx, qty in BOM_ROWS:
        pid = product_ids[prod_idx]
        cid = component_ids[comp_idx]
        await session.run(
            """
            MATCH (c:Component {id: $comp_id})
            MATCH (p:Product {id: $prod_id})
            MERGE (c)-[r:USED_IN]->(p)
            SET r.quantity = $qty
            """,
            comp_id=cid,
            prod_id=pid,
            qty=qty,
        )

    # PROCESSED_AT: (:Component)-[:PROCESSED_AT]->(:WorkCenter)
    # Assign each component to a work centre based on category
    component_wc_map = {
        0: 1,  # bearing → Frame Assembly Bay
        1: 2,  # gear ctrl → Electronics Bay
        2: 1,  # frame → Frame Assembly Bay
        3: 3,  # grip → Final Assembly
        4: 3,  # brake → Final Assembly
        5: 2,  # motor → Electronics Bay
        6: 1,  # chainring → Frame Assembly Bay
        7: 3,  # tire → Final Assembly
        8: 1,  # fork → Frame Assembly Bay
    }
    for comp_idx, wc_idx in component_wc_map.items():
        cid = component_ids[comp_idx]
        await session.run(
            """
            MATCH (c:Component {id: $comp_id})
            MATCH (w:WorkCenter {id: $wc_id})
            MERGE (c)-[:PROCESSED_AT {duration_hours: 0.5}]->(w)
            """,
            comp_id=cid,
            wc_id=wc_idx,
        )

    # SHIPS_TO: (:Product)-[:SHIPS_TO]->(:DistributionCenter)
    # All products ship to all three DCs
    for prod_idx in range(len(PRODUCTS)):
        pid = product_ids[prod_idx]
        for dc_idx, (transit_days, cost) in enumerate(
            [(3, 0.20), (7, 0.55), (14, 0.80)], start=1
        ):
            await session.run(
                """
                MATCH (p:Product {id: $prod_id})
                MATCH (d:DistributionCenter {id: $dc_id})
                MERGE (p)-[r:SHIPS_TO]->(d)
                SET r.cost = $cost, r.transit_days = $transit
                """,
                prod_id=pid,
                dc_id=dc_idx,
                cost=cost,
                transit=transit_days,
            )

    logger.info("Seeded PROVIDES, USED_IN, PROCESSED_AT, SHIPS_TO relationships.")


# ---------------------------------------------------------------------------
# Retrieve IDs already seeded by seed_adventureworks.py via PostgreSQL
# ---------------------------------------------------------------------------


async def _fetch_pg_ids() -> (
    tuple[list[int], list[int], list[int], list[int], list[int]]
):
    """Return (supplier_ids, component_ids, product_ids, dc_ids, contract_ids)."""
    import asyncpg

    settings = get_settings()
    dsn = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
    conn: asyncpg.Connection = await asyncpg.connect(dsn)
    try:
        supplier_ids = [
            row["business_entity_id"]
            for row in await conn.fetch(
                "SELECT business_entity_id FROM purchasing.vendor ORDER BY business_entity_id"
            )
        ]
        component_ids = [
            row["product_id"]
            for row in await conn.fetch(
                "SELECT product_id FROM production.product "
                "WHERE product_subcategory_id = 1 ORDER BY product_id"
            )
        ]
        product_ids_raw = [
            row["product_id"]
            for row in await conn.fetch(
                "SELECT product_id FROM production.product "
                "WHERE product_subcategory_id = 2 ORDER BY product_id"
            )
        ]
        dc_ids = [
            row["dc_id"]
            for row in await conn.fetch(
                "SELECT dc_id FROM supply_chain.distribution_centers ORDER BY dc_id"
            )
        ]
        contract_ids = [
            row["contract_id"]
            for row in await conn.fetch(
                "SELECT contract_id FROM supply_chain.contracts ORDER BY contract_id"
            )
        ]
    finally:
        await conn.close()
    return supplier_ids, component_ids, product_ids_raw, dc_ids, contract_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Seed Neo4j KG. Requires seed_adventureworks.py to have run first."""
    settings = get_settings()
    logger.info("Fetching IDs from PostgreSQL…")
    supplier_ids, component_ids, product_ids, dc_ids, contract_ids = (
        await _fetch_pg_ids()
    )

    if len(supplier_ids) != len(SUPPLIERS):
        msg = (
            f"Expected {len(SUPPLIERS)} suppliers in PostgreSQL, found {len(supplier_ids)}. "
            "Run seed_adventureworks.py first."
        )
        raise RuntimeError(msg)

    logger.info("Connecting to Neo4j at %s…", settings.neo4j_uri)
    driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=("neo4j", settings.neo4j_password)
    )
    try:
        async with driver.session() as session:
            await _create_constraints(session)
            await _seed_suppliers(session, supplier_ids)
            await _seed_components(session, component_ids)
            await _seed_products(session, product_ids)
            await _seed_work_centers(session)
            await _seed_distribution_centers(session, dc_ids)
            await _seed_contracts(session, contract_ids, supplier_ids)
            await _seed_relationships(session, supplier_ids, component_ids, product_ids)
    finally:
        await driver.close()

    logger.info(
        "✅ Neo4j seed complete — %d suppliers, %d components, %d products, "
        "%d DCs, %d contracts",
        len(supplier_ids),
        len(component_ids),
        len(product_ids),
        len(dc_ids),
        len(contract_ids),
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s — %(message)s"
    )
    asyncio.run(main())
