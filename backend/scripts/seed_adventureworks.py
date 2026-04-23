"""Seed AdventureWorks OLTP + supply-chain extension tables (Blueprint §2.1.1, §3.3).

Stage 2 implementation.

Run AFTER `docker compose up` (init.sql already executed by container).
This script is idempotent — skips inserts when rows already exist.

Usage:
    python -m backend.scripts.seed_adventureworks
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from pathlib import Path

import asyncpg
from app.config import get_settings

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).parents[2]
_INIT_SQL = _REPO_ROOT / "data" / "adventureworks" / "init.sql"

# ---------------------------------------------------------------------------
# Seed constants — 14 suppliers (Blueprint §3.3: "seed 14 suppliers")
# ---------------------------------------------------------------------------

# fmt: off
SUPPLIERS: list[dict] = [
    # --- Tier 1: 5 direct suppliers ---
    {"account_number": "TQ-ELEC-001", "name": "TQ-Electronics GmbH",       "credit_rating": 5, "tier_level": 1, "parent_idx": None, "reliability_score": 0.97, "lead_time_days": 14, "country_code": "DE"},
    {"account_number": "FAB-BEAR-002","name": "Fabrikam Bearings Inc.",      "credit_rating": 4, "tier_level": 1, "parent_idx": None, "reliability_score": 0.93, "lead_time_days": 10, "country_code": "US"},
    {"account_number": "LUC-MET-003", "name": "Lucerne Precision Metals",    "credit_rating": 5, "tier_level": 1, "parent_idx": None, "reliability_score": 0.98, "lead_time_days": 21, "country_code": "CH"},
    {"account_number": "CON-PLAS-004","name": "Contoso Polymer Solutions",   "credit_rating": 3, "tier_level": 1, "parent_idx": None, "reliability_score": 0.82, "lead_time_days": 28, "country_code": "CN"},
    {"account_number": "AWC-DIR-005", "name": "AdventureWorks Direct LLC",   "credit_rating": 5, "tier_level": 1, "parent_idx": None, "reliability_score": 0.99, "lead_time_days":  7, "country_code": "US"},
    # --- Tier 2: 5 sub-suppliers ---
    {"account_number": "TQ-SUB-006",  "name": "TQ-Sub Components GmbH",      "credit_rating": 4, "tier_level": 2, "parent_idx": 0,    "reliability_score": 0.91, "lead_time_days": 21, "country_code": "DE"},
    {"account_number": "HAN-STL-007", "name": "Hanover Steel Works GmbH",    "credit_rating": 4, "tier_level": 2, "parent_idx": 2,    "reliability_score": 0.89, "lead_time_days": 30, "country_code": "DE"},
    {"account_number": "SHA-POL-008", "name": "Shanghai Polymers Ltd.",       "credit_rating": 3, "tier_level": 2, "parent_idx": 3,    "reliability_score": 0.78, "lead_time_days": 35, "country_code": "CN"},
    {"account_number": "PAC-RUB-009", "name": "Pacific Rubber Components",   "credit_rating": 3, "tier_level": 2, "parent_idx": None,  "reliability_score": 0.85, "lead_time_days": 42, "country_code": "MY"},
    {"account_number": "NOR-PRE-010", "name": "Nordic Precision AB",          "credit_rating": 5, "tier_level": 2, "parent_idx": None,  "reliability_score": 0.95, "lead_time_days": 18, "country_code": "SE"},
    # --- Tier 3: 4 raw-material suppliers ---
    {"account_number": "AUS-IRN-011", "name": "Australian Iron Ore Co.",      "credit_rating": 4, "tier_level": 3, "parent_idx": 6,    "reliability_score": 0.87, "lead_time_days": 60, "country_code": "AU"},
    {"account_number": "CAN-ALU-012", "name": "Canadian Aluminum Corp.",      "credit_rating": 4, "tier_level": 3, "parent_idx": 2,    "reliability_score": 0.90, "lead_time_days": 45, "country_code": "CA"},
    {"account_number": "BRA-RUB-013", "name": "Borracha Brasileira SA",       "credit_rating": 3, "tier_level": 3, "parent_idx": 8,    "reliability_score": 0.80, "lead_time_days": 55, "country_code": "BR"},
    {"account_number": "KOR-ELC-014", "name": "Korea Semiconductor Ltd.",     "credit_rating": 5, "tier_level": 3, "parent_idx": 5,    "reliability_score": 0.94, "lead_time_days": 28, "country_code": "KR"},
]

# Components: product_subcategory_id = 1
COMPONENTS: list[dict] = [
    {"name": "Ball Bearing Assembly",           "product_number": "CB-001", "standard_cost":  12.50, "list_price":  18.00, "weight": 0.15},
    {"name": "Electronic Gear Control Unit",    "product_number": "CB-002", "standard_cost": 145.00, "list_price": 215.00, "weight": 0.35},
    {"name": "Aluminum Frame Tube Set",         "product_number": "CB-003", "standard_cost":  88.00, "list_price": 130.00, "weight": 1.20},
    {"name": "Polymer Handlebar Grip Pair",     "product_number": "CB-004", "standard_cost":   4.20, "list_price":   7.50, "weight": 0.12},
    {"name": "Hydraulic Brake Assembly",        "product_number": "CB-005", "standard_cost":  55.00, "list_price":  82.00, "weight": 0.45},
    {"name": "E-Drive Motor Module 250W",       "product_number": "CB-006", "standard_cost": 210.00, "list_price": 320.00, "weight": 2.10},
    {"name": "Precision Steel Chainring 44T",   "product_number": "CB-007", "standard_cost":  28.00, "list_price":  42.00, "weight": 0.30},
    {"name": "All-Terrain Rubber Tire 29in",    "product_number": "CB-008", "standard_cost":  22.00, "list_price":  35.00, "weight": 0.85},
    {"name": "Carbon Fiber Fork",               "product_number": "CB-009", "standard_cost": 175.00, "list_price": 260.00, "weight": 0.55},
]

# Finished goods: product_subcategory_id = 2
PRODUCTS: list[dict] = [
    {"name": "Mountain Bike Pro 2026",  "product_number": "FG-MTB-2026", "standard_cost": 620.00,  "list_price": 1299.00, "weight": 12.5},
    {"name": "Road Bike Elite 2026",    "product_number": "FG-RD-2026",  "standard_cost": 580.00,  "list_price": 1199.00, "weight":  9.8},
    {"name": "City Bike Standard 2026", "product_number": "FG-CT-2026",  "standard_cost": 290.00,  "list_price":  649.00, "weight": 14.2},
    {"name": "E-Bike Ultra 2026",       "product_number": "FG-EB-2026",  "standard_cost": 980.00,  "list_price": 2499.00, "weight": 22.0},
]

# BOM: (product_idx_in_PRODUCTS, component_idx_in_COMPONENTS, qty)
BOM_ROWS: list[tuple[int, int, float]] = [
    # Mountain Bike Pro: bearing, frame, brake, chainring, tire×2, fork
    (0, 0, 2.0), (0, 2, 1.0), (0, 4, 1.0), (0, 6, 1.0), (0, 7, 2.0), (0, 8, 1.0),
    # Road Bike Elite: bearing, frame, brake, chainring, tire×2, fork
    (1, 0, 2.0), (1, 2, 1.0), (1, 4, 1.0), (1, 6, 1.0), (1, 7, 2.0), (1, 8, 1.0),
    # City Bike Standard: bearing, frame, brake, grip, tire×2
    (2, 0, 2.0), (2, 2, 1.0), (2, 4, 1.0), (2, 3, 1.0), (2, 7, 2.0),
    # E-Bike Ultra: bearing, frame, brake, chainring, tire×2, fork, motor, gear ctrl
    (3, 0, 2.0), (3, 2, 1.0), (3, 4, 1.0), (3, 6, 1.0), (3, 7, 2.0), (3, 8, 1.0),
    (3, 5, 1.0), (3, 1, 1.0),
]

WORK_CENTERS: list[dict] = [
    {"name": "Frame Assembly Bay",          "cost_rate": 85.50,  "availability": 160.0},
    {"name": "Electronics Integration Bay", "cost_rate": 120.00, "availability": 120.0},
    {"name": "Final Assembly Line",         "cost_rate":  95.00, "availability": 200.0},
    {"name": "Quality Control Lab",         "cost_rate": 110.00, "availability":  80.0},
    {"name": "Packaging and Dispatch",      "cost_rate":  65.00, "availability": 200.0},
]

DISTRIBUTION_CENTERS: list[dict] = [
    {"name": "Seattle Distribution Center", "region": "North America West", "country_code": "US", "latitude":  47.6062, "longitude": -122.3321},
    {"name": "Amsterdam Distribution Hub",  "region": "Western Europe",     "country_code": "NL", "latitude":  52.3676, "longitude":    4.9041},
    {"name": "Tokyo Logistics Center",      "region": "Asia Pacific",        "country_code": "JP", "latitude":  35.6762, "longitude":  139.6503},
]

# supplier_idx → component_idx mapping (PROVIDES relationships seed)
SUPPLIER_PROVIDES: list[tuple[int, int]] = [
    (0, 1), (0, 5),   # TQ-Electronics → Gear Control, Motor
    (1, 0),           # Fabrikam → Ball Bearing
    (2, 2), (2, 6),   # Lucerne → Frame Tube, Chainring
    (3, 3),           # Contoso → Grip
    (4, 4),           # AWC Direct → Brake Assembly
    (8, 7),           # Pacific Rubber → Tire
    (9, 8),           # Nordic Precision → Carbon Fork
]

# Contracts: (supplier_idx, effective YYYY-MM-DD, expiry YYYY-MM-DD)
# 14 suppliers × mixed count = 20 contracts
CONTRACT_DEFS: list[tuple[int, str, str]] = [
    (0,  "2024-01-01", "2026-12-31"),
    (0,  "2025-01-01", "2027-12-31"),
    (1,  "2023-06-01", "2025-12-31"),
    (1,  "2025-07-01", "2027-06-30"),
    (2,  "2024-03-01", "2026-02-28"),
    (2,  "2024-09-01", "2026-08-31"),
    (3,  "2024-01-15", "2025-12-31"),
    (3,  "2025-02-01", "2027-01-31"),
    (4,  "2023-01-01", "2026-12-31"),
    (4,  "2025-06-01", "2028-05-31"),
    (5,  "2024-04-01", "2026-03-31"),
    (5,  "2025-04-01", "2027-03-31"),
    (6,  "2024-07-01", "2026-06-30"),
    (7,  "2023-10-01", "2025-09-30"),
    (8,  "2024-02-01", "2026-01-31"),
    (9,  "2024-05-01", "2026-04-30"),
    (10, "2024-08-01", "2026-07-31"),
    (11, "2024-11-01", "2026-10-31"),
    (12, "2025-01-01", "2026-12-31"),
    (13, "2025-03-01", "2027-02-28"),
]
# fmt: on

# Logistics arcs: supplier → factory (id=1) and factory → DC
# (from_type, from_id, to_type, to_id, capacity, cost_per_unit, lead_time_days)
# Tier-1 supplier IDs are 1-5 (index+1), factory ID=1, DC IDs=1-3
_FACTORY_ID = 1
_TIER1_ARCS: list[tuple] = [
    # supplier_id, capacity, cost_per_unit, lead_time (days)
    (1, 5000, 0.85, 14),  # TQ-Electronics DE → factory
    (2, 8000, 0.60, 10),  # Fabrikam US → factory
    (3, 6000, 1.20, 21),  # Lucerne CH → factory
    (4, 4000, 0.45, 28),  # Contoso CN → factory
    (5, 9000, 0.30, 7),  # AWC Direct US → factory
]
_FACTORY_DC_ARCS: list[tuple] = [
    # from_id=1 (factory), to_id=dc_id, capacity, cost_per_unit, lead_time
    (_FACTORY_ID, 1, 10000, 0.20, 3),  # factory → Seattle DC
    (_FACTORY_ID, 2, 10000, 0.55, 7),  # factory → Amsterdam DC
    (_FACTORY_ID, 3, 10000, 0.80, 14),  # factory → Tokyo DC
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_dsn(database_url: str) -> str:
    """Strip SQLAlchemy dialect prefix for raw asyncpg DSN."""
    return database_url.replace("postgresql+asyncpg://", "postgresql://")


def _split_sql(sql: str) -> list[str]:
    """Split semi-colon-delimited SQL into individual statements (no dollar-quoting)."""
    stmts = [s.strip() for s in sql.split(";") if s.strip()]
    return stmts


async def _ensure_schema(conn: asyncpg.Connection) -> None:
    """Execute init.sql to ensure all tables exist (idempotent)."""
    sql = _INIT_SQL.read_text(encoding="utf-8")
    for stmt in _split_sql(sql):
        try:
            await conn.execute(stmt)
        except asyncpg.PostgresError as exc:
            # Ignore "already exists" errors from CREATE INDEX IF NOT EXISTS
            if "already exists" not in str(exc):
                raise


async def _vendor_exists(conn: asyncpg.Connection, account_number: str) -> bool:
    row = await conn.fetchrow(
        "SELECT 1 FROM purchasing.vendor WHERE account_number = $1", account_number
    )
    return row is not None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Seed all Stage 2 data into PostgreSQL. Idempotent."""
    settings = get_settings()
    dsn = _build_dsn(settings.database_url)

    logger.info("Connecting to PostgreSQL…")
    conn: asyncpg.Connection = await asyncpg.connect(dsn)

    try:
        # Step 1: ensure schema exists (for standalone runs outside docker)
        logger.info("Ensuring schema…")
        await _ensure_schema(conn)

        # Step 2: seed vendors
        logger.info("Seeding %d vendors…", len(SUPPLIERS))
        vendor_ids: list[int] = []
        for sup in SUPPLIERS:
            if await _vendor_exists(conn, sup["account_number"]):
                row = await conn.fetchrow(
                    "SELECT business_entity_id FROM purchasing.vendor WHERE account_number = $1",
                    sup["account_number"],
                )
                vendor_ids.append(row["business_entity_id"])
                logger.debug(
                    "Vendor %s already exists — skipping", sup["account_number"]
                )
            else:
                vid = await conn.fetchval(
                    """
                    INSERT INTO purchasing.vendor
                        (account_number, name, credit_rating, preferred_vendor, active_flag)
                    VALUES ($1, $2, $3, $4, TRUE)
                    RETURNING business_entity_id
                    """,
                    sup["account_number"],
                    sup["name"],
                    sup["credit_rating"],
                    sup["credit_rating"] >= 4,
                )
                vendor_ids.append(vid)
                logger.debug("Inserted vendor %s → id=%d", sup["account_number"], vid)

        # Step 3: seed supplier_tiers
        logger.info("Seeding supplier tiers…")
        for i, sup in enumerate(SUPPLIERS):
            vid = vendor_ids[i]
            exists = await conn.fetchrow(
                "SELECT 1 FROM supply_chain.supplier_tiers WHERE supplier_id = $1", vid
            )
            if exists:
                continue
            parent_vid = (
                vendor_ids[sup["parent_idx"]] if sup["parent_idx"] is not None else None
            )
            await conn.execute(
                """
                INSERT INTO supply_chain.supplier_tiers
                    (supplier_id, tier_level, parent_supplier_id,
                     reliability_score, lead_time_days, country_code)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                vid,
                sup["tier_level"],
                parent_vid,
                sup["reliability_score"],
                sup["lead_time_days"],
                sup["country_code"],
            )

        # Step 4: seed components + products
        component_ids: list[int] = []
        logger.info("Seeding %d components…", len(COMPONENTS))
        for comp in COMPONENTS:
            row = await conn.fetchrow(
                "SELECT product_id FROM production.product WHERE product_number = $1",
                comp["product_number"],
            )
            if row:
                component_ids.append(row["product_id"])
            else:
                pid = await conn.fetchval(
                    """
                    INSERT INTO production.product
                        (name, product_number, product_subcategory_id,
                         standard_cost, list_price, weight, unit_measure_code)
                    VALUES ($1, $2, 1, $3, $4, $5, 'EA ')
                    RETURNING product_id
                    """,
                    comp["name"],
                    comp["product_number"],
                    comp["standard_cost"],
                    comp["list_price"],
                    comp["weight"],
                )
                component_ids.append(pid)

        product_ids: list[int] = []
        logger.info("Seeding %d finished goods…", len(PRODUCTS))
        for prod in PRODUCTS:
            row = await conn.fetchrow(
                "SELECT product_id FROM production.product WHERE product_number = $1",
                prod["product_number"],
            )
            if row:
                product_ids.append(row["product_id"])
            else:
                pid = await conn.fetchval(
                    """
                    INSERT INTO production.product
                        (name, product_number, product_subcategory_id,
                         standard_cost, list_price, weight, unit_measure_code)
                    VALUES ($1, $2, 2, $3, $4, $5, 'EA ')
                    RETURNING product_id
                    """,
                    prod["name"],
                    prod["product_number"],
                    prod["standard_cost"],
                    prod["list_price"],
                    prod["weight"],
                )
                product_ids.append(pid)

        # Step 5: seed BOM
        logger.info("Seeding BOM rows…")
        for prod_idx, comp_idx, qty in BOM_ROWS:
            prod_id = product_ids[prod_idx]
            comp_id = component_ids[comp_idx]
            exists = await conn.fetchrow(
                "SELECT 1 FROM production.bill_of_materials "
                "WHERE product_assembly_id = $1 AND component_id = $2",
                prod_id,
                comp_id,
            )
            if not exists:
                await conn.execute(
                    """
                    INSERT INTO production.bill_of_materials
                        (product_assembly_id, component_id, per_assembly_qty)
                    VALUES ($1, $2, $3)
                    """,
                    prod_id,
                    comp_id,
                    qty,
                )

        # Step 6: seed work centres
        logger.info("Seeding %d work centres…", len(WORK_CENTERS))
        for wc in WORK_CENTERS:
            exists = await conn.fetchrow(
                "SELECT 1 FROM production.location WHERE name = $1", wc["name"]
            )
            if not exists:
                await conn.execute(
                    "INSERT INTO production.location (name, cost_rate, availability) "
                    "VALUES ($1, $2, $3)",
                    wc["name"],
                    wc["cost_rate"],
                    wc["availability"],
                )

        # Step 7: seed distribution centres
        dc_ids: list[int] = []
        logger.info("Seeding %d distribution centres…", len(DISTRIBUTION_CENTERS))
        for dc in DISTRIBUTION_CENTERS:
            row = await conn.fetchrow(
                "SELECT dc_id FROM supply_chain.distribution_centers WHERE name = $1",
                dc["name"],
            )
            if row:
                dc_ids.append(row["dc_id"])
            else:
                dc_id = await conn.fetchval(
                    """
                    INSERT INTO supply_chain.distribution_centers
                        (name, region, country_code, latitude, longitude)
                    VALUES ($1, $2, $3, $4, $5)
                    RETURNING dc_id
                    """,
                    dc["name"],
                    dc["region"],
                    dc["country_code"],
                    dc["latitude"],
                    dc["longitude"],
                )
                dc_ids.append(dc_id)

        # Step 8: seed logistics arcs
        logger.info("Seeding logistics arcs…")
        for sup_id, capacity, cost, lead in _TIER1_ARCS:
            exists = await conn.fetchrow(
                "SELECT 1 FROM supply_chain.logistics_arcs "
                "WHERE from_node_type='supplier' AND from_node_id=$1 "
                "AND to_node_type='factory' AND to_node_id=$2",
                sup_id,
                _FACTORY_ID,
            )
            if not exists:
                await conn.execute(
                    """
                    INSERT INTO supply_chain.logistics_arcs
                        (from_node_type, from_node_id, to_node_type, to_node_id,
                         capacity, cost_per_unit, lead_time_days)
                    VALUES ('supplier', $1, 'factory', $2, $3, $4, $5)
                    """,
                    sup_id,
                    _FACTORY_ID,
                    capacity,
                    cost,
                    lead,
                )
        for factory_id, dc_id, capacity, cost, lead in _FACTORY_DC_ARCS:
            exists = await conn.fetchrow(
                "SELECT 1 FROM supply_chain.logistics_arcs "
                "WHERE from_node_type='factory' AND from_node_id=$1 "
                "AND to_node_type='dc' AND to_node_id=$2",
                factory_id,
                dc_id,
            )
            if not exists:
                await conn.execute(
                    """
                    INSERT INTO supply_chain.logistics_arcs
                        (from_node_type, from_node_id, to_node_type, to_node_id,
                         capacity, cost_per_unit, lead_time_days)
                    VALUES ('factory', $1, 'dc', $2, $3, $4, $5)
                    """,
                    factory_id,
                    dc_id,
                    capacity,
                    cost,
                    lead,
                )

        # Step 9: seed contract metadata (text content generated by seed_contracts.py)
        contracts_dir = _REPO_ROOT / "data" / "contracts"
        contracts_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Seeding %d contract records…", len(CONTRACT_DEFS))
        for i, (sup_idx, eff_date, exp_date) in enumerate(CONTRACT_DEFS):
            eff_date = datetime.date.fromisoformat(eff_date)
            exp_date = datetime.date.fromisoformat(exp_date)
            contract_num = i + 1
            txt_path = contracts_dir / f"contract_{contract_num:02d}.txt"
            sup_vid = vendor_ids[sup_idx]
            exists = await conn.fetchrow(
                "SELECT 1 FROM supply_chain.contracts WHERE contract_pdf_path = $1",
                str(txt_path),
            )
            if not exists:
                await conn.execute(
                    """
                    INSERT INTO supply_chain.contracts
                        (supplier_id, effective_date, expiry_date, contract_pdf_path)
                    VALUES ($1, $2, $3, $4)
                    """,
                    sup_vid,
                    eff_date,
                    exp_date,
                    str(txt_path),
                )

        logger.info(
            "✅ Seed complete — vendors=%d, products=%d, arcs=%d, contracts=%d",
            len(vendor_ids),
            len(component_ids) + len(product_ids),
            len(_TIER1_ARCS) + len(_FACTORY_DC_ARCS),
            len(CONTRACT_DEFS),
        )

    finally:
        await conn.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s — %(message)s"
    )
    asyncio.run(main())
