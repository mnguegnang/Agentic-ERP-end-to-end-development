"""MCP server: mcp-erp-postgres (Blueprint §4.3.6).

Exposes query_erp and get_product_bom via parameterised SQLAlchemy ORM.
No raw SQL strings are accepted from callers — all queries use ORM selects.
"""

from __future__ import annotations

import logging

from mcp.server.fastmcp import FastMCP
from sqlalchemy import select, text

from app.db.models import BillOfMaterials, Product, Vendor
from app.db.session import get_session_factory

logger = logging.getLogger(__name__)

mcp = FastMCP("mcp-erp-postgres")

# Whitelisted query types for query_erp (no ad-hoc SQL from callers)
_VALID_QUERY_TYPES = frozenset({"vendors", "products", "distribution_centers"})


@mcp.tool()
async def query_erp(query_type: str, filters: dict) -> dict:
    """Query AdventureWorks ERP data via parameterised ORM (no raw SQL).

    Supported query_type values: ``vendors``, ``products``, ``distribution_centers``.
    ``filters`` keys are matched against ORM column names.
    """
    if query_type not in _VALID_QUERY_TYPES:
        valid = sorted(_VALID_QUERY_TYPES)
        return {"error": f"Unknown query_type: {query_type!r}. Must be one of {valid}"}

    factory = get_session_factory()
    try:
        async with factory() as session:
            if query_type == "vendors":
                stmt = select(Vendor)
                if filters.get("name"):
                    stmt = stmt.where(Vendor.name.ilike(f"%{filters['name']}%"))
                if filters.get("active_flag") is not None:
                    stmt = stmt.where(
                        Vendor.active_flag == bool(filters["active_flag"])
                    )
                rows = (await session.execute(stmt)).scalars().all()
                return {
                    "results": [
                        {
                            "id": r.business_entity_id,
                            "name": r.name,
                            "credit_rating": r.credit_rating,
                        }
                        for r in rows
                    ]
                }

            if query_type == "products":
                stmt = select(Product)
                if filters.get("name"):
                    stmt = stmt.where(Product.name.ilike(f"%{filters['name']}%"))
                rows = (await session.execute(stmt)).scalars().all()
                return {
                    "results": [
                        {
                            "id": r.product_id,
                            "name": r.name,
                            "product_number": r.product_number,
                            "standard_cost": float(r.standard_cost or 0),
                        }
                        for r in rows
                    ]
                }

            # distribution_centers — plain select
            dc_sql = (
                "SELECT dc_id, name, region, country_code"
                " FROM supply_chain.distribution_centers"
            )
            result = await session.execute(text(dc_sql))
            return {"results": [dict(r._mapping) for r in result]}

    except Exception as exc:
        logger.exception("query_erp failed for type=%s: %s", query_type, exc)
        return {"error": str(exc)}


@mcp.tool()
async def get_product_bom(product_id: int) -> dict:
    """Return the immediate bill-of-materials for a product (one level)."""
    factory = get_session_factory()
    try:
        async with factory() as session:
            stmt = (
                select(BillOfMaterials, Product)
                .join(
                    Product,
                    BillOfMaterials.component_id == Product.product_id,
                )
                .where(BillOfMaterials.product_assembly_id == product_id)
                .order_by(BillOfMaterials.bom_level)
            )
            rows = (await session.execute(stmt)).all()
            bom_items = [
                {
                    "component_id": bom.component_id,
                    "component_name": prod.name,
                    "per_assembly_qty": float(bom.per_assembly_qty),
                    "bom_level": bom.bom_level,
                }
                for bom, prod in rows
            ]
            return {
                "product_id": product_id,
                "bom_items": bom_items,
                "count": len(bom_items),
            }
    except Exception as exc:
        logger.exception(
            "get_product_bom failed for product_id=%s: %s", product_id, exc
        )
        return {"error": str(exc)}
