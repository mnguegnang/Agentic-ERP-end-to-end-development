"""SQLAlchemy ORM models for AdventureWorks + supply-chain extension (Blueprint §2.1.1).

Stage 2 implementation.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    Float,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# production schema
# ---------------------------------------------------------------------------


class Vendor(Base):
    """purchasing.vendor — AdventureWorks supplier/vendor table."""

    __tablename__ = "vendor"
    __table_args__ = {"schema": "purchasing"}

    business_entity_id = Column(Integer, primary_key=True, autoincrement=True)
    account_number = Column(String(15), nullable=False, unique=True)
    name = Column(String(100), nullable=False)
    credit_rating = Column(Integer, nullable=False, default=3)
    preferred_vendor = Column(Boolean, nullable=False, default=True)
    active_flag = Column(Boolean, nullable=False, default=True)


class Product(Base):
    """production.product — components and finished goods."""

    __tablename__ = "product"
    __table_args__ = {"schema": "production"}

    product_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    product_number = Column(String(25), nullable=False, unique=True)
    product_subcategory_id = Column(Integer, nullable=False, default=1)
    standard_cost = Column(Numeric(19, 4), nullable=False, default=0)
    list_price = Column(Numeric(19, 4), nullable=False, default=0)
    weight = Column(Float)
    unit_measure_code = Column(String(3))


class BillOfMaterials(Base):
    """production.bill_of_materials — component-to-product linkage."""

    __tablename__ = "bill_of_materials"
    __table_args__ = {"schema": "production"}

    bill_of_materials_id = Column(Integer, primary_key=True, autoincrement=True)
    product_assembly_id = Column(Integer, ForeignKey("production.product.product_id"))
    component_id = Column(
        Integer, ForeignKey("production.product.product_id"), nullable=False
    )
    unit_measure_code = Column(String(3), nullable=False, default="EA ")
    bom_level = Column(Integer, nullable=False, default=1)
    per_assembly_qty = Column(Numeric(8, 2), nullable=False, default=1.0)


class ProductionLocation(Base):
    """production.location — work centres."""

    __tablename__ = "location"
    __table_args__ = {"schema": "production"}

    location_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    cost_rate = Column(Numeric(7, 4), nullable=False, default=0)
    availability = Column(Numeric(8, 2), nullable=False, default=160.0)


# ---------------------------------------------------------------------------
# supply_chain extension schema
# ---------------------------------------------------------------------------


class DistributionCenter(Base):
    """supply_chain.distribution_centers."""

    __tablename__ = "distribution_centers"
    __table_args__ = {"schema": "supply_chain"}

    dc_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    region = Column(String(50), nullable=False)
    country_code = Column(String(2), nullable=False)
    latitude = Column(Numeric(9, 6))
    longitude = Column(Numeric(9, 6))


class SupplierTier(Base):
    """supply_chain.supplier_tiers — multi-tier supplier network."""

    __tablename__ = "supplier_tiers"
    __table_args__ = {"schema": "supply_chain"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    supplier_id = Column(
        Integer, ForeignKey("purchasing.vendor.business_entity_id"), nullable=False
    )
    tier_level = Column(Integer, nullable=False)
    parent_supplier_id = Column(
        Integer, ForeignKey("purchasing.vendor.business_entity_id")
    )
    reliability_score = Column(Numeric(3, 2))
    lead_time_days = Column(Integer, nullable=False)
    country_code = Column(String(2), nullable=False)


class Contract(Base):
    """supply_chain.contracts — synthetic supplier contracts."""

    __tablename__ = "contracts"
    __table_args__ = {"schema": "supply_chain"}

    contract_id = Column(Integer, primary_key=True, autoincrement=True)
    supplier_id = Column(
        Integer, ForeignKey("purchasing.vendor.business_entity_id"), nullable=False
    )
    effective_date = Column(Date, nullable=False)
    expiry_date = Column(Date, nullable=False)
    contract_pdf_path = Column(Text, nullable=False)
    embedding_id = Column(UUID(as_uuid=True))  # FK to first chunk


class LogisticsArc(Base):
    """supply_chain.logistics_arcs — network arcs for MCNF solver."""

    __tablename__ = "logistics_arcs"
    __table_args__ = {"schema": "supply_chain"}

    arc_id = Column(Integer, primary_key=True, autoincrement=True)
    from_node_type = Column(String(20), nullable=False)
    from_node_id = Column(Integer, nullable=False)
    to_node_type = Column(String(20), nullable=False)
    to_node_id = Column(Integer, nullable=False)
    capacity = Column(Integer, nullable=False)
    cost_per_unit = Column(Numeric(10, 2), nullable=False)
    lead_time_days = Column(Integer, nullable=False)


class ContractEmbedding(Base):
    """supply_chain.contract_embeddings — BGE-large-en-v1.5 chunk vectors."""

    __tablename__ = "contract_embeddings"
    __table_args__ = {"schema": "supply_chain"}

    id = Column(UUID(as_uuid=True), primary_key=True)
    contract_id = Column(Integer, ForeignKey("supply_chain.contracts.contract_id"))
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Vector(1024))
