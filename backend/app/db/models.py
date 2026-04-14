"""SQLAlchemy ORM models for AdventureWorks + supply-chain extension (Blueprint §2.1.1).

Stage 2 implementation.
"""
from __future__ import annotations

from sqlalchemy import Column, Integer, Numeric, String, Date, Text, CheckConstraint, ForeignKey
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class SupplierTier(Base):
    __tablename__ = "supplier_tiers"
    __table_args__ = {"schema": "supply_chain"}

    id = Column(Integer, primary_key=True, autoincrement=True)
    supplier_id = Column(Integer, ForeignKey("purchasing.vendor.business_entity_id"), nullable=False)
    tier_level = Column(Integer, nullable=False)
    parent_supplier_id = Column(Integer, ForeignKey("purchasing.vendor.business_entity_id"))
    reliability_score = Column(Numeric(3, 2))
    lead_time_days = Column(Integer, nullable=False)
    country_code = Column(String(2), nullable=False)


class Contract(Base):
    __tablename__ = "contracts"
    __table_args__ = {"schema": "supply_chain"}

    contract_id = Column(Integer, primary_key=True, autoincrement=True)
    supplier_id = Column(Integer, ForeignKey("purchasing.vendor.business_entity_id"), nullable=False)
    effective_date = Column(Date, nullable=False)
    expiry_date = Column(Date, nullable=False)
    contract_pdf_path = Column(Text, nullable=False)
    embedding_id = Column(String(36))  # UUID as string


class LogisticsArc(Base):
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
