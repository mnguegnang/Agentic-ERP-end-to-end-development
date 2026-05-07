"""initial_schema

Revision ID: a7d94eba88dc
Revises:
Create Date: 2026-05-07 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "a7d94eba88dc"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable pgvector extension (pre-installed in pgvector/pgvector:pg16 CI image)
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Schemas are created by env.py before migrations run

    # --- purchasing schema ---
    op.create_table(
        "vendor",
        sa.Column("business_entity_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("account_number", sa.String(length=15), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("credit_rating", sa.Integer(), nullable=False),
        sa.Column("preferred_vendor", sa.Boolean(), nullable=False),
        sa.Column("active_flag", sa.Boolean(), nullable=False),
        sa.PrimaryKeyConstraint("business_entity_id"),
        sa.UniqueConstraint("account_number"),
        schema="purchasing",
    )

    # --- production schema ---
    op.create_table(
        "product",
        sa.Column("product_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("product_number", sa.String(length=25), nullable=False),
        sa.Column("product_subcategory_id", sa.Integer(), nullable=False),
        sa.Column("standard_cost", sa.Numeric(precision=19, scale=4), nullable=False),
        sa.Column("list_price", sa.Numeric(precision=19, scale=4), nullable=False),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column("unit_measure_code", sa.String(length=3), nullable=True),
        sa.PrimaryKeyConstraint("product_id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("product_number"),
        schema="production",
    )
    op.create_table(
        "location",
        sa.Column("location_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("cost_rate", sa.Numeric(precision=7, scale=4), nullable=False),
        sa.Column("availability", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.PrimaryKeyConstraint("location_id"),
        sa.UniqueConstraint("name"),
        schema="production",
    )
    op.create_table(
        "bill_of_materials",
        sa.Column("bill_of_materials_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("product_assembly_id", sa.Integer(), nullable=True),
        sa.Column("component_id", sa.Integer(), nullable=False),
        sa.Column("unit_measure_code", sa.String(length=3), nullable=False),
        sa.Column("bom_level", sa.Integer(), nullable=False),
        sa.Column("per_assembly_qty", sa.Numeric(precision=8, scale=2), nullable=False),
        sa.ForeignKeyConstraint(
            ["component_id"],
            ["production.product.product_id"],
        ),
        sa.ForeignKeyConstraint(
            ["product_assembly_id"],
            ["production.product.product_id"],
        ),
        sa.PrimaryKeyConstraint("bill_of_materials_id"),
        schema="production",
    )

    # --- supply_chain schema ---
    op.create_table(
        "distribution_centers",
        sa.Column("dc_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("region", sa.String(length=50), nullable=False),
        sa.Column("country_code", sa.String(length=2), nullable=False),
        sa.Column("latitude", sa.Numeric(precision=9, scale=6), nullable=True),
        sa.Column("longitude", sa.Numeric(precision=9, scale=6), nullable=True),
        sa.PrimaryKeyConstraint("dc_id"),
        schema="supply_chain",
    )
    op.create_table(
        "supplier_tiers",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("supplier_id", sa.Integer(), nullable=False),
        sa.Column("tier_level", sa.Integer(), nullable=False),
        sa.Column("parent_supplier_id", sa.Integer(), nullable=True),
        sa.Column("reliability_score", sa.Numeric(precision=3, scale=2), nullable=True),
        sa.Column("lead_time_days", sa.Integer(), nullable=False),
        sa.Column("country_code", sa.String(length=2), nullable=False),
        sa.ForeignKeyConstraint(
            ["supplier_id"],
            ["purchasing.vendor.business_entity_id"],
        ),
        sa.ForeignKeyConstraint(
            ["parent_supplier_id"],
            ["purchasing.vendor.business_entity_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="supply_chain",
    )
    op.create_table(
        "contracts",
        sa.Column("contract_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("supplier_id", sa.Integer(), nullable=False),
        sa.Column("effective_date", sa.Date(), nullable=False),
        sa.Column("expiry_date", sa.Date(), nullable=False),
        sa.Column("contract_pdf_path", sa.Text(), nullable=False),
        sa.Column("embedding_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["supplier_id"],
            ["purchasing.vendor.business_entity_id"],
        ),
        sa.PrimaryKeyConstraint("contract_id"),
        schema="supply_chain",
    )
    op.create_table(
        "logistics_arcs",
        sa.Column("arc_id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("from_node_type", sa.String(length=20), nullable=False),
        sa.Column("from_node_id", sa.Integer(), nullable=False),
        sa.Column("to_node_type", sa.String(length=20), nullable=False),
        sa.Column("to_node_id", sa.Integer(), nullable=False),
        sa.Column("capacity", sa.Integer(), nullable=False),
        sa.Column("cost_per_unit", sa.Numeric(precision=10, scale=2), nullable=False),
        sa.Column("lead_time_days", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("arc_id"),
        schema="supply_chain",
    )
    op.create_table(
        "contract_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("contract_id", sa.Integer(), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.ForeignKeyConstraint(
            ["contract_id"],
            ["supply_chain.contracts.contract_id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        schema="supply_chain",
    )


def downgrade() -> None:
    op.drop_table("contract_embeddings", schema="supply_chain")
    op.drop_table("logistics_arcs", schema="supply_chain")
    op.drop_table("contracts", schema="supply_chain")
    op.drop_table("supplier_tiers", schema="supply_chain")
    op.drop_table("distribution_centers", schema="supply_chain")
    op.drop_table("bill_of_materials", schema="production")
    op.drop_table("location", schema="production")
    op.drop_table("product", schema="production")
    op.drop_table("vendor", schema="purchasing")
