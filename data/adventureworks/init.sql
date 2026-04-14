-- =============================================================================
-- AdventureWorks OLTP (focused subset) + Supply Chain Extension Schema
-- Blueprint §2.1.1 | Stage 2
-- Mounted into Docker PostgreSQL container init dir:
--   docker/docker-compose.yml → /docker-entrypoint-initdb.d/01-init.sql
-- Runs automatically on first container start (volume empty).
-- All statements use IF NOT EXISTS — idempotent for seed script re-runs.
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions (requires superuser — aw_user is POSTGRES_USER → superuser)
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;       -- pgvector 1024-dim embeddings
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- gen_random_uuid()

-- ---------------------------------------------------------------------------
-- Schemas
-- ---------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS purchasing;
CREATE SCHEMA IF NOT EXISTS production;
CREATE SCHEMA IF NOT EXISTS supply_chain;

-- =============================================================================
-- purchasing schema  (AdventureWorks Purchasing)
-- =============================================================================
CREATE TABLE IF NOT EXISTS purchasing.vendor (
    business_entity_id  SERIAL       PRIMARY KEY,
    account_number      VARCHAR(15)  NOT NULL UNIQUE,
    name                VARCHAR(100) NOT NULL,
    credit_rating       SMALLINT     NOT NULL DEFAULT 3
                            CHECK (credit_rating BETWEEN 1 AND 5),
    preferred_vendor    BOOLEAN      NOT NULL DEFAULT TRUE,
    active_flag         BOOLEAN      NOT NULL DEFAULT TRUE,
    modified_date       TIMESTAMP    NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- production schema  (AdventureWorks Production)
-- =============================================================================
CREATE TABLE IF NOT EXISTS production.product (
    product_id              SERIAL        PRIMARY KEY,
    name                    VARCHAR(100)  NOT NULL UNIQUE,
    product_number          VARCHAR(25)   NOT NULL UNIQUE,
    -- 1 = component, 2 = finished_good
    product_subcategory_id  INT           NOT NULL DEFAULT 1,
    standard_cost           NUMERIC(19,4) NOT NULL DEFAULT 0,
    list_price              NUMERIC(19,4) NOT NULL DEFAULT 0,
    weight                  NUMERIC(8,2),
    unit_measure_code       CHAR(3),
    modified_date           TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS production.bill_of_materials (
    bill_of_materials_id  SERIAL       PRIMARY KEY,
    product_assembly_id   INT          REFERENCES production.product(product_id),
    component_id          INT          NOT NULL
                              REFERENCES production.product(product_id),
    start_date            DATE         NOT NULL DEFAULT CURRENT_DATE,
    end_date              DATE,
    unit_measure_code     CHAR(3)      NOT NULL DEFAULT 'EA ',
    bom_level             SMALLINT     NOT NULL DEFAULT 1,
    per_assembly_qty      NUMERIC(8,2) NOT NULL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS production.location (
    location_id    SERIAL        PRIMARY KEY,
    name           VARCHAR(100)  NOT NULL UNIQUE,
    cost_rate      NUMERIC(7,4)  NOT NULL DEFAULT 0,
    availability   NUMERIC(8,2)  NOT NULL DEFAULT 160.0,  -- hours / month
    modified_date  TIMESTAMP     NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =============================================================================
-- supply_chain extension schema  (Blueprint §2.1.1 + KG schema §2.1.2)
-- =============================================================================

-- Distribution centres (not in standard AdventureWorks OLTP)
CREATE TABLE IF NOT EXISTS supply_chain.distribution_centers (
    dc_id        SERIAL        PRIMARY KEY,
    name         VARCHAR(100)  NOT NULL,
    region       VARCHAR(50)   NOT NULL,
    country_code CHAR(2)       NOT NULL,
    latitude     NUMERIC(9,6),
    longitude    NUMERIC(9,6)
);

-- Multi-tier supplier network
CREATE TABLE IF NOT EXISTS supply_chain.supplier_tiers (
    id                  SERIAL        PRIMARY KEY,
    supplier_id         INT           NOT NULL
                            REFERENCES purchasing.vendor(business_entity_id),
    tier_level          INT           NOT NULL CHECK (tier_level BETWEEN 1 AND 4),
    parent_supplier_id  INT           REFERENCES purchasing.vendor(business_entity_id),
    reliability_score   NUMERIC(3,2)  CHECK (reliability_score BETWEEN 0 AND 1),
    lead_time_days      INT           NOT NULL,
    country_code        CHAR(2)       NOT NULL
);

-- Synthetic supplier contracts
CREATE TABLE IF NOT EXISTS supply_chain.contracts (
    contract_id       SERIAL  PRIMARY KEY,
    supplier_id       INT     NOT NULL
                          REFERENCES purchasing.vendor(business_entity_id),
    effective_date    DATE    NOT NULL,
    expiry_date       DATE    NOT NULL,
    contract_pdf_path TEXT    NOT NULL,
    embedding_id      UUID                            -- FK to first chunk UUID
);

-- Logistics arcs for MCNF solver  (Blueprint §2.1.1)
CREATE TABLE IF NOT EXISTS supply_chain.logistics_arcs (
    arc_id          SERIAL        PRIMARY KEY,
    from_node_type  VARCHAR(20)   NOT NULL
                        CHECK (from_node_type IN ('supplier', 'factory', 'dc')),
    from_node_id    INT           NOT NULL,
    to_node_type    VARCHAR(20)   NOT NULL
                        CHECK (to_node_type IN ('supplier', 'factory', 'dc')),
    to_node_id      INT           NOT NULL,
    capacity        INT           NOT NULL CHECK (capacity >= 0),
    cost_per_unit   NUMERIC(10,2) NOT NULL,
    lead_time_days  INT           NOT NULL
);

-- Contract text chunks + BGE-large-en-v1.5 embeddings (1024-dim)
CREATE TABLE IF NOT EXISTS supply_chain.contract_embeddings (
    id           UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id  INT          REFERENCES supply_chain.contracts(contract_id),
    chunk_index  INT          NOT NULL,
    chunk_text   TEXT         NOT NULL,
    embedding    vector(1024)
);

-- IVFFlat cosine index (lists=100 is appropriate for ~100 K vectors)
CREATE INDEX IF NOT EXISTS idx_contract_emb_ivfflat
    ON supply_chain.contract_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
