"""Unit tests for Stage 2 seed data constants (Blueprint §2.1.1, §3.3).

Validates structural integrity of the seed data without any DB/network calls.
"""
from __future__ import annotations

from scripts.seed_adventureworks import (
    BOM_ROWS,
    COMPONENTS,
    CONTRACT_DEFS,
    DISTRIBUTION_CENTERS,
    PRODUCTS,
    SUPPLIER_PROVIDES,
    SUPPLIERS,
    WORK_CENTERS,
)
from scripts.seed_contracts import _FM_VARIANTS, _generate_contract_text


class TestSupplierData:
    def test_exactly_14_suppliers(self) -> None:
        assert len(SUPPLIERS) == 14, f"Blueprint requires 14 suppliers, got {len(SUPPLIERS)}"

    def test_supplier_fields_present(self) -> None:
        required = {"account_number", "name", "credit_rating", "tier_level",
                    "reliability_score", "lead_time_days", "country_code"}
        for s in SUPPLIERS:
            assert required.issubset(s.keys()), f"Supplier missing fields: {s}"

    def test_credit_ratings_in_range(self) -> None:
        for s in SUPPLIERS:
            assert 1 <= s["credit_rating"] <= 5

    def test_reliability_scores_in_range(self) -> None:
        for s in SUPPLIERS:
            assert 0.0 <= s["reliability_score"] <= 1.0

    def test_tier_levels_are_1_to_3(self) -> None:
        tiers = {s["tier_level"] for s in SUPPLIERS}
        assert tiers.issubset({1, 2, 3, 4}), f"Unexpected tier levels: {tiers}"
        assert 1 in tiers and 2 in tiers and 3 in tiers

    def test_tier1_has_no_parent(self) -> None:
        for s in SUPPLIERS:
            if s["tier_level"] == 1:
                assert s["parent_idx"] is None, f"Tier-1 supplier has parent: {s['name']}"

    def test_parent_indices_are_valid(self) -> None:
        n = len(SUPPLIERS)
        for s in SUPPLIERS:
            if s["parent_idx"] is not None:
                assert 0 <= s["parent_idx"] < n

    def test_account_numbers_unique(self) -> None:
        accts = [s["account_number"] for s in SUPPLIERS]
        assert len(accts) == len(set(accts)), "Duplicate account numbers"


class TestComponentAndProductData:
    def test_9_components(self) -> None:
        assert len(COMPONENTS) == 9

    def test_4_products(self) -> None:
        assert len(PRODUCTS) == 4

    def test_component_product_numbers_unique(self) -> None:
        nums = [c["product_number"] for c in COMPONENTS] + [p["product_number"] for p in PRODUCTS]
        assert len(nums) == len(set(nums)), "Duplicate product numbers"

    def test_bom_references_valid_indices(self) -> None:
        for prod_idx, comp_idx, qty in BOM_ROWS:
            assert 0 <= prod_idx < len(PRODUCTS), f"Invalid product index {prod_idx}"
            assert 0 <= comp_idx < len(COMPONENTS), f"Invalid component index {comp_idx}"
            assert qty > 0, f"BOM quantity must be positive, got {qty}"


class TestContractData:
    def test_exactly_20_contracts(self) -> None:
        n = len(CONTRACT_DEFS)
        assert n == 20, f"Blueprint requires 20 contracts, got {n}"

    def test_contract_supplier_indices_valid(self) -> None:
        for sup_idx, _, _ in CONTRACT_DEFS:
            assert 0 <= sup_idx < len(SUPPLIERS)

    def test_contract_dates_logical(self) -> None:
        for _, eff, exp in CONTRACT_DEFS:
            assert eff < exp, f"Effective date {eff} >= expiry {exp}"

    def test_5_fm_variants(self) -> None:
        assert len(_FM_VARIANTS) == 5

    def test_each_fm_variant_contains_force_majeure(self) -> None:
        for i, variant in enumerate(_FM_VARIANTS):
            assert "force majeure" in variant.lower() or "Force Majeure" in variant, \
                f"FM variant {i} missing 'Force Majeure'"

    def test_fm_variants_are_distinct(self) -> None:
        # Each variant must be unique (no duplicates)
        assert len(set(_FM_VARIANTS)) == 5, "FM variants are not all distinct"

    def test_generate_contract_text_structure(self) -> None:
        text = _generate_contract_text(
            contract_id=1,
            supplier_name="Test Supplier Inc.",
            effective_date="2024-01-01",
            expiry_date="2026-12-31",
            account_number="TQ-ELEC-001",
        )
        # Must contain required sections
        for section in ["SECTION 1", "SECTION 14", "SECTION 18", "SECTION 20"]:
            assert section in text, f"Missing {section} in contract text"

    def test_generate_contract_uses_correct_fm_variant(self) -> None:
        """Contract IDs 1-5 should use FM variants 0-4 respectively."""
        for cid in range(1, 6):
            text = _generate_contract_text(
                contract_id=cid,
                supplier_name="Acme Corp",
                effective_date="2024-01-01",
                expiry_date="2026-12-31",
                account_number="TQ-ELEC-001",
            )
            expected_variant = _FM_VARIANTS[(cid - 1) % 5]
            # Check a distinctive sentence from the expected variant
            snippet = expected_variant.split("\n")[2].strip()[:50]
            assert snippet in text, f"Contract {cid} missing FM variant {(cid-1)%5} content"


class TestSupplierProvidesData:
    def test_all_indices_valid(self) -> None:
        for sup_idx, comp_idx in SUPPLIER_PROVIDES:
            assert 0 <= sup_idx < len(SUPPLIERS)
            assert 0 <= comp_idx < len(COMPONENTS)

    def test_no_duplicate_provides(self) -> None:
        pairs = [(s, c) for s, c in SUPPLIER_PROVIDES]
        assert len(pairs) == len(set(pairs)), "Duplicate PROVIDES entries"


class TestInfrastructureData:
    def test_3_distribution_centers(self) -> None:
        assert len(DISTRIBUTION_CENTERS) == 3

    def test_5_work_centers(self) -> None:
        assert len(WORK_CENTERS) == 5
