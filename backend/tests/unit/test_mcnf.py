"""Solver correctness tests — MCNF hand-solved instances (Blueprint §5.1.1).

All tests are offline (no DB, no LLM, no network).
Each ``TestMcnfOptimality`` case has a hand-calculated expected objective value
that can be verified by inspection (shown in the docstring).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from app.solvers.mcnf import solve_mcnf

# ---------------------------------------------------------------------------
# Optimality tests — 4 hand-solved LP instances
# ---------------------------------------------------------------------------


class TestMcnfOptimality:
    """Blueprint §5.1.1 — MCNF optimality: exact match against hand-solved values."""

    def test_simple_path_optimal(self) -> None:
        """Single commodity; 2-hop path (cost=3/unit) beats direct arc (cost=5/unit).

        Network : A → B (cap=100, c=2); A → C (cap=100, c=5); B → C (cap=100, c=1)
        Demand  : A → C, 10 units
        Optimal : route all 10 via A→B→C, total_cost = (2+1)×10 = 30
        """
        nodes = ["A", "B", "C"]
        arcs = [
            {"from": "A", "to": "B", "capacity": 100.0, "cost_per_unit": 2.0},
            {"from": "A", "to": "C", "capacity": 100.0, "cost_per_unit": 5.0},
            {"from": "B", "to": "C", "capacity": 100.0, "cost_per_unit": 1.0},
        ]
        commodities = [{"source": "A", "sink": "C", "demand": 10.0}]

        result = solve_mcnf(nodes, arcs, commodities)

        assert result["status"] == "OPTIMAL"
        assert abs(result["total_cost"] - 30.0) < 1e-4

    def test_capacity_constrained_split_flow(self) -> None:
        """Demand (8) > cheapest-path capacity (5) → LP splits across two paths.

        Network : A → B (cap=5, c=1); A → C (cap=5, c=3); B → C (cap=5, c=1)
        Demand  : A → C, 8 units
        Optimal : 5 via A→B→C (cost=10) + 3 via A→C (cost=9) = 19
        """
        nodes = ["A", "B", "C"]
        arcs = [
            {"from": "A", "to": "B", "capacity": 5.0, "cost_per_unit": 1.0},
            {"from": "A", "to": "C", "capacity": 5.0, "cost_per_unit": 3.0},
            {"from": "B", "to": "C", "capacity": 5.0, "cost_per_unit": 1.0},
        ]
        commodities = [{"source": "A", "sink": "C", "demand": 8.0}]

        result = solve_mcnf(nodes, arcs, commodities)

        assert result["status"] == "OPTIMAL"
        assert abs(result["total_cost"] - 19.0) < 1e-4
        # Both paths carry flow.
        assert len(result["flows"]) >= 2

    def test_multi_commodity_independent_routes(self) -> None:
        """Two commodities route independently; capacities allow optimal paths.

        Network : S1→W (cap=20,c=1); S2→W (cap=20,c=2);
                  W→C1 (cap=15,c=3); W→C2 (cap=15,c=4)
        Demand  : K0: S1→C1 10 units, K1: S2→C2 8 units
        Optimal : K0 via S1→W→C1: (1+3)×10=40; K1 via S2→W→C2: (2+4)×8=48 → 88
        """
        nodes = ["S1", "S2", "W", "C1", "C2"]
        arcs = [
            {"from": "S1", "to": "W", "capacity": 20.0, "cost_per_unit": 1.0},
            {"from": "S2", "to": "W", "capacity": 20.0, "cost_per_unit": 2.0},
            {"from": "W", "to": "C1", "capacity": 15.0, "cost_per_unit": 3.0},
            {"from": "W", "to": "C2", "capacity": 15.0, "cost_per_unit": 4.0},
        ]
        commodities = [
            {"source": "S1", "sink": "C1", "demand": 10.0},
            {"source": "S2", "sink": "C2", "demand": 8.0},
        ]

        result = solve_mcnf(nodes, arcs, commodities)

        assert result["status"] == "OPTIMAL"
        assert abs(result["total_cost"] - 88.0) < 1e-4

    def test_infeasible_demand_exceeds_capacity(self) -> None:
        """Single arc with capacity 5 cannot route demand of 10 → INFEASIBLE.

        Network  : A → B (cap=5, c=1)
        Demand   : A → B, 10 units
        Expected : status=INFEASIBLE, total_cost=0, empty flows list.
        """
        nodes = ["A", "B"]
        arcs = [{"from": "A", "to": "B", "capacity": 5.0, "cost_per_unit": 1.0}]
        commodities = [{"source": "A", "sink": "B", "demand": 10.0}]

        result = solve_mcnf(nodes, arcs, commodities)

        assert result["status"] == "INFEASIBLE"
        assert result["total_cost"] == 0.0
        assert result["flows"] == []


# ---------------------------------------------------------------------------
# Output-format tests
# ---------------------------------------------------------------------------


class TestMcnfOutputFormat:
    """Verify shape and content of the returned dict."""

    # Shared fixture data reused across tests.
    _NODES = ["A", "B", "C"]
    _ARCS = [
        {"from": "A", "to": "B", "capacity": 100.0, "cost_per_unit": 2.0},
        {"from": "A", "to": "C", "capacity": 100.0, "cost_per_unit": 5.0},
        {"from": "B", "to": "C", "capacity": 100.0, "cost_per_unit": 1.0},
    ]
    _COMMODITIES = [{"source": "A", "sink": "C", "demand": 10.0}]

    def test_flows_only_contain_positive_arcs(self) -> None:
        """Flows list must not include arcs with negligible (≤1e-6) flow."""
        result = solve_mcnf(self._NODES, self._ARCS, self._COMMODITIES)
        for flow in result["flows"]:
            assert flow["flow"] > 1e-6

    def test_shadow_prices_one_entry_per_arc(self) -> None:
        """shadow_prices must have exactly one entry per arc."""
        result = solve_mcnf(self._NODES, self._ARCS, self._COMMODITIES)
        assert len(result["shadow_prices"]) == len(self._ARCS)

    def test_shadow_price_keys(self) -> None:
        """Each shadow_price dict must contain 'from', 'to', 'dual'."""
        result = solve_mcnf(self._NODES, self._ARCS, self._COMMODITIES)
        for sp in result["shadow_prices"]:
            assert "from" in sp
            assert "to" in sp
            assert "dual" in sp

    def test_solver_unavailable_returns_clean_dict(self, monkeypatch) -> None:
        """If GLOP solver cannot be created, status=SOLVER_UNAVAILABLE with defaults."""
        import app.solvers.mcnf as mcnf_mod

        mock_pywraplp = MagicMock()
        mock_pywraplp.Solver.CreateSolver.return_value = None
        monkeypatch.setattr(mcnf_mod, "pywraplp", mock_pywraplp)

        result = mcnf_mod.solve_mcnf(["A"], [], [])

        assert result["status"] == "SOLVER_UNAVAILABLE"
        assert result["total_cost"] == 0.0
        assert result["flows"] == []
        assert result["shadow_prices"] == []
