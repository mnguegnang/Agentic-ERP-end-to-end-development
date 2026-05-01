"""Integration tests — Stage 3 baseline chain and WebSocket endpoint (Blueprint §3.4).

The LLM is mocked throughout; tests verify:
  1. WebSocket endpoint schema (role, content, tool_used, solver_result)
  2. Invalid JSON handling (graceful error response)
  3. Tool-call path: LLM tool-call → solve_mcnf executes → REAL solver result returned
  4. run_baseline_chain() directly: no-tool path and tool-call path
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app.main import app
from langchain_core.messages import AIMessage
from starlette.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_settings(mock_settings_fn: MagicMock) -> None:
    """Configure a mock Settings object with safe test values."""
    s = MagicMock()
    s.llm_model = "gpt-4o"
    s.llm_base_url = "https://models.inference.ai.azure.com"
    s.github_token = "test-token-placeholder"
    s.llm_temperature = 0.0
    s.llm_max_tokens = 4096
    mock_settings_fn.return_value = s


def _ai_tool_call(args: dict, tc_id: str = "tc_001") -> AIMessage:
    """Synthesise an AIMessage that contains exactly one solve_mcnf tool call."""
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": "solve_mcnf",
                "args": args,
                "id": tc_id,
                "type": "tool_call",
            }
        ],
    )


def _ai_text(content: str) -> AIMessage:
    """Synthesise a plain-text AIMessage (no tool calls)."""
    return AIMessage(content=content)


# ---------------------------------------------------------------------------
# WebSocket endpoint tests (LLM mocked via monkeypatch on the chain function)
# ---------------------------------------------------------------------------


class TestWebSocketEndpoint:
    def test_health_endpoint_returns_ok(self) -> None:
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_ws_accepts_connection(self) -> None:
        """WebSocket at /ws/chat should accept without error."""

        async def _mock_chain(query: str):
            from app.api.schemas import WsResponse

            return WsResponse(content="pong", tool_used=None, solver_result=None)

        with patch("app.api.routes_chat.run_orchestrator", _mock_chain):
            client = TestClient(app)
            with client.websocket_connect("/ws/chat") as ws:
                ws.send_json({"role": "user", "content": "ping"})
                resp = ws.receive_json()

        assert resp["role"] == "assistant"
        assert resp["content"] == "pong"
        assert resp["tool_used"] is None

    def test_ws_invalid_json_returns_error_message(self) -> None:
        """Malformed JSON should produce a graceful 'Invalid message format' reply."""
        client = TestClient(app)
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_text("NOT_VALID_JSON{{")
            resp = ws.receive_json()

        assert resp["role"] == "assistant"
        assert "Invalid message format" in resp["content"]

    def test_ws_returns_solver_result_when_tool_used(self, monkeypatch) -> None:
        """When chain calls solve_mcnf, solver_result and tool_used are forwarded."""
        solver_payload = {
            "status": "OPTIMAL",
            "total_cost": 50.0,
            "flows": [{"commodity": 0, "from": "A", "to": "B", "flow": 50.0}],
            "shadow_prices": [],
        }

        async def _mock_chain(query: str):
            from app.api.schemas import WsResponse

            return WsResponse(
                content="Optimal routing: $50 total.",
                tool_used="solve_mcnf",
                solver_result=solver_payload,
            )

        monkeypatch.setattr("app.api.routes_chat.run_orchestrator", _mock_chain)
        client = TestClient(app)
        with client.websocket_connect("/ws/chat") as ws:
            ws.send_json({"role": "user", "content": "Route 50 units A to B"})
            resp = ws.receive_json()

        assert resp["tool_used"] == "solve_mcnf"
        assert resp["solver_result"]["status"] == "OPTIMAL"
        assert abs(resp["solver_result"]["total_cost"] - 50.0) < 1e-4


# ---------------------------------------------------------------------------
# run_baseline_chain() direct tests (LLM mocked at ChatOpenAI level)
# ---------------------------------------------------------------------------


class TestBaselineChainDirect:
    """Test run_baseline_chain() with a mocked LangChain LLM.

    The MCNF solver is NOT mocked — when the tool path is exercised the real
    OR-Tools GLOP solver runs, verifying end-to-end numerical correctness.
    """

    @pytest.mark.asyncio
    async def test_no_tool_call_returns_text_response(self) -> None:
        """LLM answers without calling a tool → WsResponse with content set."""
        with (
            patch("app.chains.baseline_chain.get_settings") as mock_settings_fn,
            patch("app.chains.baseline_chain.ChatOpenAI") as mock_llm_cls,
        ):
            _patch_settings(mock_settings_fn)
            mock_instance = MagicMock()
            mock_llm_with_tools = MagicMock()
            mock_instance.bind_tools.return_value = mock_llm_with_tools
            mock_llm_with_tools.ainvoke = AsyncMock(
                return_value=_ai_text("Hello! I am your supply chain copilot.")
            )
            mock_llm_cls.return_value = mock_instance

            # Re-import to pick up the patched constructor.
            from app.chains.baseline_chain import run_baseline_chain

            result = await run_baseline_chain("Hello")

        assert result.role == "assistant"
        assert len(result.content) > 0
        assert result.tool_used is None
        assert result.solver_result is None

    @pytest.mark.asyncio
    async def test_tool_call_executes_real_solver(self) -> None:
        """LLM issues a tool call → real GLOP solver runs → optimal result returned.

        Network : A→B (cap=100, c=2); A→C (cap=100, c=5); B→C (cap=100, c=1)
        Demand  : A→C, 10 units  →  expected total_cost = 30.0
        """
        mcnf_args = {
            "nodes": ["A", "B", "C"],
            "arcs": [
                {"from": "A", "to": "B", "capacity": 100.0, "cost_per_unit": 2.0},
                {"from": "A", "to": "C", "capacity": 100.0, "cost_per_unit": 5.0},
                {"from": "B", "to": "C", "capacity": 100.0, "cost_per_unit": 1.0},
            ],
            "commodities": [{"source": "A", "sink": "C", "demand": 10.0}],
        }

        with (
            patch("app.chains.baseline_chain.get_settings") as mock_settings_fn,
            patch("app.chains.baseline_chain.ChatOpenAI") as mock_llm_cls,
        ):
            _patch_settings(mock_settings_fn)
            mock_instance = MagicMock()
            mock_llm_with_tools = MagicMock()
            mock_instance.bind_tools.return_value = mock_llm_with_tools
            # First call: tool call; second call: synthesis.
            mock_llm_with_tools.ainvoke = AsyncMock(
                side_effect=[
                    _ai_tool_call(mcnf_args),
                    _ai_text("Optimal routing costs $30. Use A→B→C."),
                ]
            )
            mock_llm_cls.return_value = mock_instance

            from app.chains.baseline_chain import run_baseline_chain

            result = await run_baseline_chain(
                "Re-route 10 units of ball bearings from A to C at minimum cost."
            )

        assert result.tool_used == "solve_mcnf"
        assert result.solver_result is not None
        assert result.solver_result["status"] == "OPTIMAL"
        assert (
            abs(result.solver_result["total_cost"] - 30.0) < 1e-3
        ), f"Expected total_cost≈30.0, got {result.solver_result['total_cost']}"

    @pytest.mark.asyncio
    async def test_unknown_tool_name_does_not_crash(self) -> None:
        """If the LLM requests a non-existent tool, the chain handles it gracefully."""
        with (
            patch("app.chains.baseline_chain.get_settings") as mock_settings_fn,
            patch("app.chains.baseline_chain.ChatOpenAI") as mock_llm_cls,
        ):
            _patch_settings(mock_settings_fn)
            mock_instance = MagicMock()
            mock_llm_with_tools = MagicMock()
            mock_instance.bind_tools.return_value = mock_llm_with_tools
            mock_llm_with_tools.ainvoke = AsyncMock(
                side_effect=[
                    # First call: unknown tool name.
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "nonexistent_tool",
                                "args": {},
                                "id": "tc_xyz",
                                "type": "tool_call",
                            }
                        ],
                    ),
                    # Second call: LLM synthesises without tool.
                    _ai_text("I cannot process that request."),
                ]
            )
            mock_llm_cls.return_value = mock_instance

            from app.chains.baseline_chain import run_baseline_chain

            result = await run_baseline_chain("Do something impossible.")

        # Should return a response without crashing.
        assert result.role == "assistant"
        assert result.content != ""
