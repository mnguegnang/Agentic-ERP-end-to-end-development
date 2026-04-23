"""Red-team test fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_orchestrator_graph() -> None:
    """Reset the LangGraph singleton before each test.

    The orchestrator caches the compiled graph in _GRAPH. Resetting it ensures
    each test compiles a fresh graph with whatever patches are active at the
    time run_orchestrator() is called, so patch() context managers take effect.
    """
    import app.agents.orchestrator as _orch

    _orch._GRAPH = None
    yield
    _orch._GRAPH = None
