"""WebSocket chat endpoint — wired to the Stage 4 LangGraph orchestrator (Blueprint §4.1)."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agents.orchestrator import run_orchestrator
from app.api.schemas import WsMessage, WsResponse
from app.cache.semantic_cache import get_semantic_cache

logger = logging.getLogger(__name__)

router = APIRouter()


def _is_cacheable(response: WsResponse) -> bool:
    """Only cache deterministic, low-stakes answers.

    Never cached:
      * approval-gated decisions (carry a live decision_id),
      * degraded/error responses (``error`` set — e.g. LLM outage fallbacks;
        caching those would replay stale template answers for the full TTL),
      * "unclear" classifications (clarification prompts, usually a symptom
        of the keyword fallback running during an LLM outage).
    """
    return (
        not response.human_approval_required
        and response.decision_id is None
        and response.error is None
        and response.intent != "unclear"
        and bool(response.content)
    )


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """Accept a WebSocket connection and route each user message through the
    Stage 4 LangGraph orchestrator (classify → kg/contract/solver → synthesize).

    Message format (client → server):  {"role": "user", "content": "..."}
    Message format (server → client):  WsResponse JSON (role, content, intent,
                                        solver_result, rag_documents, ...)
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()

            try:
                msg = WsMessage.model_validate_json(raw)
            except Exception:
                await websocket.send_json(
                    {
                        "role": "assistant",
                        "content": (
                            "Invalid message format. "
                            'Expected JSON {"role": "user", "content": "..."}.'
                        ),
                    }
                )
                continue

            try:
                cache = get_semantic_cache()
                cached = await cache.get(msg.content)
                if cached is not None:
                    await websocket.send_json(cached)
                    continue

                response = await run_orchestrator(msg.content)
                # Coerce any non-JSON-native values (UUID, datetime, Decimal…)
                # to strings so a single stray field can't crash the whole
                # response with "Object of type X is not JSON serializable".
                payload = json.loads(json.dumps(response.model_dump(), default=str))
                await websocket.send_json(payload)
                if _is_cacheable(response):
                    await cache.set(msg.content, payload)
            except Exception as exc:
                logger.exception("Orchestrator error for query: %r", msg.content)
                await websocket.send_json(
                    {
                        "role": "assistant",
                        "content": f"Internal error: {exc}",
                    }
                )

    except WebSocketDisconnect:
        logger.debug("Client disconnected from /ws/chat")
