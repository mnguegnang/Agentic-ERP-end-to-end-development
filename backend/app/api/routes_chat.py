"""WebSocket chat endpoint — wired to the Stage 4 LangGraph orchestrator (Blueprint §4.1)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.agents.orchestrator import run_orchestrator
from app.api.schemas import WsMessage

logger = logging.getLogger(__name__)

router = APIRouter()


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
                response = await run_orchestrator(msg.content)
                await websocket.send_json(response.model_dump())
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
