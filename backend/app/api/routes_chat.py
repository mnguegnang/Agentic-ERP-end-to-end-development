"""WebSocket chat endpoint — wired to the Stage 3 baseline chain (Blueprint §3.3)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.schemas import WsMessage
from app.chains.baseline_chain import run_baseline_chain

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """Accept a WebSocket connection and route each user message through the
    Stage 3 baseline chain (ChatOpenAI + solve_mcnf tool).

    Message format (client → server):  {"role": "user", "content": "..."}
    Message format (server → client):  {"role": "assistant", "content": "...",
                                         "tool_used": "...", "solver_result": {...}}
    """
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()

            # Validate incoming message schema.
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

            # Run the baseline chain and stream the response.
            try:
                response = await run_baseline_chain(msg.content)
                await websocket.send_json(response.model_dump())
            except Exception as exc:
                logger.exception("Baseline chain error for query: %r", msg.content)
                await websocket.send_json(
                    {
                        "role": "assistant",
                        "content": f"Internal error: {exc}",
                    }
                )

    except WebSocketDisconnect:
        logger.debug("Client disconnected from /ws/chat")
