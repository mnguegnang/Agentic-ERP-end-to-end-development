from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket) -> None:
    """WebSocket chat endpoint — wired to LangGraph orchestrator in Stage 3."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # TODO Stage 3: route through orchestrator graph
            await websocket.send_text(f"[stub] received: {data}")
    except WebSocketDisconnect:
        pass
