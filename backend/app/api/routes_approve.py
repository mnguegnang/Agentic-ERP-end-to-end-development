"""Human-in-the-Loop approval REST endpoints (Blueprint §4.7).

Flow:
    1.  Orchestrator detects cost > threshold → generates decision_id, stores
        JSON record in Redis under key  ``hitl:{decision_id}``  (TTL 24 h).
    2.  Frontend shows ⚠️ banner with Approve / Reject buttons.
    3.  Manager clicks → browser POSTs to  ``POST /api/approve/{decision_id}``.
    4.  This endpoint updates the Redis record (status → approved | rejected)
        and returns the updated record.
    5.  Frontend reads the response and shows the outcome in the chat.
    6.  ``GET /api/approve/{decision_id}`` lets the frontend poll status.
"""

from __future__ import annotations

import hmac
import json
import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/approve", tags=["approval"])

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ApprovalRequest(BaseModel):
    approved: bool = Field(..., description="True = approve, False = reject")
    approved_by: str = Field(default="supply-chain-manager", description="Approver identifier")
    reason: str | None = Field(default=None, description="Optional reason / comment")
    password: str = Field(
        ...,
        min_length=1,
        description="Manager approval password — required to approve OR reject",
    )


class ApprovalRecord(BaseModel):
    decision_id: str
    status: str  # pending | approved | rejected
    query: str
    intent: str | None
    total_cost: float
    approved_by: str | None
    reason: str | None
    solver_output: dict | None
    final_response: str | None = None  # synthesized answer produced on graph resume


# ---------------------------------------------------------------------------
# Shared Redis client (lazy)
# ---------------------------------------------------------------------------

_REDIS: aioredis.Redis | None = None  # type: ignore[type-arg]


def _get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    global _REDIS
    if _REDIS is None:
        _REDIS = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _REDIS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _verify_manager_password(provided: str) -> None:
    """Constant-time check of the manager approval password.

    Raises 403 on mismatch and 503 when no password is configured — approvals
    are locked out rather than open by default.  Called BEFORE the decision
    record is loaded so an unauthorized caller cannot probe which decision IDs
    exist (a wrong password always yields the same 403).
    """
    expected = get_settings().manager_approval_password
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "Manager approval is not configured on this server "
                "(MANAGER_APPROVAL_PASSWORD is unset) — approvals are locked."
            ),
        )
    if not hmac.compare_digest(provided.encode(), expected.encode()):
        logger.warning("HiTL approval attempt with invalid manager password — denied")
        raise HTTPException(
            status_code=403,
            detail="Invalid manager password — approval denied.",
        )


async def _load_record(decision_id: str) -> dict:
    """Load and parse the pending-decision record from Redis."""
    raw = await _get_redis().get(f"hitl:{decision_id}")
    if raw is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Decision {decision_id!r} not found. "
                "It may have expired (TTL 24 h) or never existed."
            ),
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt record in Redis: {exc}") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/{decision_id}", response_model=ApprovalRecord, summary="Check decision status")
async def get_approval_status(decision_id: str) -> ApprovalRecord:
    """Return the current status of a pending HiTL decision."""
    record = await _load_record(decision_id)
    return ApprovalRecord(**record)


@router.post(
    "/{decision_id}",
    response_model=ApprovalRecord,
    summary="Approve or reject a decision",
)
async def submit_approval(decision_id: str, body: ApprovalRequest) -> ApprovalRecord:
    """Supply-chain manager approves or rejects a flagged routing decision.

    - Verifies the manager password FIRST — wrong password → 403, and the
      decision stays pending (neither approval nor rejection is recorded).
    - Resumes the LangGraph run paused at the human-approval gate, which
      synthesizes the final answer with the decision applied.
    - Updates Redis record  ``status → approved | rejected``.
    - Returns the updated record so the frontend can display the outcome.
    """
    _verify_manager_password(body.password)

    record = await _load_record(decision_id)

    if record.get("status") != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Decision already resolved: status={record['status']!r}",
        )

    # Resume the paused graph — this is what actually unblocks the pipeline.
    from app.agents.orchestrator import resume_orchestrator

    resumed = await resume_orchestrator(
        decision_id,
        approved=body.approved,
        approved_by=body.approved_by,
        reason=body.reason,
    )
    if resumed is None:
        logger.warning(
            "No paused graph run for decision_id=%s (expired or process restart); "
            "recording the decision without a synthesized response",
            decision_id,
        )

    record["status"] = "approved" if body.approved else "rejected"
    record["approved_by"] = body.approved_by
    record["reason"] = body.reason
    record["final_response"] = resumed.content if resumed else None

    # Persist updated record (keep same TTL by re-setting with same key)
    ttl = await _get_redis().ttl(f"hitl:{decision_id}")
    ttl = max(ttl, 1)  # guard against already-expired edge case
    await _get_redis().setex(f"hitl:{decision_id}", ttl, json.dumps(record))

    action = "APPROVED" if body.approved else "REJECTED"
    logger.info(
        "HiTL decision %s %s by %s (cost=%.2f, reason=%r)",
        decision_id,
        action,
        body.approved_by,
        record.get("total_cost", 0),
        body.reason,
    )

    return ApprovalRecord(**record)
