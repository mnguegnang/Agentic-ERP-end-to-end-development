"""Unit tests — manager-password gate on the HiTL approval endpoint.

Contract:
  * wrong password  → 403, and the decision must stay pending
  * empty/missing password → schema-level validation error (422 at the API)
  * no password configured on the server → 503 (approvals locked, not open)
  * correct password → passes the gate
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from app.api.routes_approve import ApprovalRequest, _verify_manager_password
from fastapi import HTTPException
from pydantic import ValidationError


def _settings_with(password: str) -> MagicMock:
    s = MagicMock()
    s.manager_approval_password = password
    return s


def test_correct_password_passes() -> None:
    with patch(
        "app.api.routes_approve.get_settings",
        return_value=_settings_with("s3cret"),
    ):
        _verify_manager_password("s3cret")  # must not raise


def test_wrong_password_raises_403() -> None:
    with patch(
        "app.api.routes_approve.get_settings",
        return_value=_settings_with("s3cret"),
    ):
        with pytest.raises(HTTPException) as exc:
            _verify_manager_password("not-the-password")
    assert exc.value.status_code == 403


def test_unconfigured_password_locks_approvals_with_503() -> None:
    with patch(
        "app.api.routes_approve.get_settings",
        return_value=_settings_with(""),
    ):
        with pytest.raises(HTTPException) as exc:
            _verify_manager_password("anything")
    assert exc.value.status_code == 503


def test_empty_password_rejected_even_if_configured_empty() -> None:
    """'' == '' must NOT authenticate — unset config locks, never opens."""
    with patch(
        "app.api.routes_approve.get_settings",
        return_value=_settings_with(""),
    ):
        with pytest.raises(HTTPException) as exc:
            _verify_manager_password("")
    assert exc.value.status_code == 503


def test_request_schema_requires_password() -> None:
    with pytest.raises(ValidationError):
        ApprovalRequest(approved=True)  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        ApprovalRequest(approved=True, password="")  # min_length=1
    # And a well-formed request parses
    req = ApprovalRequest(approved=False, password="s3cret")
    assert req.approved is False
    assert req.password == "s3cret"
