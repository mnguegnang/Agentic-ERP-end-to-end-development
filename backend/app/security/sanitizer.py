"""PII sanitisation — regex scrubbing (Blueprint §2.4).

Stage 4 will add presidio-analyzer for more comprehensive PII detection.
"""
from __future__ import annotations

import re

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\b(\+?1[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}\b")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def scrub_pii(text: str) -> str:
    """Replace identifiable PII tokens before injecting text into LLM context."""
    text = _EMAIL_RE.sub("[EMAIL_REDACTED]", text)
    text = _PHONE_RE.sub("[PHONE_REDACTED]", text)
    text = _SSN_RE.sub("[SSN_REDACTED]", text)
    return text
