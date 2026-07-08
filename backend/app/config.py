"""
backend/app/config.py
======================
Central configuration module.

Non-secret values are read from config.yaml (committed to git).
Secrets (API keys, passwords) are read from .env (gitignored).

Usage
-----
    from app.config import get_settings

    settings = get_settings()
    print(settings.llm_model)       # "gpt-4o"
    print(settings.github_token)    # from .env
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def _find_config_yaml() -> Path:
    """Search upward from this file for config.yaml.

    Works in both local dev (backend/app/config.py → 3 levels up to repo root)
    and Docker (WORKDIR=/app, file at /app/app/config.py → 2 levels up to /app).
    """
    here = Path(__file__).parent
    for directory in [here, here.parent, here.parent.parent, here.parent.parent.parent]:
        candidate = directory / "config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"config.yaml not found searching upward from {here}")


_REPO_ROOT = _find_config_yaml().parent


def _load_yaml() -> dict:
    """Load config.yaml. Called once per process."""
    with (_REPO_ROOT / "config.yaml").open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Shell-export footgun detection
#
# pydantic-settings gives an actual process environment variable priority
# over the same key in .env — silently.  If a shell still has an old
# GITHUB_TOKEN (or another secret) exported from an earlier debugging
# session, it shadows a freshly rotated .env value with zero indication why
# auth keeps failing (surfaces as a bare 401, hours later, in an unrelated
# test).  This check doesn't change precedence — a deliberate shell/CI
# override is legitimate — it just makes an *accidental* mismatch loud
# instead of silent.
# ---------------------------------------------------------------------------
_SECRET_ENV_VARS = (
    "GITHUB_TOKEN",
    "PG_PASSWORD",
    "NEO4J_PASSWORD",
    "MANAGER_APPROVAL_PASSWORD",
    "JWT_SECRET_KEY",
    "LANGSMITH_API_KEY",
)


def _parse_dotenv(path: Path) -> dict[str, str]:
    """Minimal KEY=VALUE reader for comparison only — pydantic-settings does
    the real parsing that actually configures the app."""
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _warn_if_shell_overrides_dotenv() -> None:
    dotenv_values = _parse_dotenv(_REPO_ROOT / ".env")
    for var in _SECRET_ENV_VARS:
        shell_value = os.environ.get(var)
        dotenv_value = dotenv_values.get(var)
        if shell_value is not None and dotenv_value is not None and shell_value != dotenv_value:
            logger.warning(
                "%s is exported in the shell environment and does NOT match "
                "the value in .env — the shell export silently wins (this is "
                "pydantic-settings' normal precedence, not a bug). If you "
                "rotated this secret in .env, run `unset %s` in every open "
                "terminal before restarting the app or tests, or the old "
                "value keeps being used and failures look unrelated (e.g. a "
                "bare 401/403 with no obvious cause).",
                var,
                var,
            )


# ---------------------------------------------------------------------------
# Settings — Pydantic V2 BaseSettings
# Fields declared here are loaded from .env / environment variables.
# Non-secret values (model names, thresholds) are read via @property from YAML.
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- GitHub Models API (primary LLM) ----------------------------------
    github_token: str = Field(..., description="GitHub PAT for GitHub Models API")

    # -- PostgreSQL --------------------------------------------------------
    database_url: str = Field(
        default="postgresql+asyncpg://aw_user:changeme@localhost:5432/adventureworks",
        description="Async SQLAlchemy connection URL",
    )
    pg_password: str = Field(default="changeme")

    # -- Neo4j -------------------------------------------------------------
    neo4j_uri: str = Field(default="bolt://localhost:7687")
    neo4j_password: str = Field(default="changeme")

    # -- Redis -------------------------------------------------------------
    redis_url: str = Field(default="redis://localhost:6379")

    # -- LangSmith ---------------------------------------------------------
    langsmith_api_key: str = Field(default="")
    langchain_project: str = Field(default="agentic-erp-supply-chain")
    langchain_tracing_v2: str = Field(default="false")

    # -- JWT ---------------------------------------------------------------
    jwt_secret_key: str = Field(default="changeme_replace_with_64_char_hex")
    jwt_algorithm: str = Field(default="HS256")
    jwt_expire_minutes: int = Field(default=60)

    # -- Human-in-the-Loop -------------------------------------------------
    manager_approval_password: str = Field(
        default="",
        description=(
            "Secret required to approve/reject HiTL decisions via "
            "POST /api/approve/{id}. Empty = approvals are locked out."
        ),
    )

    # -----------------------------------------------------------------------
    # YAML-backed properties (no Pydantic overhead on hot paths)
    # -----------------------------------------------------------------------

    @property
    def _yaml(self) -> dict:
        """Cached YAML config. Re-read only once per process."""
        return _load_yaml()

    @property
    def llm_base_url(self) -> str:
        return self._yaml["llm"]["base_url"]

    @property
    def llm_model(self) -> str:
        return self._yaml["llm"]["model"]

    @property
    def llm_temperature(self) -> float:
        return float(self._yaml["llm"]["temperature"])

    @property
    def llm_max_tokens(self) -> int:
        return int(self._yaml["llm"]["max_tokens"])

    @property
    def intent_confidence_threshold(self) -> float:
        return float(self._yaml["agent"]["intent_confidence_threshold"])

    @property
    def human_approval_cost_threshold(self) -> float:
        return float(self._yaml["agent"]["human_approval_cost_threshold"])

    @property
    def rag_config(self) -> dict:
        return self._yaml["rag"]

    @property
    def cache_ttl(self) -> int:
        return int(self._yaml["cache"]["ttl_seconds"])

    @property
    def solver_seed(self) -> int:
        return int(self._yaml["solvers"]["random_seed"])

    @property
    def solver_time_limit(self) -> int:
        return int(self._yaml["solvers"]["default_time_limit_sec"])

    @property
    def otel_endpoint(self) -> str:
        return self._yaml["observability"]["otel_endpoint"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Cached after first call — safe for both sync and async contexts.
    """
    _warn_if_shell_overrides_dotenv()
    return Settings()  # type: ignore[call-arg]
