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

from functools import lru_cache
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root is three levels above this file:
#   backend/app/config.py → backend/app/ → backend/ → <repo root>
_REPO_ROOT = Path(__file__).parents[2]


def _load_yaml() -> dict:
    """Load config.yaml from repo root. Called once per process."""
    cfg_path = _REPO_ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


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
    return Settings()  # type: ignore[call-arg]
