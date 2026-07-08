"""Unit tests — shell-exported secret shadowing .env (Blueprint config.py).

pydantic-settings gives an actual process environment variable priority over
the same key in .env, silently. A stale shell export (e.g. GITHUB_TOKEN left
over from an earlier debugging session) then shadows a freshly rotated .env
value with no indication why — auth failures look completely unrelated. These
tests pin that the mismatch is now surfaced as a loud warning instead.
"""

from __future__ import annotations

import logging

from app.config import _parse_dotenv, _warn_if_shell_overrides_dotenv


def test_parse_dotenv_reads_key_value_pairs(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# a comment\nGITHUB_TOKEN=abc123\nPG_PASSWORD='quoted'\n\nNOT_A_LINE\n")
    values = _parse_dotenv(env_file)
    assert values == {"GITHUB_TOKEN": "abc123", "PG_PASSWORD": "quoted"}


def test_parse_dotenv_missing_file_returns_empty(tmp_path) -> None:
    assert _parse_dotenv(tmp_path / "missing.env") == {}


def test_warns_when_shell_env_mismatches_dotenv(monkeypatch, tmp_path, caplog) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GITHUB_TOKEN=correct_new_token\n")
    monkeypatch.setattr("app.config._REPO_ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "stale_old_token")

    with caplog.at_level(logging.WARNING, logger="app.config"):
        _warn_if_shell_overrides_dotenv()

    assert any("GITHUB_TOKEN" in r.message and "shell" in r.message.lower() for r in caplog.records)


def test_no_warning_when_shell_env_matches_dotenv(monkeypatch, tmp_path, caplog) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GITHUB_TOKEN=same_token\n")
    monkeypatch.setattr("app.config._REPO_ROOT", tmp_path)
    monkeypatch.setenv("GITHUB_TOKEN", "same_token")

    with caplog.at_level(logging.WARNING, logger="app.config"):
        _warn_if_shell_overrides_dotenv()

    assert len(caplog.records) == 0


def test_no_warning_when_var_not_exported_in_shell(monkeypatch, tmp_path, caplog) -> None:
    """No shell export at all → no mismatch to report (the normal case)."""
    env_file = tmp_path / ".env"
    env_file.write_text("GITHUB_TOKEN=only_in_dotenv\n")
    monkeypatch.setattr("app.config._REPO_ROOT", tmp_path)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    with caplog.at_level(logging.WARNING, logger="app.config"):
        _warn_if_shell_overrides_dotenv()

    assert len(caplog.records) == 0
