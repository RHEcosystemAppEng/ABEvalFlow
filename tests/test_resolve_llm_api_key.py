#!/usr/bin/env python3
"""Tests for scripts/resolve_llm_api_key.sh."""

from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "resolve_llm_api_key.sh"


def run(*args: str) -> str:
    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def test_auto_generates_from_run_name() -> None:
    assert run("", "abevalflow-run-abc") == "sk-run-abevalflow-run-abc"


def test_uses_explicit_key_when_provided() -> None:
    assert run("mock", "abevalflow-run-abc") == "mock"


def test_explicit_key_preserved_for_correlation_override() -> None:
    key = run("sk-run-custom", "ignored-run")
    assert key == "sk-run-custom"
