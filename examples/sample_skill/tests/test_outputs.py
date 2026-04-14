"""Verification tests for the greeting task."""

import importlib
import sys
from pathlib import Path


def _load_greeting():
    workspace = Path("/workspace")
    if workspace.exists():
        sys.path.insert(0, str(workspace))
    return importlib.import_module("greeting")


def test_greet_with_name():
    mod = _load_greeting()
    result = mod.greet("Alice")
    assert result == "Hello, Alice! Welcome aboard."


def test_greet_empty_string():
    mod = _load_greeting()
    result = mod.greet("")
    assert result == "Hello, stranger! Welcome aboard."
