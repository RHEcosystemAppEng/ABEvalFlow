# #ai-generated-oracle
"""Deterministic tests for URL shortener skill."""

import importlib
import sys
from pathlib import Path

WORKSPACE = Path("/workspace")


def _load_shortener():
    sys.path.insert(0, str(WORKSPACE))
    if "shortener" in sys.modules:
        del sys.modules["shortener"]
    return importlib.import_module("shortener")


def test_shorten_returns_string():
    mod = _load_shortener()
    result = mod.shorten("https://example.com")
    assert isinstance(result, str)


def test_short_id_length():
    mod = _load_shortener()
    short_id = mod.shorten("https://example.com/page")
    assert len(short_id) == 6


def test_short_id_is_alphanumeric():
    mod = _load_shortener()
    short_id = mod.shorten("https://example.com/test")
    assert short_id.isalnum()


def test_expand_returns_original_url():
    mod = _load_shortener()
    url = "https://example.com/roundtrip"
    short_id = mod.shorten(url)
    assert mod.expand(short_id) == url


def test_shorten_invalid_url_raises():
    mod = _load_shortener()
    try:
        mod.shorten("ftp://invalid.com")
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_expand_unknown_id_raises():
    mod = _load_shortener()
    try:
        mod.expand("XXXXXX")
        assert False, "Expected KeyError"
    except KeyError:
        pass


def test_multiple_urls_get_different_ids():
    mod = _load_shortener()
    id1 = mod.shorten("https://example.com/a")
    id2 = mod.shorten("https://example.com/b")
    assert id1 != id2
