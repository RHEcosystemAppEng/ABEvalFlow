"""High-risk corpus category denylist for eval submissions (ABE-2).

Submissions must not declare or ship corpora tagged with these categories.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

DENYLISTED_CORPUS_CATEGORIES: frozenset[str] = frozenset(
    {
        "hr_personal",
        "customer_support_identifiable",
        "credentials_secrets",
        "financial_payment",
        "sensitive_slack_dms",
        "legal_privileged",
    }
)


def normalize_category(value: object) -> str | None:
    """Return a normalized category slug, or None if not a usable string."""
    if not isinstance(value, str):
        return None
    slug = value.strip().lower().replace("-", "_").replace(" ", "_")
    return slug or None


def collect_categories(values: object) -> list[str]:
    """Extract category slugs from a scalar or list value."""
    if isinstance(values, str):
        slug = normalize_category(values)
        return [slug] if slug else []
    if isinstance(values, list):
        out: list[str] = []
        for item in values:
            slug = normalize_category(item)
            if slug:
                out.append(slug)
        return out
    return []


def find_denylisted(categories: list[str]) -> list[str]:
    """Return denylisted categories present in *categories* (stable order)."""
    seen: set[str] = set()
    blocked: list[str] = []
    for cat in categories:
        if cat in DENYLISTED_CORPUS_CATEGORIES and cat not in seen:
            seen.add(cat)
            blocked.append(cat)
    return blocked


def _categories_from_mapping(data: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for key in ("corpus_categories", "corpus_category"):
        if key in data:
            found.extend(collect_categories(data[key]))
    return found


def scan_supportive_dir(supportive_dir: Path) -> list[str]:
    """Scan supportive/ YAML/JSON for declared corpus categories."""
    if not supportive_dir.is_dir():
        return []

    categories: list[str] = []
    for path in sorted(supportive_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".yaml", ".yml", ".json"}:
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if suffix == ".json":
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                categories.extend(_categories_from_mapping(data))
            continue
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError:
            continue
        if isinstance(data, dict):
            categories.extend(_categories_from_mapping(data))
    return categories


def validate_corpus_categories(
    declared: list[str] | None,
    supportive_dir: Path,
) -> list[str]:
    """Return validation error strings when denylisted categories are found."""
    combined = collect_categories(declared) if declared else []
    combined.extend(scan_supportive_dir(supportive_dir))
    blocked = find_denylisted(combined)
    if not blocked:
        return []
    listed = ", ".join(sorted(blocked))
    return [
        "corpus categories are denylisted for eval submissions (ABE-2): "
        f"{listed}. Remove High-risk corpora or use approved public fixtures."
    ]
