"""Tests for high-risk corpus category denylist (ABE-2)."""

from pathlib import Path

import yaml

from abevalflow.corpus_denylist import (
    DENYLISTED_CORPUS_CATEGORIES,
    find_denylisted,
    scan_supportive_dir,
    validate_corpus_categories,
)
from scripts.validate import validate_submission


def test_denylist_contains_expected_categories() -> None:
    assert "hr_personal" in DENYLISTED_CORPUS_CATEGORIES
    assert "credentials_secrets" in DENYLISTED_CORPUS_CATEGORIES
    assert "public_docs" not in DENYLISTED_CORPUS_CATEGORIES


def test_find_denylisted_returns_blocked_only() -> None:
    blocked = find_denylisted(["public_docs", "hr_personal", "hr_personal"])
    assert blocked == ["hr_personal"]


def test_scan_supportive_dir_reads_yaml_and_json(tmp_path: Path) -> None:
    supportive = tmp_path / "supportive"
    supportive.mkdir()
    (supportive / "fixture.yaml").write_text(yaml.dump({"corpus_categories": ["legal_privileged"]}))
    (supportive / "fixture.json").write_text('{"corpus_category": "financial_payment"}')
    assert set(scan_supportive_dir(supportive)) == {"legal_privileged", "financial_payment"}


def test_validate_corpus_categories_errors_on_denylisted(tmp_path: Path) -> None:
    supportive = tmp_path / "supportive"
    supportive.mkdir()
    errors = validate_corpus_categories(["customer_support_identifiable"], supportive)
    assert len(errors) == 1
    assert "customer_support_identifiable" in errors[0]


def test_validate_submission_fails_on_metadata_declared_category(tmp_path: Path) -> None:
    (tmp_path / "metadata.yaml").write_text(
        yaml.dump({"name": "bad-corpus", "corpus_categories": ["sensitive_slack_dms"]})
    )
    (tmp_path / "instruction.md").write_text("Do the thing.")
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text("---\nname: bad-corpus\n---\n\nSkill body.")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_outputs.py").write_text("def test_ok():\n    assert True\n")

    errors = validate_submission(tmp_path)
    assert any("sensitive_slack_dms" in e for e in errors)


def test_validate_metadata_normalizes_hyphen_and_case(tmp_path: Path) -> None:
    """Regression: hr-personal and HR-Personal must be caught, not just hr_personal."""
    supportive = tmp_path / "supportive"
    supportive.mkdir()
    errors = validate_corpus_categories(["hr-personal"], supportive)
    assert len(errors) == 1
    assert "hr_personal" in errors[0]

    errors2 = validate_corpus_categories(["HR-Personal", "Financial Payment"], supportive)
    assert len(errors2) == 1
    assert "hr_personal" in errors2[0]
    assert "financial_payment" in errors2[0]


def test_validate_submission_passes_without_denylisted_categories(tmp_path: Path) -> None:
    (tmp_path / "metadata.yaml").write_text(yaml.dump({"name": "ok-corpus"}))
    (tmp_path / "instruction.md").write_text("Do the thing.")
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text("---\nname: ok-corpus\n---\n\nSkill body.")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_outputs.py").write_text("def test_ok():\n    assert True\n")

    errors = validate_submission(tmp_path)
    assert not any("denylisted" in e for e in errors)
