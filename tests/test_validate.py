"""Tests for scripts/validate.py and skillsevalflow/schemas.py."""

import json
import shutil
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from scripts.validate import (
    MAX_SUPPORTIVE_SIZE_BYTES,
    main,
    validate_submission,
)
from skillsevalflow.schemas import GenerationMode, SubmissionMetadata

VALID_METADATA = {
    "schema_version": "1.0",
    "name": "my-skill",
    "description": "A test skill",
    "persona": "rh-sre",
    "version": "0.1.0",
    "author": "test-team",
    "generation_mode": "manual",
}

TEST_OUTPUTS_PY = "def test_something():\n    assert True\n"


# ──────────────────────────────────────────────
# Schema tests
# ──────────────────────────────────────────────


class TestSubmissionMetadata:
    def test_valid_metadata(self) -> None:
        model = SubmissionMetadata(**VALID_METADATA)
        assert model.name == "my-skill"
        assert model.generation_mode == GenerationMode.MANUAL
        assert model.tags is None

    def test_valid_with_tags(self) -> None:
        data = {**VALID_METADATA, "tags": ["openshift", "sre"]}
        model = SubmissionMetadata(**data)
        assert model.tags == ["openshift", "sre"]

    def test_ai_generation_mode(self) -> None:
        data = {**VALID_METADATA, "generation_mode": "ai"}
        model = SubmissionMetadata(**data)
        assert model.generation_mode == GenerationMode.AI

    def test_missing_required_field(self) -> None:
        data = {k: v for k, v in VALID_METADATA.items() if k != "name"}
        with pytest.raises(ValidationError):
            SubmissionMetadata(**data)

    def test_empty_name_rejected(self) -> None:
        data = {**VALID_METADATA, "name": ""}
        with pytest.raises(ValidationError):
            SubmissionMetadata(**data)

    def test_invalid_generation_mode(self) -> None:
        data = {**VALID_METADATA, "generation_mode": "unknown"}
        with pytest.raises(ValidationError):
            SubmissionMetadata(**data)

    def test_extra_fields_rejected(self) -> None:
        data = {**VALID_METADATA, "unknown_field": "value"}
        with pytest.raises(ValidationError, match="extra"):
            SubmissionMetadata(**data)

    def test_invalid_schema_version_format(self) -> None:
        data = {**VALID_METADATA, "schema_version": "banana"}
        with pytest.raises(ValidationError, match="MAJOR.MINOR"):
            SubmissionMetadata(**data)

    def test_schema_version_integer_string_rejected(self) -> None:
        data = {**VALID_METADATA, "schema_version": "1"}
        with pytest.raises(ValidationError, match="MAJOR.MINOR"):
            SubmissionMetadata(**data)


# ──────────────────────────────────────────────
# Fixture: builds a valid submission directory
# ──────────────────────────────────────────────


@pytest.fixture()
def valid_submission(tmp_path: Path) -> Path:
    sub = tmp_path / "my-skill"
    sub.mkdir()
    (sub / "instruction.md").write_text("Solve the task.\n")
    (sub / "skills").mkdir()
    (sub / "skills" / "SKILL.md").write_text("# Skill\nDo something.\n")
    (sub / "tests").mkdir()
    (sub / "tests" / "test_outputs.py").write_text(TEST_OUTPUTS_PY)
    (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
    return sub


# ──────────────────────────────────────────────
# Validation function tests
# ──────────────────────────────────────────────


class TestValidateSubmission:
    def test_valid_submission(self, valid_submission: Path) -> None:
        errors = validate_submission(valid_submission)
        assert errors == []

    def test_missing_instruction_md(self, valid_submission: Path) -> None:
        (valid_submission / "instruction.md").unlink()
        errors = validate_submission(valid_submission)
        assert any("instruction.md is missing" in e for e in errors)

    def test_empty_instruction_md(self, valid_submission: Path) -> None:
        (valid_submission / "instruction.md").write_text("")
        errors = validate_submission(valid_submission)
        assert any("instruction.md is empty" in e for e in errors)

    def test_whitespace_only_instruction_md(self, valid_submission: Path) -> None:
        (valid_submission / "instruction.md").write_text("   \n\n  ")
        errors = validate_submission(valid_submission)
        assert any("instruction.md is empty" in e for e in errors)

    def test_missing_skills_dir(self, valid_submission: Path) -> None:
        shutil.rmtree(valid_submission / "skills")
        errors = validate_submission(valid_submission)
        assert any("skills/ directory is missing" in e for e in errors)

    def test_skills_dir_missing_skill_md(self, valid_submission: Path) -> None:
        (valid_submission / "skills" / "SKILL.md").unlink()
        errors = validate_submission(valid_submission)
        assert any("SKILL.md is missing" in e for e in errors)

    def test_skills_dir_wrong_name_rejected(self, valid_submission: Path) -> None:
        (valid_submission / "skills" / "SKILL.md").unlink()
        (valid_submission / "skills" / "my-custom-skill.md").write_text("# Skill\n")
        errors = validate_submission(valid_submission)
        assert any("SKILL.md is missing" in e for e in errors)

    def test_skills_dir_empty_skill_md(self, valid_submission: Path) -> None:
        (valid_submission / "skills" / "SKILL.md").write_text("")
        errors = validate_submission(valid_submission)
        assert any("SKILL.md is empty" in e for e in errors)

    def test_skills_dir_whitespace_only_skill_md(self, valid_submission: Path) -> None:
        (valid_submission / "skills" / "SKILL.md").write_text("  \n\n  ")
        errors = validate_submission(valid_submission)
        assert any("SKILL.md is empty" in e for e in errors)

    def test_test_outputs_missing(self, valid_submission: Path) -> None:
        (valid_submission / "tests" / "test_outputs.py").unlink()
        errors = validate_submission(valid_submission)
        assert any("test_outputs.py is missing" in e for e in errors)

    def test_test_outputs_bad_syntax(self, valid_submission: Path) -> None:
        (valid_submission / "tests" / "test_outputs.py").write_text("def bad(\n")
        errors = validate_submission(valid_submission)
        assert any("test_outputs.py does not compile" in e for e in errors)

    def test_llm_judge_not_required(self, valid_submission: Path) -> None:
        errors = validate_submission(valid_submission)
        assert errors == []

    def test_llm_judge_valid(self, valid_submission: Path) -> None:
        (valid_submission / "tests" / "llm_judge.py").write_text("score = 1\n")
        errors = validate_submission(valid_submission)
        assert errors == []

    def test_llm_judge_bad_syntax(self, valid_submission: Path) -> None:
        (valid_submission / "tests" / "llm_judge.py").write_text("def bad(\n")
        errors = validate_submission(valid_submission)
        assert any("llm_judge.py does not compile" in e for e in errors)

    def test_metadata_missing(self, valid_submission: Path) -> None:
        (valid_submission / "metadata.yaml").unlink()
        errors = validate_submission(valid_submission)
        assert any("metadata.yaml is missing" in e for e in errors)

    def test_metadata_invalid_yaml(self, valid_submission: Path) -> None:
        (valid_submission / "metadata.yaml").write_text(": : bad\n  :\n")
        errors = validate_submission(valid_submission)
        assert any("metadata.yaml" in e for e in errors)

    def test_metadata_not_a_mapping(self, valid_submission: Path) -> None:
        (valid_submission / "metadata.yaml").write_text("- item1\n- item2\n")
        errors = validate_submission(valid_submission)
        assert any("must contain a YAML mapping" in e for e in errors)

    def test_metadata_schema_error(self, valid_submission: Path) -> None:
        bad = {**VALID_METADATA}
        del bad["name"]
        (valid_submission / "metadata.yaml").write_text(yaml.dump(bad))
        errors = validate_submission(valid_submission)
        assert any("metadata.yaml validation" in e for e in errors)

    def test_supportive_under_limit(self, valid_submission: Path) -> None:
        sup = valid_submission / "supportive"
        sup.mkdir()
        (sup / "data.txt").write_bytes(b"x" * 1024)
        errors = validate_submission(valid_submission)
        assert errors == []

    def test_supportive_over_limit(self, valid_submission: Path) -> None:
        sup = valid_submission / "supportive"
        sup.mkdir()
        (sup / "big.bin").write_bytes(b"x" * (MAX_SUPPORTIVE_SIZE_BYTES + 1))
        errors = validate_submission(valid_submission)
        assert any("exceeds 50 MB" in e for e in errors)

    def test_no_supportive_dir_is_fine(self, valid_submission: Path) -> None:
        errors = validate_submission(valid_submission)
        assert errors == []

    def test_multiple_errors_collected(self, valid_submission: Path) -> None:
        (valid_submission / "instruction.md").unlink()
        (valid_submission / "metadata.yaml").unlink()
        errors = validate_submission(valid_submission)
        assert len(errors) >= 2


# ──────────────────────────────────────────────
# CLI (main) tests
# ──────────────────────────────────────────────


class TestMain:
    def test_valid_returns_zero(
        self,
        valid_submission: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main([str(valid_submission)])
        assert rc == 0
        output = json.loads(capsys.readouterr().out)
        assert output["valid"] is True
        assert output["errors"] == []

    def test_invalid_returns_one(
        self,
        valid_submission: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        (valid_submission / "instruction.md").unlink()
        rc = main([str(valid_submission)])
        assert rc == 1
        output = json.loads(capsys.readouterr().out)
        assert output["valid"] is False
        assert len(output["errors"]) > 0

    def test_nonexistent_dir(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rc = main([str(tmp_path / "does-not-exist")])
        assert rc == 1
        output = json.loads(capsys.readouterr().out)
        assert output["valid"] is False
