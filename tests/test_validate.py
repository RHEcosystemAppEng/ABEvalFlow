"""Tests for scripts/validate.py and abevalflow/schemas.py."""

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
from abevalflow.schemas import (
    CopySpec,
    ExperimentConfig,
    ExperimentType,
    GenerationMode,
    SubmissionMetadata,
    VariantSpec,
)

VALID_METADATA = {
    "schema_version": "1.0",
    "name": "my-skill",
    "description": "A test submission",
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

    def test_name_with_spaces_rejected(self) -> None:
        data = {**VALID_METADATA, "name": "My Skill"}
        with pytest.raises(ValidationError, match="OCI-safe"):
            SubmissionMetadata(**data)

    def test_name_with_uppercase_rejected(self) -> None:
        data = {**VALID_METADATA, "name": "MySkill"}
        with pytest.raises(ValidationError, match="OCI-safe"):
            SubmissionMetadata(**data)

    def test_name_starting_with_hyphen_rejected(self) -> None:
        data = {**VALID_METADATA, "name": "-my-skill"}
        with pytest.raises(ValidationError, match="OCI-safe"):
            SubmissionMetadata(**data)

    def test_name_with_trailing_separator_rejected(self) -> None:
        data = {**VALID_METADATA, "name": "my-skill-"}
        with pytest.raises(ValidationError, match="OCI-safe"):
            SubmissionMetadata(**data)

    def test_name_with_consecutive_dots_rejected(self) -> None:
        data = {**VALID_METADATA, "name": "my..skill"}
        with pytest.raises(ValidationError, match="OCI-safe"):
            SubmissionMetadata(**data)

    def test_name_with_dots_and_underscores_valid(self) -> None:
        model = SubmissionMetadata(**{**VALID_METADATA, "name": "my_skill.v2"})
        assert model.name == "my_skill.v2"

    def test_invalid_generation_mode(self) -> None:
        data = {**VALID_METADATA, "generation_mode": "unknown"}
        with pytest.raises(ValidationError):
            SubmissionMetadata(**data)

    def test_minimal_metadata_only_name(self) -> None:
        model = SubmissionMetadata(name="my-skill")
        assert model.schema_version == "1.0"
        assert model.generation_mode == GenerationMode.MANUAL
        assert model.version == "0.1.0"
        assert model.persona is None
        assert model.description is None
        assert model.author is None

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
# CopySpec tests
# ──────────────────────────────────────────────


class TestCopySpec:
    def test_valid_copy_spec(self) -> None:
        cs = CopySpec(src="skills", dest="/skills")
        assert cs.src == "skills"
        assert cs.dest == "/skills"

    def test_trailing_slash_stripped(self) -> None:
        cs = CopySpec(src="skills/", dest="/skills")
        assert cs.src == "skills"

    def test_multiple_trailing_slashes_stripped(self) -> None:
        cs = CopySpec(src="skills///", dest="/skills")
        assert cs.src == "skills"

    def test_absolute_src_rejected(self) -> None:
        with pytest.raises(ValidationError, match="relative"):
            CopySpec(src="/etc/passwd", dest="/skills")

    def test_path_traversal_rejected(self) -> None:
        with pytest.raises(ValidationError, match="relative"):
            CopySpec(src="../secrets", dest="/skills")

    def test_dotdot_in_middle_rejected(self) -> None:
        with pytest.raises(ValidationError, match="relative"):
            CopySpec(src="foo/../bar", dest="/skills")

    def test_empty_src_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            CopySpec(src="", dest="/skills")

    def test_slash_only_src_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CopySpec(src="/", dest="/skills")

    def test_dest_trailing_slash_stripped(self) -> None:
        cs = CopySpec(src="skills", dest="/skills/")
        assert cs.dest == "/skills"

    def test_dest_empty_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            CopySpec(src="skills", dest="")

    def test_dest_dotdot_rejected(self) -> None:
        with pytest.raises(ValidationError, match="\\.\\."):
            CopySpec(src="skills", dest="/foo/../bar")

    def test_dest_relative_rejected(self) -> None:
        with pytest.raises(ValidationError, match="absolute"):
            CopySpec(src="skills", dest="skills")

    def test_dest_slash_only_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            CopySpec(src="skills", dest="/")


# ──────────────────────────────────────────────
# VariantSpec tests
# ──────────────────────────────────────────────


class TestVariantSpec:
    def test_empty_variant(self) -> None:
        vs = VariantSpec()
        assert vs.copy_dirs == []
        assert vs.env_from_secrets == {}

    def test_valid_copy_list(self) -> None:
        vs = VariantSpec(
            copy=[
                CopySpec(src="skills", dest="/skills"),
                CopySpec(src="docs", dest="/workspace/docs"),
            ]
        )
        assert len(vs.copy_dirs) == 2

    def test_copy_dirs_via_field_name(self) -> None:
        vs = VariantSpec(
            copy_dirs=[CopySpec(src="skills", dest="/skills")]
        )
        assert len(vs.copy_dirs) == 1

    def test_duplicate_src_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate"):
            VariantSpec(
                copy=[
                    CopySpec(src="skills", dest="/skills"),
                    CopySpec(src="skills", dest="/other"),
                ]
            )

    def test_env_from_secrets(self) -> None:
        vs = VariantSpec(
            env_from_secrets={"ANTHROPIC_API_KEY": "llm-keys/anthropic-key"}
        )
        assert vs.env_from_secrets["ANTHROPIC_API_KEY"] == "llm-keys/anthropic-key"


# ──────────────────────────────────────────────
# ExperimentConfig tests
# ──────────────────────────────────────────────


class TestExperimentConfig:
    def test_default_config(self) -> None:
        ec = ExperimentConfig()
        assert ec.type == ExperimentType.SKILL
        assert ec.n_trials == 20
        assert len(ec.treatment.copy_dirs) == 2
        assert ec.treatment.copy_dirs[0].src == "skills"
        assert ec.treatment.copy_dirs[1].src == "docs"
        assert ec.control.copy_dirs == []

    def test_valid_experiment_types(self) -> None:
        for t in ("skill", "model", "prompt", "custom"):
            ec = ExperimentConfig(type=t)
            assert ec.type == ExperimentType(t)

    def test_invalid_experiment_type_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(type="unknown")

    def test_n_trials_lower_bound(self) -> None:
        ec = ExperimentConfig(n_trials=1)
        assert ec.n_trials == 1

    def test_n_trials_upper_bound(self) -> None:
        ec = ExperimentConfig(n_trials=100)
        assert ec.n_trials == 100

    def test_n_trials_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(n_trials=0)

    def test_n_trials_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(n_trials=-1)

    def test_n_trials_over_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ExperimentConfig(n_trials=101)

    def test_custom_treatment_and_control(self) -> None:
        ec = ExperimentConfig(
            type="model",
            n_trials=10,
            treatment=VariantSpec(
                env_from_secrets={"MODEL_ENDPOINT": "models/endpoint-a"}
            ),
            control=VariantSpec(
                env_from_secrets={"MODEL_ENDPOINT": "models/endpoint-b"}
            ),
        )
        assert ec.type == ExperimentType.MODEL
        assert ec.n_trials == 10
        assert ec.treatment.copy_dirs == []
        assert "MODEL_ENDPOINT" in ec.treatment.env_from_secrets


# ──────────────────────────────────────────────
# ExperimentConfig in SubmissionMetadata
# ──────────────────────────────────────────────


class TestSubmissionMetadataExperiment:
    def test_default_experiment_when_omitted(self) -> None:
        model = SubmissionMetadata(name="my-skill")
        assert model.experiment.type == ExperimentType.SKILL
        assert model.experiment.n_trials == 20

    def test_backward_compat_valid_metadata_unchanged(self) -> None:
        model = SubmissionMetadata(**VALID_METADATA)
        assert model.experiment.type == ExperimentType.SKILL
        assert model.experiment.n_trials == 20

    def test_explicit_experiment_config(self) -> None:
        data = {
            **VALID_METADATA,
            "experiment": {
                "type": "model",
                "n_trials": 5,
                "treatment": {
                    "env_from_secrets": {"MODEL": "models/gpt-4o"}
                },
                "control": {
                    "env_from_secrets": {"MODEL": "models/gpt-3.5"}
                },
            },
        }
        model = SubmissionMetadata(**data)
        assert model.experiment.type == ExperimentType.MODEL
        assert model.experiment.n_trials == 5

    def test_experiment_with_copy_specs(self) -> None:
        data = {
            **VALID_METADATA,
            "experiment": {
                "type": "custom",
                "treatment": {
                    "copy": [{"src": "skills", "dest": "/skills"}]
                },
            },
        }
        model = SubmissionMetadata(**data)
        assert model.experiment.type == ExperimentType.CUSTOM
        assert model.experiment.treatment.copy_dirs[0].src == "skills"
        assert model.experiment.control.copy_dirs == []

    def test_invalid_experiment_type_in_metadata_rejected(self) -> None:
        data = {**VALID_METADATA, "experiment": {"type": "invalid"}}
        with pytest.raises(ValidationError):
            SubmissionMetadata(**data)

    def test_invalid_n_trials_in_metadata_rejected(self) -> None:
        data = {**VALID_METADATA, "experiment": {"n_trials": 0}}
        with pytest.raises(ValidationError):
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

    def test_ai_mode_still_validates_files(self, valid_submission: Path) -> None:
        """AI-mode submissions must pass the same file checks as manual ones."""
        ai_meta = {**VALID_METADATA, "generation_mode": "ai"}
        (valid_submission / "metadata.yaml").write_text(yaml.dump(ai_meta))
        errors = validate_submission(valid_submission)
        assert errors == []

    def test_ai_mode_missing_instruction_detected(self, valid_submission: Path) -> None:
        ai_meta = {**VALID_METADATA, "generation_mode": "ai"}
        (valid_submission / "metadata.yaml").write_text(yaml.dump(ai_meta))
        (valid_submission / "instruction.md").unlink()
        errors = validate_submission(valid_submission)
        assert any("instruction.md is missing" in e for e in errors)

    def test_ai_mode_bad_test_syntax_detected(self, valid_submission: Path) -> None:
        ai_meta = {**VALID_METADATA, "generation_mode": "ai"}
        (valid_submission / "metadata.yaml").write_text(yaml.dump(ai_meta))
        (valid_submission / "tests" / "test_outputs.py").write_text("def bad(\n")
        errors = validate_submission(valid_submission)
        assert any("test_outputs.py does not compile" in e for e in errors)


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
