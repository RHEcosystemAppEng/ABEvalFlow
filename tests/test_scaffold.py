"""Tests for scripts/scaffold.py — scaffold a submission into Harbor task dirs."""

from __future__ import annotations

import stat
import tomllib
from pathlib import Path

import pytest
import yaml

from scripts.scaffold import scaffold_submission

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"


@pytest.fixture()
def valid_submission(tmp_path: Path) -> Path:
    """Create a minimal valid submission directory."""
    sub = tmp_path / "my-skill"
    sub.mkdir()

    (sub / "instruction.md").write_text("Do the thing.\n")

    skills = sub / "skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text("# My Skill\nUse this skill to do the thing.\n")

    tests = sub / "tests"
    tests.mkdir()
    (tests / "test_outputs.py").write_text("def test_pass(): assert True\n")

    meta = {
        "schema_version": "1.0",
        "name": "my-skill",
        "description": "A test skill",
        "persona": "rh-developer",
        "version": "0.1.0",
        "author": "tester",
        "tags": ["test", "demo"],
        "generation_mode": "manual",
    }
    (sub / "metadata.yaml").write_text(yaml.dump(meta))

    return sub


@pytest.fixture()
def full_submission(valid_submission: Path) -> Path:
    """Extend valid_submission with optional dirs (docs, supportive, llm_judge)."""
    docs = valid_submission / "docs"
    docs.mkdir()
    (docs / "reference.md").write_text("# Reference\nSome docs.\n")

    supportive = valid_submission / "supportive"
    supportive.mkdir()
    (supportive / "data.json").write_text('{"key": "value"}\n')

    (valid_submission / "tests" / "llm_judge.py").write_text(
        "def judge(): return {'score': 1.0, 'rationale': 'ok'}\n"
    )

    return valid_submission


class TestScaffoldBasic:
    """Test scaffolding with a minimal submission (no optional dirs)."""

    def test_creates_treatment_and_control_dirs(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        assert treatment.is_dir()
        assert control.is_dir()
        assert treatment.name == "my-skill"
        assert control.name == "my-skill"

    def test_treatment_parent_dir_name(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        assert treatment.parent.name == "tasks-treatment"
        assert control.parent.name == "tasks-control"

    def test_treatment_dir_structure(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)

        assert (treatment / "instruction.md").is_file()
        assert (treatment / "task.toml").is_file()
        assert (treatment / "tests" / "test.sh").is_file()
        assert (treatment / "environment" / "Dockerfile").is_file()
        assert (treatment / "environment" / "instruction.md").is_file()
        assert (treatment / "environment" / "skills" / "SKILL.md").is_file()
        assert (treatment / "environment" / "tests" / "test_outputs.py").is_file()

    def test_control_dir_structure(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        _, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)

        assert (control / "instruction.md").is_file()
        assert (control / "task.toml").is_file()
        assert (control / "tests" / "test.sh").is_file()
        assert (control / "environment" / "Dockerfile").is_file()
        assert (control / "environment" / "tests" / "test_outputs.py").is_file()
        assert not (control / "environment" / "skills").exists()
        assert not (control / "environment" / "docs").exists()

    def test_treatment_dockerfile_contains_skills_copy(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (treatment / "environment" / "Dockerfile").read_text()
        assert "COPY skills/" in content

    def test_control_dockerfile_excludes_skills(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        _, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (control / "environment" / "Dockerfile").read_text()
        assert "COPY skills/" not in content
        assert "COPY docs/" not in content

    def test_test_sh_no_runtime_install(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (treatment / "tests" / "test.sh").read_text()
        assert "dnf install" not in content
        assert "curl -LsSf" not in content
        assert "source" not in content

    def test_test_sh_is_executable(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        for d in (treatment, control):
            mode = (d / "tests" / "test.sh").stat().st_mode
            assert mode & stat.S_IEXEC


class TestScaffoldTaskToml:
    """Test task.toml rendering."""

    def test_treatment_task_toml_has_skills_dir(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (treatment / "task.toml").read_text()
        assert 'skills_dir = "/skills"' in content

    def test_control_task_toml_no_skills_dir(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        _, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (control / "task.toml").read_text()
        assert "skills_dir" not in content

    def test_task_toml_metadata_fields(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (treatment / "task.toml").read_text()
        assert 'category = "rh-developer"' in content
        assert '"test"' in content
        assert '"demo"' in content

    def test_task_toml_is_valid_toml_without_llm_judge(
        self, valid_submission: Path, tmp_path: Path
    ):
        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        for d in (treatment, control):
            tomllib.loads((d / "task.toml").read_text())

    def test_task_toml_is_valid_toml_with_llm_judge(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, control = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        for d in (treatment, control):
            parsed = tomllib.loads((d / "task.toml").read_text())
            assert "verifier" in parsed

    def test_task_toml_llm_judge_env_key_populated(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        parsed = tomllib.loads((treatment / "task.toml").read_text())
        env = parsed["verifier"]["env"]
        assert "LLM_API_KEY" in env
        assert "LLM_API_BASE" in env
        assert "MODEL_NAME" in env

    def test_task_toml_custom_timeouts(self, valid_submission: Path, tmp_path: Path):
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["agent_timeout_sec"] = 1200.0
        meta["memory_mb"] = 4096
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        parsed = tomllib.loads((treatment / "task.toml").read_text())
        assert parsed["agent"]["timeout_sec"] == 1200.0
        assert parsed["environment"]["memory_mb"] == 4096


class TestScaffoldOptionalDirs:
    """Test scaffolding with optional directories (docs, supportive, llm_judge)."""

    def test_supportive_copied_when_present(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, control = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        for d in (treatment, control):
            assert (d / "environment" / "supportive" / "data.json").is_file()

    def test_docs_copied_only_for_treatment(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, control = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        assert (treatment / "environment" / "docs" / "reference.md").is_file()
        assert not (control / "environment" / "docs").exists()

    def test_dockerfile_includes_supportive_when_present(
        self, full_submission: Path, tmp_path: Path
    ):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        content = (treatment / "environment" / "Dockerfile").read_text()
        assert "COPY supportive/" in content

    def test_dockerfile_includes_docs_for_treatment(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        content = (treatment / "environment" / "Dockerfile").read_text()
        assert "COPY docs/" in content

    def test_no_supportive_in_dockerfile_when_absent(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (treatment / "environment" / "Dockerfile").read_text()
        assert "COPY supportive/" not in content

    def test_test_sh_includes_llm_judge_when_present(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        content = (treatment / "tests" / "test.sh").read_text()
        assert "llm_judge.py" in content

    def test_test_sh_excludes_llm_judge_when_absent(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (treatment / "tests" / "test.sh").read_text()
        assert "llm_judge.py" not in content


class TestScaffoldEdgeCases:
    """Edge cases and error handling."""

    def test_missing_metadata_raises(self, tmp_path: Path):
        sub = tmp_path / "bad-skill"
        sub.mkdir()
        (sub / "instruction.md").write_text("Hello")
        output = tmp_path / "output"
        with pytest.raises(FileNotFoundError):
            scaffold_submission(sub, output, TEMPLATES_DIR)

    def test_empty_tags_produces_empty_list(self, valid_submission: Path, tmp_path: Path):
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["tags"] = []
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (treatment / "task.toml").read_text()
        assert "tags = []" in content

    def test_none_tags_produces_empty_list(self, valid_submission: Path, tmp_path: Path):
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        del meta["tags"]
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (treatment / "task.toml").read_text()
        assert "tags = []" in content

    def test_minimal_metadata_scaffolds(self, tmp_path: Path):
        """A submission with only 'name' in metadata.yaml should scaffold."""
        sub = tmp_path / "minimal-skill"
        sub.mkdir()
        (sub / "instruction.md").write_text("Do it.\n")
        (sub / "skills").mkdir()
        (sub / "skills" / "SKILL.md").write_text("# Skill\n")
        (sub / "tests").mkdir()
        (sub / "tests" / "test_outputs.py").write_text("def test(): pass\n")
        (sub / "metadata.yaml").write_text(yaml.dump({"name": "minimal-skill"}))

        output = tmp_path / "output"
        treatment, control = scaffold_submission(sub, output, TEMPLATES_DIR)
        assert treatment.is_dir()
        assert control.is_dir()
        content = (treatment / "task.toml").read_text()
        assert 'category = "general"' in content
        tomllib.loads(content)

    def test_idempotent_scaffold(self, valid_submission: Path, tmp_path: Path):
        """Running scaffold twice should overwrite without error."""
        output = tmp_path / "output"
        scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        assert treatment.is_dir()
        assert control.is_dir()


class TestScaffoldBackwardCompat:
    """Verify that submissions without an experiment block produce identical output."""

    def test_no_experiment_key_uses_skill_defaults(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        treatment_df = (treatment / "environment" / "Dockerfile").read_text()
        control_df = (control / "environment" / "Dockerfile").read_text()
        assert "COPY skills/" in treatment_df
        assert "COPY skills/" not in control_df
        assert (treatment / "environment" / "skills" / "SKILL.md").is_file()
        assert not (control / "environment" / "skills").exists()

    def test_treatment_has_skills_dir_in_toml(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        assert 'skills_dir = "/skills"' in (treatment / "task.toml").read_text()
        assert "skills_dir" not in (control / "task.toml").read_text()


class TestScaffoldExperimentConfig:
    """Test scaffolding with explicit experiment configurations."""

    def test_model_experiment_no_skills_copied(self, valid_submission: Path, tmp_path: Path):
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["experiment"] = {
            "type": "model",
            "n_trials": 5,
            "treatment": {"env_from_secrets": {"MODEL": "models/a"}},
            "control": {"env_from_secrets": {"MODEL": "models/b"}},
        }
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        assert not (treatment / "environment" / "skills").exists()
        assert not (control / "environment" / "skills").exists()

    def test_model_experiment_no_skills_dir_in_toml(self, valid_submission: Path, tmp_path: Path):
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["experiment"] = {
            "type": "model",
            "treatment": {"env_from_secrets": {"MODEL": "models/a"}},
            "control": {"env_from_secrets": {"MODEL": "models/b"}},
        }
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        assert "skills_dir" not in (treatment / "task.toml").read_text()
        assert "skills_dir" not in (control / "task.toml").read_text()

    def test_custom_experiment_with_skills(self, valid_submission: Path, tmp_path: Path):
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["experiment"] = {
            "type": "custom",
            "treatment": {"copy": [{"src": "skills", "dest": "/skills"}]},
            "control": {},
        }
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        assert (treatment / "environment" / "skills" / "SKILL.md").is_file()
        assert not (control / "environment" / "skills").exists()

    def test_n_trials_accessible_from_metadata(self, valid_submission: Path, tmp_path: Path):
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["experiment"] = {"n_trials": 42}
        meta_path.write_text(yaml.dump(meta))

        from abevalflow.schemas import SubmissionMetadata
        parsed = SubmissionMetadata(**yaml.safe_load(meta_path.read_text()))
        assert parsed.experiment.n_trials == 42

    def test_prompt_experiment_identical_to_skill(self, valid_submission: Path, tmp_path: Path):
        """Prompt experiments use the same scaffold behavior as skill experiments."""
        output_skill = tmp_path / "output-skill"
        scaffold_submission(valid_submission, output_skill, TEMPLATES_DIR)

        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["experiment"] = {"type": "prompt"}
        meta_path.write_text(yaml.dump(meta))

        output_prompt = tmp_path / "output-prompt"
        scaffold_submission(valid_submission, output_prompt, TEMPLATES_DIR)

        skill_df = (output_skill / "tasks-treatment" / "my-skill" / "environment" / "Dockerfile").read_text()
        prompt_df = (output_prompt / "tasks-treatment" / "my-skill" / "environment" / "Dockerfile").read_text()
        assert skill_df == prompt_df

        skill_ctrl = (output_skill / "tasks-control" / "my-skill" / "environment" / "Dockerfile").read_text()
        prompt_ctrl = (output_prompt / "tasks-control" / "my-skill" / "environment" / "Dockerfile").read_text()
        assert skill_ctrl == prompt_ctrl

    def test_missing_configured_dir_not_copied(
        self, valid_submission: Path, tmp_path: Path,
    ):
        """Configured copy dirs that don't exist in submission must not be copied."""
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["experiment"] = {
            "type": "custom",
            "treatment": {
                "copy": [
                    {"src": "skills", "dest": "/skills"},
                    {"src": "nonexistent", "dest": "/nonexistent"},
                ],
            },
            "control": {},
        }
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        treatment, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)

        assert (treatment / "environment" / "skills" / "SKILL.md").is_file()
        assert not (treatment / "environment" / "nonexistent").exists()

    def test_control_with_skills_treatment_without(
        self, valid_submission: Path, tmp_path: Path,
    ):
        """Asymmetric config: control gets skills, treatment does not."""
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["experiment"] = {
            "type": "custom",
            "treatment": {},
            "control": {"copy": [{"src": "skills", "dest": "/skills"}]},
        }
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        treatment, control = scaffold_submission(valid_submission, output, TEMPLATES_DIR)

        assert not (treatment / "environment" / "skills").exists()
        assert "skills_dir" not in (treatment / "task.toml").read_text()

        assert (control / "environment" / "skills" / "SKILL.md").is_file()
        assert 'skills_dir = "/skills"' in (control / "task.toml").read_text()
