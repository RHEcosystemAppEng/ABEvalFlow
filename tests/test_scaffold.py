"""Tests for scripts/scaffold.py — scaffold a skill submission into Harbor task dirs."""

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

    def test_creates_skilled_and_unskilled_dirs(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, unskilled = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        assert skilled.is_dir()
        assert unskilled.is_dir()
        assert skilled.name == "my-skill"
        assert unskilled.name == "my-skill"

    def test_skilled_dir_structure(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)

        assert (skilled / "instruction.md").is_file()
        assert (skilled / "task.toml").is_file()
        assert (skilled / "test.sh").is_file()
        assert (skilled / "environment" / "Dockerfile").is_file()
        assert (skilled / "environment" / "instruction.md").is_file()
        assert (skilled / "environment" / "skills" / "SKILL.md").is_file()
        assert (skilled / "environment" / "tests" / "test_outputs.py").is_file()

    def test_unskilled_dir_structure(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        _, unskilled = scaffold_submission(valid_submission, output, TEMPLATES_DIR)

        assert (unskilled / "instruction.md").is_file()
        assert (unskilled / "task.toml").is_file()
        assert (unskilled / "test.sh").is_file()
        assert (unskilled / "environment" / "Dockerfile").is_file()
        assert (unskilled / "environment" / "tests" / "test_outputs.py").is_file()
        # Unskilled must NOT contain skills/ or docs/
        assert not (unskilled / "environment" / "skills").exists()
        assert not (unskilled / "environment" / "docs").exists()

    def test_skilled_dockerfile_contains_skills_copy(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (skilled / "environment" / "Dockerfile").read_text()
        assert "COPY skills/" in content

    def test_unskilled_dockerfile_excludes_skills(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        _, unskilled = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (unskilled / "environment" / "Dockerfile").read_text()
        assert "COPY skills/" not in content
        assert "COPY docs/" not in content

    def test_test_sh_is_executable(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, unskilled = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        for d in (skilled, unskilled):
            mode = (d / "test.sh").stat().st_mode
            assert mode & stat.S_IEXEC


class TestScaffoldTaskToml:
    """Test task.toml rendering."""

    def test_skilled_task_toml_has_skills_dir(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (skilled / "task.toml").read_text()
        assert 'skills_dir = "/skills"' in content

    def test_unskilled_task_toml_no_skills_dir(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        _, unskilled = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (unskilled / "task.toml").read_text()
        assert "skills_dir" not in content

    def test_task_toml_metadata_fields(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (skilled / "task.toml").read_text()
        assert 'category = "rh-developer"' in content
        assert '"test"' in content
        assert '"demo"' in content

    def test_task_toml_is_valid_toml_without_llm_judge(
        self, valid_submission: Path, tmp_path: Path
    ):
        output = tmp_path / "output"
        skilled, unskilled = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        for d in (skilled, unskilled):
            tomllib.loads((d / "task.toml").read_text())

    def test_task_toml_is_valid_toml_with_llm_judge(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, unskilled = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        for d in (skilled, unskilled):
            parsed = tomllib.loads((d / "task.toml").read_text())
            assert "verifier" in parsed

    def test_task_toml_llm_judge_env_key_populated(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        parsed = tomllib.loads((skilled / "task.toml").read_text())
        env = parsed["verifier"]["env"]
        assert "LLM_API_KEY" in env

    def test_task_toml_custom_timeouts(self, valid_submission: Path, tmp_path: Path):
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        meta["agent_timeout_sec"] = 1200.0
        meta["memory_mb"] = 4096
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        skilled, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        parsed = tomllib.loads((skilled / "task.toml").read_text())
        assert parsed["agent"]["timeout_sec"] == 1200.0
        assert parsed["environment"]["memory_mb"] == 4096


class TestScaffoldOptionalDirs:
    """Test scaffolding with optional directories (docs, supportive, llm_judge)."""

    def test_supportive_copied_when_present(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, unskilled = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        for d in (skilled, unskilled):
            assert (d / "environment" / "supportive" / "data.json").is_file()

    def test_docs_copied_only_for_skilled(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, unskilled = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        assert (skilled / "environment" / "docs" / "reference.md").is_file()
        assert not (unskilled / "environment" / "docs").exists()

    def test_dockerfile_includes_supportive_when_present(
        self, full_submission: Path, tmp_path: Path
    ):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        content = (skilled / "environment" / "Dockerfile").read_text()
        assert "COPY supportive/" in content

    def test_dockerfile_includes_docs_for_skilled(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        content = (skilled / "environment" / "Dockerfile").read_text()
        assert "COPY docs/" in content

    def test_no_supportive_in_dockerfile_when_absent(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (skilled / "environment" / "Dockerfile").read_text()
        assert "COPY supportive/" not in content

    def test_test_sh_includes_llm_judge_when_present(self, full_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(full_submission, output, TEMPLATES_DIR)
        content = (skilled / "test.sh").read_text()
        assert "llm_judge.py" in content

    def test_test_sh_excludes_llm_judge_when_absent(self, valid_submission: Path, tmp_path: Path):
        output = tmp_path / "output"
        skilled, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (skilled / "test.sh").read_text()
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
        skilled, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (skilled / "task.toml").read_text()
        assert "tags = []" in content

    def test_none_tags_produces_empty_list(self, valid_submission: Path, tmp_path: Path):
        meta_path = valid_submission / "metadata.yaml"
        meta = yaml.safe_load(meta_path.read_text())
        del meta["tags"]
        meta_path.write_text(yaml.dump(meta))

        output = tmp_path / "output"
        skilled, _ = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        content = (skilled / "task.toml").read_text()
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
        skilled, unskilled = scaffold_submission(sub, output, TEMPLATES_DIR)
        assert skilled.is_dir()
        assert unskilled.is_dir()
        content = (skilled / "task.toml").read_text()
        assert 'category = "general"' in content
        tomllib.loads(content)

    def test_idempotent_scaffold(self, valid_submission: Path, tmp_path: Path):
        """Running scaffold twice should overwrite without error."""
        output = tmp_path / "output"
        scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        skilled, unskilled = scaffold_submission(valid_submission, output, TEMPLATES_DIR)
        assert skilled.is_dir()
        assert unskilled.is_dir()
