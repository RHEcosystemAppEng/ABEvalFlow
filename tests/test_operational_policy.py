"""Tests for operational policy compliance checks."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from abevalflow.certification import CheckId
from abevalflow.operational_policy import (
    MAX_AGENT_TIMEOUT_SEC,
    MAX_CPUS,
    MAX_MEMORY_MB,
    _check_error_handling,
    _check_logging_suppression,
    _check_resource_limits,
    _check_timeout_compliance,
    check_operational_policy,
)
from abevalflow.schemas import SubmissionMetadata


def _make_metadata(**overrides) -> SubmissionMetadata:
    base = {"name": "test-submission"}
    base.update(overrides)
    return SubmissionMetadata(**base)


def _write_submission(tmp_path: Path, metadata: dict | None = None, skill_content: str = "", test_content: str = ""):
    if metadata is not None:
        (tmp_path / "metadata.yaml").write_text(yaml.dump(metadata))
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir(exist_ok=True)
    if skill_content:
        (skills_dir / "SKILL.md").write_text(skill_content)
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(exist_ok=True)
    if test_content:
        (tests_dir / "test_outputs.py").write_text(test_content)


class TestResourceLimits:
    def test_default_values_pass(self):
        metadata = _make_metadata()
        issues = _check_resource_limits(metadata)
        assert issues == []

    def test_within_limits_pass(self):
        metadata = _make_metadata(cpus=MAX_CPUS, memory_mb=MAX_MEMORY_MB)
        issues = _check_resource_limits(metadata)
        assert issues == []

    def test_cpu_exceeds_limit(self):
        metadata = _make_metadata(cpus=MAX_CPUS + 1)
        issues = _check_resource_limits(metadata)
        assert len(issues) == 1
        assert "CPU" in issues[0]

    def test_memory_exceeds_limit(self):
        metadata = _make_metadata(memory_mb=MAX_MEMORY_MB + 1)
        issues = _check_resource_limits(metadata)
        assert len(issues) == 1
        assert "Memory" in issues[0]

    def test_both_exceed(self):
        metadata = _make_metadata(cpus=MAX_CPUS + 1, memory_mb=MAX_MEMORY_MB + 1)
        issues = _check_resource_limits(metadata)
        assert len(issues) == 2


class TestTimeoutCompliance:
    def test_default_timeout_passes(self):
        metadata = _make_metadata()
        issues = _check_timeout_compliance(metadata)
        assert issues == []

    def test_within_limit_passes(self):
        metadata = _make_metadata(agent_timeout_sec=MAX_AGENT_TIMEOUT_SEC)
        issues = _check_timeout_compliance(metadata)
        assert issues == []

    def test_exceeds_limit(self):
        metadata = _make_metadata(agent_timeout_sec=MAX_AGENT_TIMEOUT_SEC + 1)
        issues = _check_timeout_compliance(metadata)
        assert len(issues) == 1
        assert "timeout" in issues[0].lower()


class TestErrorHandling:
    def test_no_test_file(self, tmp_path: Path):
        issues = _check_error_handling(tmp_path / "nonexistent.py")
        assert issues == []

    def test_clean_test_file(self, tmp_path: Path):
        test_file = tmp_path / "test_outputs.py"
        test_file.write_text(
            "def test_something():\n"
            "    try:\n"
            "        result = run()\n"
            "    except ValueError as e:\n"
            "        raise AssertionError(f'Failed: {e}')\n"
        )
        issues = _check_error_handling(test_file)
        assert issues == []

    def test_bare_except_pass(self, tmp_path: Path):
        test_file = tmp_path / "test_outputs.py"
        test_file.write_text(
            "def test_something():\n"
            "    try:\n"
            "        result = run()\n"
            "    except:\n"
            "        pass\n"
        )
        issues = _check_error_handling(test_file)
        assert len(issues) == 1
        assert "Bare" in issues[0]

    def test_except_exception_pass(self, tmp_path: Path):
        test_file = tmp_path / "test_outputs.py"
        test_file.write_text(
            "def test_something():\n"
            "    try:\n"
            "        result = run()\n"
            "    except Exception:\n"
            "        pass\n"
        )
        issues = _check_error_handling(test_file)
        assert len(issues) == 1
        assert "Exception" in issues[0]

    def test_except_with_body_is_ok(self, tmp_path: Path):
        test_file = tmp_path / "test_outputs.py"
        test_file.write_text(
            "def test_something():\n"
            "    try:\n"
            "        result = run()\n"
            "    except Exception:\n"
            "        print('error')\n"
            "        raise\n"
        )
        issues = _check_error_handling(test_file)
        assert issues == []

    def test_except_ellipsis(self, tmp_path: Path):
        test_file = tmp_path / "test_outputs.py"
        test_file.write_text(
            "def test_something():\n"
            "    try:\n"
            "        result = run()\n"
            "    except:\n"
            "        ...\n"
        )
        issues = _check_error_handling(test_file)
        assert len(issues) == 1

    def test_syntax_error_is_ignored(self, tmp_path: Path):
        test_file = tmp_path / "test_outputs.py"
        test_file.write_text("def broken(:\n")
        issues = _check_error_handling(test_file)
        assert issues == []


class TestLoggingSuppression:
    def test_no_skill_file(self, tmp_path: Path):
        issues = _check_logging_suppression(tmp_path / "nonexistent.md")
        assert issues == []

    def test_clean_skill(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nThis skill helps with code review.\n")
        issues = _check_logging_suppression(skill_file)
        assert issues == []

    def test_do_not_log(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nDo not log any output.\n")
        issues = _check_logging_suppression(skill_file)
        assert len(issues) == 1

    def test_disable_logging(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nPlease disable logging for this task.\n")
        issues = _check_logging_suppression(skill_file)
        assert len(issues) == 1

    def test_suppress_output(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nSuppress output from the agent.\n")
        issues = _check_logging_suppression(skill_file)
        assert len(issues) == 1

    def test_case_insensitive(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nDISABLE LOGGING immediately.\n")
        issues = _check_logging_suppression(skill_file)
        assert len(issues) == 1


class TestCheckOperationalPolicy:
    def test_all_pass(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "good-skill"},
            skill_content="# Good Skill\n\nHelps with testing.\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path)
        assert result.check_id == CheckId.OPERATIONAL_POLICY_COMPLIANCE
        assert result.passed is True
        assert result.score == 1.0

    def test_resource_limit_failure(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "heavy-skill", "cpus": MAX_CPUS + 1},
            skill_content="# Skill\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path)
        assert result.passed is False
        assert "CPU" in result.message

    def test_error_handling_failure(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "bad-tests"},
            skill_content="# Skill\n",
            test_content="def test_bad():\n    try:\n        run()\n    except:\n        pass\n",
        )
        result = check_operational_policy(tmp_path)
        assert result.passed is False
        assert "except" in result.message.lower() or "Bare" in result.message

    def test_logging_suppression_failure(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "noisy-skill"},
            skill_content="# Skill\n\nDo not log any output.\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path)
        assert result.passed is False
        assert "log" in result.message.lower()

    def test_no_metadata_still_checks_files(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata=None,
            skill_content="# Skill\n\nDisable logging please.\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path)
        assert result.passed is False

    def test_empty_submission(self, tmp_path: Path):
        result = check_operational_policy(tmp_path)
        assert result.passed is True
        assert result.score == 1.0

    def test_score_degrades_with_issues(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "bad", "cpus": MAX_CPUS + 1, "memory_mb": MAX_MEMORY_MB + 1},
            skill_content="# Skill\n\nDo not log output.\n",
            test_content="def test_bad():\n    try:\n        x()\n    except:\n        pass\n",
        )
        result = check_operational_policy(tmp_path)
        assert result.passed is False
        assert result.score < 1.0
