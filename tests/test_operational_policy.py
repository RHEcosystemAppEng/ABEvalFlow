"""Tests for operational policy compliance checks."""

from __future__ import annotations

from pathlib import Path

import yaml

from abevalflow.certification import CheckId
from abevalflow.operational_policy import (
    _check_error_handling,
    _check_logging_suppression,
    _check_resource_declaration,
    _check_resource_limits,
    _check_timeout_compliance,
    check_operational_policy,
)
from abevalflow.schemas import OperationalLimits, SubmissionMetadata

_DEFAULTS = OperationalLimits(enabled=True)
MAX_CPUS = _DEFAULTS.max_cpus
MAX_MEMORY_MB = _DEFAULTS.max_memory_mb
MAX_AGENT_TIMEOUT_SEC = _DEFAULTS.max_agent_timeout_sec
ENABLED_LIMITS = OperationalLimits(enabled=True)


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


class TestResourceDeclaration:
    def test_all_declared(self):
        data = {"name": "test", "cpus": 2, "memory_mb": 4096, "agent_timeout_sec": 600}
        issues = _check_resource_declaration(data)
        assert issues == []

    def test_none_declared(self):
        data = {"name": "test"}
        issues = _check_resource_declaration(data)
        assert len(issues) == 1
        assert "cpus" in issues[0]
        assert "memory_mb" in issues[0]
        assert "agent_timeout_sec" in issues[0]

    def test_partial_declared(self):
        data = {"name": "test", "cpus": 2}
        issues = _check_resource_declaration(data)
        assert len(issues) == 1
        assert "memory_mb" in issues[0]
        assert "agent_timeout_sec" in issues[0]
        assert "cpus" not in issues[0]


class TestResourceLimits:
    def test_default_values_pass(self):
        metadata = _make_metadata()
        issues = _check_resource_limits(metadata, OperationalLimits())
        assert issues == []

    def test_within_limits_pass(self):
        metadata = _make_metadata(cpus=MAX_CPUS, memory_mb=MAX_MEMORY_MB)
        issues = _check_resource_limits(metadata, OperationalLimits())
        assert issues == []

    def test_cpu_exceeds_limit(self):
        metadata = _make_metadata(cpus=MAX_CPUS + 1)
        issues = _check_resource_limits(metadata, OperationalLimits())
        assert len(issues) == 1
        assert "CPU" in issues[0]

    def test_memory_exceeds_limit(self):
        metadata = _make_metadata(memory_mb=MAX_MEMORY_MB + 1)
        issues = _check_resource_limits(metadata, OperationalLimits())
        assert len(issues) == 1
        assert "Memory" in issues[0]

    def test_both_exceed(self):
        metadata = _make_metadata(cpus=MAX_CPUS + 1, memory_mb=MAX_MEMORY_MB + 1)
        issues = _check_resource_limits(metadata, OperationalLimits())
        assert len(issues) == 2


class TestTimeoutCompliance:
    def test_default_timeout_passes(self):
        metadata = _make_metadata()
        issues = _check_timeout_compliance(metadata, OperationalLimits())
        assert issues == []

    def test_within_limit_passes(self):
        metadata = _make_metadata(agent_timeout_sec=MAX_AGENT_TIMEOUT_SEC)
        issues = _check_timeout_compliance(metadata, OperationalLimits())
        assert issues == []

    def test_exceeds_limit(self):
        metadata = _make_metadata(agent_timeout_sec=MAX_AGENT_TIMEOUT_SEC + 1)
        issues = _check_timeout_compliance(metadata, OperationalLimits())
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
        test_file.write_text("def test_something():\n    try:\n        result = run()\n    except:\n        pass\n")
        issues = _check_error_handling(test_file)
        assert len(issues) == 1
        assert "Bare" in issues[0]

    def test_except_exception_pass(self, tmp_path: Path):
        test_file = tmp_path / "test_outputs.py"
        test_file.write_text(
            "def test_something():\n    try:\n        result = run()\n    except Exception:\n        pass\n"
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
        test_file.write_text("def test_something():\n    try:\n        result = run()\n    except:\n        ...\n")
        issues = _check_error_handling(test_file)
        assert len(issues) == 1

    def test_except_base_exception_pass(self, tmp_path: Path):
        test_file = tmp_path / "test_outputs.py"
        test_file.write_text(
            "def test_something():\n    try:\n        run()\n    except BaseException:\n        pass\n"
        )
        issues = _check_error_handling(test_file)
        assert len(issues) == 1
        assert "BaseException" in issues[0]

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

    def test_security_qualified_not_flagged(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nDo not log passwords or PII.\n")
        issues = _check_logging_suppression(skill_file)
        assert issues == []

    def test_security_qualified_credentials(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nNo logging of credentials.\n")
        issues = _check_logging_suppression(skill_file)
        assert issues == []

    def test_security_qualified_tokens(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nDo not log sensitive tokens.\n")
        issues = _check_logging_suppression(skill_file)
        assert issues == []

    def test_inside_code_block_not_flagged(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\n```\ndo not log anything\n```\n")
        issues = _check_logging_suppression(skill_file)
        assert issues == []

    def test_inside_blockquote_not_flagged(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\n> do not log anything\n")
        issues = _check_logging_suppression(skill_file)
        assert issues == []

    def test_outside_code_block_still_flagged(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\n```\nexample\n```\n\nDo not log output.\n")
        issues = _check_logging_suppression(skill_file)
        assert len(issues) == 1

    def test_security_qualified_api_keys(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nDo not log api_keys.\n")
        issues = _check_logging_suppression(skill_file)
        assert issues == []

    def test_security_qualified_bearer(self, tmp_path: Path):
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text("# My Skill\n\nDo not log bearer tokens.\n")
        issues = _check_logging_suppression(skill_file)
        assert issues == []


class TestCheckOperationalPolicy:
    def test_all_pass(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "good-skill", "cpus": 2, "memory_mb": 4096, "agent_timeout_sec": 600},
            skill_content="# Good Skill\n\nHelps with testing.\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
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
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is False
        assert "CPU" in result.message

    def test_error_handling_failure(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "bad-tests"},
            skill_content="# Skill\n",
            test_content="def test_bad():\n    try:\n        run()\n    except:\n        pass\n",
        )
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is False
        assert "except" in result.message.lower() or "Bare" in result.message

    def test_error_handling_checks_all_test_files(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "multi-tests"},
            skill_content="# Skill\n",
            test_content="def test_ok():\n    assert True\n",
        )
        other_test = tmp_path / "tests" / "llm_judge.py"
        other_test.write_text("def judge():\n    try:\n        run()\n    except:\n        pass\n")
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is False
        assert "Bare" in result.message

    def test_logging_suppression_low_severity_still_passes(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "noisy-skill", "cpus": 2, "memory_mb": 4096, "agent_timeout_sec": 600},
            skill_content="# Skill\n\nDo not log any output.\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is True
        assert result.score == 0.85
        assert "[low]" in result.message

    def test_root_skill_md_checked(self, tmp_path: Path):
        (tmp_path / "metadata.yaml").write_text("name: flat-skill\ncpus: 2\nmemory_mb: 4096\nagent_timeout_sec: 600\n")
        (tmp_path / "SKILL.md").write_text("# Skill\n\nDo not log any output.\n")
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is True
        assert "[low]" in result.message

    def test_no_metadata_low_severity_still_passes(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata=None,
            skill_content="# Skill\n\nDisable logging please.\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is True
        assert result.score < 1.0

    def test_empty_submission(self, tmp_path: Path):
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is True
        assert result.score == 1.0

    def test_score_severity_weighting(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "bad", "cpus": MAX_CPUS + 1, "memory_mb": MAX_MEMORY_MB + 1},
            skill_content="# Skill\n\nDo not log output.\n",
            test_content="def test_bad():\n    try:\n        x()\n    except:\n        pass\n",
        )
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is False
        assert result.score < 1.0
        assert result.score == max(0.0, 1.0 - (0.15 + 0.35 + 0.35 + 0.35 + 0.15))

    def test_low_severity_only_scores_differently(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "ok-skill", "cpus": 2, "memory_mb": 4096, "agent_timeout_sec": 600},
            skill_content="# Skill\n\nDo not log any output.\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is True
        assert result.score == 0.85

    def test_high_severity_blocks_passed(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "bad", "cpus": MAX_CPUS + 1, "memory_mb": 4096, "agent_timeout_sec": 600},
            skill_content="# Skill\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.passed is False
        assert result.score == 0.65

    def test_severity_shown_in_message(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "bad", "cpus": MAX_CPUS + 1},
            skill_content="# Skill\n\nDo not log output.\n",
            test_content="def test_ok():\n    assert True\n",
        )
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert "[high]" in result.message
        assert "[low]" in result.message

    def test_both_root_and_nested_skill_checked(self, tmp_path: Path):
        (tmp_path / "metadata.yaml").write_text("name: dual\ncpus: 2\nmemory_mb: 4096\nagent_timeout_sec: 600\n")
        (tmp_path / "SKILL.md").write_text("# Root\n\nDo not log any output.\n")
        skills_dir = tmp_path / "skills"
        skills_dir.mkdir()
        (skills_dir / "SKILL.md").write_text("# Nested\n\nDisable logging here.\n")
        result = check_operational_policy(tmp_path, limits=ENABLED_LIMITS)
        assert result.message.count("[low]") == 2

    def test_disabled_by_default(self, tmp_path: Path):
        _write_submission(
            tmp_path,
            metadata={"name": "bad", "cpus": MAX_CPUS + 1},
            skill_content="# Skill\n\nDo not log output.\n",
        )
        result = check_operational_policy(tmp_path)
        assert result.passed is True
        assert result.score == 1.0
        assert "disabled" in result.message.lower()

    def test_disabled_lists_skipped_checks(self, tmp_path: Path):
        result = check_operational_policy(tmp_path)
        assert "resource declarations" in result.message
        assert "resource limits" in result.message
        assert "timeout compliance" in result.message
        assert "error handling" in result.message
        assert "logging suppression" in result.message


class TestCustomLimits:
    _full_resources = {"cpus": 2, "memory_mb": 4096, "agent_timeout_sec": 600}

    def test_custom_limits_allow_higher_cpu(self, tmp_path: Path):
        meta = {"name": "big-skill", **self._full_resources, "cpus": 8}
        _write_submission(tmp_path, metadata=meta)
        result = check_operational_policy(tmp_path, limits=OperationalLimits(enabled=True, max_cpus=10))
        assert result.passed is True

    def test_custom_limits_still_enforce(self, tmp_path: Path):
        meta = {"name": "big-skill", **self._full_resources, "cpus": 8}
        _write_submission(tmp_path, metadata=meta)
        result = check_operational_policy(tmp_path, limits=OperationalLimits(enabled=True, max_cpus=4))
        assert result.passed is False

    def test_custom_memory_limit(self, tmp_path: Path):
        meta = {"name": "big-skill", **self._full_resources, "memory_mb": 16384}
        _write_submission(tmp_path, metadata=meta)
        result = check_operational_policy(tmp_path, limits=OperationalLimits(enabled=True, max_memory_mb=32768))
        assert result.passed is True

    def test_custom_timeout_limit(self, tmp_path: Path):
        meta = {"name": "big-skill", **self._full_resources, "agent_timeout_sec": 7200}
        _write_submission(tmp_path, metadata=meta)
        result = check_operational_policy(tmp_path, limits=OperationalLimits(enabled=True, max_agent_timeout_sec=10000))
        assert result.passed is True
