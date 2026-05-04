"""Tests for abevalflow/generation_validator.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from abevalflow.generation_validator import (
    content_check,
    final_review,
    multi_reviewer_check,
    pytest_collect_check,
    structural_check,
)


@pytest.fixture()
def submission(tmp_path: Path) -> Path:
    sub = tmp_path / "submission"
    sub.mkdir()
    (sub / "skills").mkdir()
    (sub / "skills" / "SKILL.md").write_text("# Skill\nDo something.\n")
    (sub / "instruction.md").write_text("# Task\nBuild a module.\n")
    (sub / "tests").mkdir()
    (sub / "tests" / "test_outputs.py").write_text("def test_ok(): assert True\n")
    return sub


class TestStructuralCheck:
    def test_valid_submission_passes(self, submission: Path) -> None:
        errors = structural_check(submission)
        assert errors == []

    def test_missing_instruction(self, submission: Path) -> None:
        (submission / "instruction.md").unlink()
        errors = structural_check(submission)
        assert any("instruction.md is missing" in e for e in errors)

    def test_empty_instruction(self, submission: Path) -> None:
        (submission / "instruction.md").write_text("")
        errors = structural_check(submission)
        assert any("instruction.md is empty" in e for e in errors)

    def test_missing_test_outputs(self, submission: Path) -> None:
        (submission / "tests" / "test_outputs.py").unlink()
        errors = structural_check(submission)
        assert any("test_outputs.py is missing" in e for e in errors)

    def test_empty_test_outputs(self, submission: Path) -> None:
        (submission / "tests" / "test_outputs.py").write_text("")
        errors = structural_check(submission)
        assert any("test_outputs.py is empty" in e for e in errors)

    def test_syntax_error_in_test_outputs(self, submission: Path) -> None:
        (submission / "tests" / "test_outputs.py").write_text("def bad(\n")
        errors = structural_check(submission)
        assert any("SyntaxError" in e for e in errors)

    def test_valid_test_outputs_compiles(self, submission: Path) -> None:
        (submission / "tests" / "test_outputs.py").write_text(
            "import sys\ndef test_it(): assert sys.version_info >= (3, 9)\n"
        )
        errors = structural_check(submission)
        assert errors == []

    def test_llm_judge_syntax_error(self, submission: Path) -> None:
        (submission / "tests" / "llm_judge.py").write_text("def bad(\n")
        errors = structural_check(submission)
        assert any("llm_judge.py" in e and "SyntaxError" in e for e in errors)

    def test_llm_judge_valid_ignored(self, submission: Path) -> None:
        (submission / "tests" / "llm_judge.py").write_text("score = 0.9\n")
        errors = structural_check(submission)
        assert errors == []

    def test_multiple_errors_collected(self, submission: Path) -> None:
        (submission / "instruction.md").unlink()
        (submission / "tests" / "test_outputs.py").write_text("def bad(\n")
        errors = structural_check(submission)
        assert len(errors) >= 2


class TestContentCheck:
    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_pass_response(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({"pass": True, "issues": []})
        result = content_check(submission)
        assert result["passed"] is True
        assert result["issues"] == []

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_fail_response(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({
            "pass": False,
            "issues": ["instruction does not match skill"],
        })
        result = content_check(submission)
        assert result["passed"] is False
        assert "instruction does not match skill" in result["issues"]

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_json_in_markdown_fence_extracted(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = 'Sure, here is the result:\n{"pass": true, "issues": []}\nDone.'
        result = content_check(submission)
        assert result["passed"] is True

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_issues_as_string_normalised(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({"pass": False, "issues": "single issue"})
        result = content_check(submission)
        assert result["passed"] is False
        assert result["issues"] == ["single issue"]

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_prompt_includes_all_files(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({"pass": True, "issues": []})
        content_check(submission)
        call_args = mock_chat.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "SKILL.md" in user_msg
        assert "instruction.md" in user_msg
        assert "test_outputs.py" in user_msg


class TestPytestCollectCheck:
    def test_valid_tests_pass(self, submission: Path) -> None:
        errors = pytest_collect_check(submission)
        assert errors == []

    def test_missing_file_returns_error(self, submission: Path) -> None:
        (submission / "tests" / "test_outputs.py").unlink()
        errors = pytest_collect_check(submission)
        assert any("missing" in e for e in errors)

    def test_syntax_error_returns_error(self, submission: Path) -> None:
        (submission / "tests" / "test_outputs.py").write_text("def bad(\n")
        errors = pytest_collect_check(submission)
        assert len(errors) == 1
        assert "pytest --collect-only failed" in errors[0]

    def test_no_tests_collected_returns_error(self, submission: Path) -> None:
        (submission / "tests" / "test_outputs.py").write_text("x = 1\n")
        errors = pytest_collect_check(submission)
        assert len(errors) == 1
        assert "pytest --collect-only failed" in errors[0]

    @patch("abevalflow.generation_validator.subprocess.run")
    def test_timeout_returns_error(self, mock_run: MagicMock, submission: Path) -> None:
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=10)
        errors = pytest_collect_check(submission)
        assert any("timed out" in e for e in errors)

    @patch("abevalflow.generation_validator.subprocess.run")
    def test_pytest_not_found_skips(self, mock_run: MagicMock, submission: Path) -> None:
        mock_run.side_effect = FileNotFoundError()
        errors = pytest_collect_check(submission)
        assert errors == []


class TestMultiReviewerCheck:
    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_all_pass(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({"pass": True, "issues": []})
        result = multi_reviewer_check(submission)
        assert result["passed"] is True
        assert result["issues"] == []
        assert len(result["reviewer_results"]) == 3
        assert mock_chat.call_count == 3

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_one_fails(self, mock_chat: MagicMock, submission: Path) -> None:
        responses = [
            json.dumps({"pass": True, "issues": []}),
            json.dumps({"pass": False, "issues": ["tests miss edge case"]}),
            json.dumps({"pass": True, "issues": []}),
        ]
        mock_chat.side_effect = responses
        result = multi_reviewer_check(submission)
        assert result["passed"] is False
        assert len(result["issues"]) == 1
        assert "tests miss edge case" in result["issues"][0]

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_all_fail_collects_all_issues(self, mock_chat: MagicMock, submission: Path) -> None:
        responses = [
            json.dumps({"pass": False, "issues": ["issue A"]}),
            json.dumps({"pass": False, "issues": ["issue B"]}),
            json.dumps({"pass": False, "issues": ["issue C"]}),
        ]
        mock_chat.side_effect = responses
        result = multi_reviewer_check(submission)
        assert result["passed"] is False
        assert len(result["issues"]) == 3

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_reviewer_names_in_issues(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({"pass": False, "issues": ["bad"]})
        result = multi_reviewer_check(submission)
        assert result["passed"] is False
        reviewer_names = {"coverage", "alignment", "feasibility"}
        found_names = set()
        for issue in result["issues"]:
            for name in reviewer_names:
                if f"[{name}]" in issue:
                    found_names.add(name)
        assert found_names == reviewer_names

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_issues_as_string_normalised(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({"pass": False, "issues": "single issue"})
        result = multi_reviewer_check(submission)
        assert result["passed"] is False
        assert any("single issue" in i for i in result["issues"])


class TestFinalReview:
    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_pass_response(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({"pass": True, "issues": []})
        result = final_review(submission)
        assert result["passed"] is True
        assert result["issues"] == []

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_fail_response(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({
            "pass": False,
            "issues": ["tests are not deterministic"],
        })
        result = final_review(submission)
        assert result["passed"] is False
        assert "tests are not deterministic" in result["issues"]

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_json_in_wrapper_extracted(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = 'Here is my review:\n{"pass": true, "issues": []}\nEnd.'
        result = final_review(submission)
        assert result["passed"] is True

    @patch("abevalflow.generation_validator.llm_client.chat_completion")
    def test_uses_temperature_zero(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = json.dumps({"pass": True, "issues": []})
        final_review(submission)
        call_kwargs = mock_chat.call_args.kwargs if mock_chat.call_args.kwargs else {}
        if "temperature" in call_kwargs:
            assert call_kwargs["temperature"] == 0.0
