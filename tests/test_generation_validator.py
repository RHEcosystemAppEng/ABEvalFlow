"""Tests for abevalflow/generation_validator.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from abevalflow.generation_validator import content_check, structural_check


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
    def test_invalid_json_returns_fail(self, mock_chat: MagicMock, submission: Path) -> None:
        mock_chat.return_value = "This is not JSON"
        result = content_check(submission)
        assert result["passed"] is False
        assert any("invalid JSON" in i for i in result["issues"])

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
