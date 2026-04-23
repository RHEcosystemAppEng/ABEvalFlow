"""Tests for scripts/generate_tests.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.generate_tests import generate, main

AI_METADATA = {
    "name": "ai-skill",
    "description": "A skill that needs AI-generated tests",
    "generation_mode": "ai",
}

MANUAL_METADATA = {
    "name": "manual-skill",
    "generation_mode": "manual",
}

SKILL_CONTENT = """\
# Code Review Skill

When asked to review code, follow these guidelines:
- Check for common bugs and anti-patterns
- Suggest improvements for readability
- Verify error handling
"""

MOCK_LLM_RESPONSE = json.dumps({
    "instruction_md": (
        "# Code Review Task\n\n"
        "You are given a Python file. Review it for bugs and suggest fixes.\n\n"
        "## Requirements\n"
        "- Create a file `review.py` with a function `review(code: str) -> list[str]`\n"
        "- Return a list of issues found\n"
    ),
    "test_outputs_py": (
        '"""Tests for code review task."""\n\n'
        "import importlib\nimport sys\nfrom pathlib import Path\n\n\n"
        "def _load_review():\n"
        '    sys.path.insert(0, "/workspace")\n'
        '    return importlib.import_module("review")\n\n\n'
        "def test_review_returns_list():\n"
        "    mod = _load_review()\n"
        '    result = mod.review("x = 1")\n'
        "    assert isinstance(result, list)\n"
    ),
})


@pytest.fixture()
def ai_submission(tmp_path: Path) -> Path:
    sub = tmp_path / "submissions" / "ai-skill"
    sub.mkdir(parents=True)
    (sub / "metadata.yaml").write_text(yaml.dump(AI_METADATA))
    (sub / "skills").mkdir()
    (sub / "skills" / "SKILL.md").write_text(SKILL_CONTENT)
    return sub


@pytest.fixture()
def manual_submission(tmp_path: Path) -> Path:
    sub = tmp_path / "submissions" / "manual-skill"
    sub.mkdir(parents=True)
    (sub / "metadata.yaml").write_text(yaml.dump(MANUAL_METADATA))
    (sub / "skills").mkdir()
    (sub / "skills" / "SKILL.md").write_text(SKILL_CONTENT)
    (sub / "instruction.md").write_text("Do something.\n")
    (sub / "tests").mkdir()
    (sub / "tests" / "test_outputs.py").write_text("def test_ok(): assert True\n")
    return sub


class TestGenerate:
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_ai_mode_generates_files(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.return_value = MOCK_LLM_RESPONSE

        generated = generate(ai_submission, tmp_path, agent_type="api")

        assert "instruction.md" in generated
        assert "tests/test_outputs.py" in generated
        assert (ai_submission / "instruction.md").is_file()
        assert (ai_submission / "tests" / "test_outputs.py").is_file()
        assert "Code Review Task" in (ai_submission / "instruction.md").read_text()

    def test_manual_mode_skips_generation(
        self,
        manual_submission: Path,
        tmp_path: Path,
    ) -> None:
        generated = generate(manual_submission, tmp_path, agent_type="api")
        assert generated == []

    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_empty_skill_raises(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        tmp_path: Path,
    ) -> None:
        sub = tmp_path / "submissions" / "empty-skill"
        sub.mkdir(parents=True)
        (sub / "metadata.yaml").write_text(yaml.dump(AI_METADATA))
        (sub / "skills").mkdir()
        (sub / "skills" / "SKILL.md").write_text("")

        with pytest.raises(ValueError, match="empty"):
            generate(sub, tmp_path, agent_type="api")

    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_invalid_llm_response_raises(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.return_value = "This is not JSON at all"

        with pytest.raises(ValueError, match="not valid JSON"):
            generate(ai_submission, tmp_path, agent_type="api")

    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_missing_instruction_in_response_raises(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.return_value = json.dumps({"test_outputs_py": "def test(): pass"})

        with pytest.raises(ValueError, match="instruction_md"):
            generate(ai_submission, tmp_path, agent_type="api")

    @patch("scripts.generate_tests.skill_loader.fetch_skill")
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_quality_criteria_included_when_skill_fetched(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        skill_cache = tmp_path / "_skill_cache" / "agentic-contribution-skill"
        skill_cache.mkdir(parents=True)
        (skill_cache / "SKILL.md").write_text(
            "---\nname: test\n---\n\n## Workflow\n\nDo good work.\n"
        )
        mock_fetch.return_value = skill_cache / "SKILL.md"
        mock_chat.return_value = MOCK_LLM_RESPONSE

        generate(ai_submission, tmp_path, agent_type="api")

        call_args = mock_chat.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "Quality Criteria" in system_msg

    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_optional_llm_judge(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        response_with_judge = json.dumps({
            "instruction_md": "# Task\nDo something.\n",
            "test_outputs_py": "def test_ok(): assert True\n",
            "llm_judge_py": "score = 0.9\n",
        })
        mock_chat.return_value = response_with_judge

        generated = generate(ai_submission, tmp_path, agent_type="api")

        assert "tests/llm_judge.py" in generated
        assert (ai_submission / "tests" / "llm_judge.py").is_file()


class TestMain:
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_success_returns_zero(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        ai_submission: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_chat.return_value = MOCK_LLM_RESPONSE
        rc = main([str(ai_submission)])
        assert rc == 0
        output = json.loads(capsys.readouterr().out)
        assert "instruction.md" in output["generated"]

    def test_nonexistent_dir_returns_one(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main([str(tmp_path / "nope")])
        assert rc == 1
        output = json.loads(capsys.readouterr().out)
        assert output["generated"] == []

    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_llm_error_returns_one(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        ai_submission: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_chat.side_effect = RuntimeError("API unreachable")
        rc = main([str(ai_submission)])
        assert rc == 1
        output = json.loads(capsys.readouterr().out)
        assert "error" in output
