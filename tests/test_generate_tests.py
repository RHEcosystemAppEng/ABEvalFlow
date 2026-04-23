"""Tests for scripts/generate_tests.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.generate_tests import DEFAULT_MAX_RETRIES, generate, main

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

MOCK_ANALYSIS = json.dumps({
    "novel_aspects": ["Check for anti-patterns specific to this project"],
    "common_knowledge": ["Basic code review practices"],
    "test_focus_areas": ["Verify anti-pattern detection covers project-specific cases"],
})

MOCK_INSTRUCTION = (
    "# Code Review Task\n\n"
    "You are given a Python file. Review it for bugs and suggest fixes.\n\n"
    "## Requirements\n"
    "- Create a file `review.py` with a function `review(code: str) -> list[str]`\n"
    "- Return a list of issues found\n"
)

MOCK_TEST = (
    '"""Tests for code review task."""\n\n'
    "import importlib\nimport sys\nfrom pathlib import Path\n\n\n"
    "def _load_review():\n"
    '    sys.path.insert(0, "/workspace")\n'
    '    return importlib.import_module("review")\n\n\n'
    "def test_review_returns_list():\n"
    "    mod = _load_review()\n"
    '    result = mod.review("x = 1")\n'
    "    assert isinstance(result, list)\n"
)

MOCK_JUDGE_SKIP = "SKIP"
MOCK_JUDGE_CODE = "score = 0.9\n"

CONTENT_PASS = {"passed": True, "issues": []}
CONTENT_FAIL = {"passed": False, "issues": ["instruction does not match skill"]}


def _four_step_responses(
    analysis: str = MOCK_ANALYSIS,
    instruction: str = MOCK_INSTRUCTION,
    test: str = MOCK_TEST,
    judge: str = MOCK_JUDGE_SKIP,
) -> list[str]:
    """Return the 4 sequential LLM responses for a single attempt."""
    return [analysis, instruction, test, judge]


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
    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_ai_mode_generates_files(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = _four_step_responses()
        generated = generate(ai_submission, tmp_path, agent_type="api")

        assert "instruction.md" in generated
        assert "tests/test_outputs.py" in generated
        assert (ai_submission / "instruction.md").is_file()
        assert (ai_submission / "tests" / "test_outputs.py").is_file()
        assert "Code Review Task" in (ai_submission / "instruction.md").read_text()
        assert mock_chat.call_count == 4

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

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_empty_instruction_retries(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Step 0 succeeds, Step 1 produces empty → retry from Step 0."""
        mock_chat.side_effect = [
            MOCK_ANALYSIS,
            "",
            *_four_step_responses(),
        ]
        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=2)
        assert "instruction.md" in generated

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_bad_python_in_tests_retries(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Step 2 produces invalid Python → retry from Step 0."""
        mock_chat.side_effect = [
            MOCK_ANALYSIS,
            MOCK_INSTRUCTION,
            "def bad(\n",
            *_four_step_responses(),
        ]
        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=2)
        assert "tests/test_outputs.py" in generated

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill")
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_quality_criteria_included_when_skill_fetched(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        skill_cache = tmp_path / "_skill_cache" / "agentic-contribution-skill"
        skill_cache.mkdir(parents=True)
        (skill_cache / "SKILL.md").write_text(
            "---\nname: test\n---\n\n## Workflow\n\nDo good work.\n"
        )
        mock_fetch.return_value = skill_cache / "SKILL.md"
        mock_chat.side_effect = _four_step_responses()

        generate(ai_submission, tmp_path, agent_type="api")

        instruction_call = mock_chat.call_args_list[1]
        messages = instruction_call.args[0] if instruction_call.args else instruction_call.kwargs["messages"]
        system_msg = messages[0]["content"]
        assert "Quality Criteria" in system_msg

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_llm_judge_included_when_produced(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = _four_step_responses(judge=MOCK_JUDGE_CODE)
        generated = generate(ai_submission, tmp_path, agent_type="api")

        assert "tests/llm_judge.py" in generated
        assert (ai_submission / "tests" / "llm_judge.py").is_file()

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_llm_judge_skipped_when_skip_returned(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = _four_step_responses(judge="SKIP")
        generated = generate(ai_submission, tmp_path, agent_type="api")

        assert "tests/llm_judge.py" not in generated

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_bad_judge_silently_skipped(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = _four_step_responses(judge="def bad(\n")
        generated = generate(ai_submission, tmp_path, agent_type="api")

        assert "tests/llm_judge.py" not in generated

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_markdown_fence_stripped(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        fenced_instruction = "```markdown\n# Task\nDo something.\n```"
        fenced_test = "```python\ndef test_ok(): assert True\n```"
        mock_chat.side_effect = [MOCK_ANALYSIS, fenced_instruction, fenced_test, "SKIP"]
        generated = generate(ai_submission, tmp_path, agent_type="api")
        assert "instruction.md" in generated
        assert "tests/test_outputs.py" in generated


class TestSkillAnalysis:
    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_analysis_passed_to_instruction_prompt(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = _four_step_responses()
        generate(ai_submission, tmp_path, agent_type="api")

        instruction_call = mock_chat.call_args_list[1]
        messages = instruction_call.args[0] if instruction_call.args else instruction_call.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Novel aspects" in user_msg
        assert "anti-patterns specific to this project" in user_msg

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_analysis_passed_to_test_prompt(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = _four_step_responses()
        generate(ai_submission, tmp_path, agent_type="api")

        test_call = mock_chat.call_args_list[2]
        messages = test_call.args[0] if test_call.args else test_call.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "Test focus areas" in user_msg
        assert "anti-pattern detection" in user_msg

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_invalid_analysis_json_triggers_retry(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Bad analysis JSON on attempt 1, full success on attempt 2."""
        mock_chat.side_effect = [
            "not valid json at all",
            *_four_step_responses(),
        ]
        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=2)
        assert "instruction.md" in generated

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_analysis_missing_key_triggers_retry(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Analysis missing test_focus_areas key → retry."""
        bad_analysis = json.dumps({"novel_aspects": ["x"], "common_knowledge": ["y"]})
        mock_chat.side_effect = [
            bad_analysis,
            *_four_step_responses(),
        ]
        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=2)
        assert "instruction.md" in generated

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_four_calls_per_attempt(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Each attempt makes exactly 4 LLM calls (analysis, instruction, tests, judge)."""
        mock_chat.side_effect = _four_step_responses()
        generate(ai_submission, tmp_path, agent_type="api", max_retries=1)
        assert mock_chat.call_count == 4


class TestRetryLoop:
    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check")
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_structural_fail_then_pass_retries(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = [*_four_step_responses(), *_four_step_responses()]
        mock_struct.side_effect = [["some leftover issue"], []]

        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=3)
        assert "instruction.md" in generated
        assert mock_chat.call_count == 8

    @patch("scripts.generate_tests.content_check")
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_content_fail_then_pass_retries(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = [*_four_step_responses(), *_four_step_responses()]
        mock_content.side_effect = [CONTENT_FAIL, CONTENT_PASS]

        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=3)
        assert "instruction.md" in generated
        assert mock_chat.call_count == 8

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_FAIL)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_all_retries_exhausted_raises(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = [*_four_step_responses(), *_four_step_responses()]

        with pytest.raises(ValueError, match="content review after 2 attempts"):
            generate(ai_submission, tmp_path, agent_type="api", max_retries=2)

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_feedback_injected_on_retry(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """On retry, previous errors appear in the instruction prompt.

        Attempt 1: analysis(0), instruction(1)="" → ValueError.
        Attempt 2: analysis(2), instruction(3) ← should contain feedback.
        """
        mock_chat.side_effect = [
            MOCK_ANALYSIS,
            "",
            *_four_step_responses(),
        ]
        generate(ai_submission, tmp_path, agent_type="api", max_retries=2)

        retry_instruction_call = mock_chat.call_args_list[3]
        messages = (
            retry_instruction_call.args[0]
            if retry_instruction_call.args
            else retry_instruction_call.kwargs["messages"]
        )
        user_msg = messages[1]["content"]
        assert "Previous Attempt Issues" in user_msg


class TestMain:
    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_success_returns_zero(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_chat.side_effect = _four_step_responses()
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

    @patch("scripts.generate_tests.content_check", return_value=CONTENT_PASS)
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_max_retries_cli_arg(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_content: MagicMock,
        ai_submission: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_chat.side_effect = _four_step_responses()
        rc = main([str(ai_submission), "--max-retries", "1"])
        assert rc == 0
