"""Tests for scripts/generate_tests.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.generate_tests import (
    DEFAULT_MAX_RETRIES,
    _correction_pass,
    _upload_to_minio,
    generate,
    main,
)

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

REVIEW_PASS = {"passed": True, "issues": [], "reviewer_results": {}}
REVIEW_FAIL = {
    "passed": False,
    "issues": ["[coverage] instruction does not match skill"],
    "reviewer_results": {},
}
FINAL_PASS = {"passed": True, "issues": []}
FINAL_FAIL = {"passed": False, "issues": ["tests are not deterministic"]}


def _four_step_responses(
    analysis: str = MOCK_ANALYSIS,
    instruction: str = MOCK_INSTRUCTION,
    test: str = MOCK_TEST,
    judge: str = MOCK_JUDGE_CODE,
) -> list[str]:
    """Return the 4 sequential LLM responses for a single attempt."""
    return [analysis, instruction, test, judge]


def _three_step_responses(
    analysis: str = MOCK_ANALYSIS,
    instruction: str = MOCK_INSTRUCTION,
    test: str = MOCK_TEST,
) -> list[str]:
    """Return the 3 LLM responses when llm_judge is skipped."""
    return [analysis, instruction, test]


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
    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_ai_mode_generates_files(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
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

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_empty_instruction_retries(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Step 0 succeeds, Step 1 produces empty -> retry from Step 0."""
        mock_chat.side_effect = [
            MOCK_ANALYSIS,
            "",
            *_four_step_responses(),
        ]
        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=2)
        assert "instruction.md" in generated

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_bad_python_in_tests_retries(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Step 2 produces invalid Python -> retry from Step 0."""
        mock_chat.side_effect = [
            MOCK_ANALYSIS,
            MOCK_INSTRUCTION,
            "def bad(\n",
            *_four_step_responses(),
        ]
        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=2)
        assert "tests/test_outputs.py" in generated

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill")
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_quality_criteria_included_when_skill_fetched(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
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

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_llm_judge_included_when_produced(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = _four_step_responses(judge=MOCK_JUDGE_CODE)
        generated = generate(ai_submission, tmp_path, agent_type="api")

        assert "tests/llm_judge.py" in generated
        assert (ai_submission / "tests" / "llm_judge.py").is_file()

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_llm_judge_skipped_when_metadata_flag_set(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        tmp_path: Path,
    ) -> None:
        """skip_llm_judge=true in metadata skips judge generation entirely."""
        sub = tmp_path / "submissions" / "no-judge"
        sub.mkdir(parents=True)
        meta = {**AI_METADATA, "name": "no-judge", "skip_llm_judge": True}
        (sub / "metadata.yaml").write_text(yaml.dump(meta))
        (sub / "skills").mkdir()
        (sub / "skills" / "SKILL.md").write_text(SKILL_CONTENT)

        mock_chat.side_effect = _three_step_responses()
        generated = generate(sub, tmp_path, agent_type="api")

        assert "tests/llm_judge.py" not in generated
        assert mock_chat.call_count == 3

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_bad_judge_silently_skipped(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = _four_step_responses(judge="def bad(\n")
        generated = generate(ai_submission, tmp_path, agent_type="api")

        assert "tests/llm_judge.py" not in generated

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_markdown_fence_stripped(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        fenced_instruction = "```markdown\n# Task\nDo something.\n```"
        fenced_test = "```python\ndef test_ok(): assert True\n```"
        mock_chat.side_effect = [MOCK_ANALYSIS, fenced_instruction, fenced_test, MOCK_JUDGE_CODE]
        generated = generate(ai_submission, tmp_path, agent_type="api")
        assert "instruction.md" in generated
        assert "tests/test_outputs.py" in generated


class TestSkillAnalysis:
    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_analysis_passed_to_instruction_prompt(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
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

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_analysis_passed_to_test_prompt(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
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

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_invalid_analysis_json_triggers_retry(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
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

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_analysis_missing_key_triggers_retry(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Analysis missing test_focus_areas key -> retry."""
        bad_analysis = json.dumps({"novel_aspects": ["x"], "common_knowledge": ["y"]})
        mock_chat.side_effect = [
            bad_analysis,
            *_four_step_responses(),
        ]
        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=2)
        assert "instruction.md" in generated

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_four_calls_per_attempt(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Each attempt makes exactly 4 LLM calls (analysis, instruction, tests, judge)."""
        mock_chat.side_effect = _four_step_responses()
        generate(ai_submission, tmp_path, agent_type="api", max_retries=1)
        assert mock_chat.call_count == 4


class TestRetryLoop:
    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check")
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_structural_fail_then_pass_retries(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = [*_four_step_responses(), *_four_step_responses()]
        mock_struct.side_effect = [["some leftover issue"], []]

        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=3)
        assert "instruction.md" in generated
        assert mock_chat.call_count == 8

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check")
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_reviewer_fail_triggers_correction_then_pass(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """Multi-reviewer fails, correction pass runs, final review passes."""
        mock_chat.side_effect = [
            *_four_step_responses(),
            "corrected instruction",
            "def test_ok(): assert True\n",
        ]
        mock_review.return_value = REVIEW_FAIL

        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=1)
        assert "instruction.md" in generated
        assert mock_chat.call_count == 6

    @patch("scripts.generate_tests.final_review", return_value=FINAL_FAIL)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_final_review_fail_exhausts_retries(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = [*_four_step_responses(), *_four_step_responses()]

        with pytest.raises(ValueError, match="final review after 2 attempts"):
            generate(ai_submission, tmp_path, agent_type="api", max_retries=2)

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check")
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_pytest_collect_fail_then_pass_retries(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        mock_chat.side_effect = [*_four_step_responses(), *_four_step_responses()]
        mock_collect.side_effect = [["pytest --collect-only failed"], []]

        generated = generate(ai_submission, tmp_path, agent_type="api", max_retries=3)
        assert "instruction.md" in generated

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_feedback_injected_on_retry(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        tmp_path: Path,
    ) -> None:
        """On retry, previous errors appear in the instruction prompt."""
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


class TestCorrectionPass:
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_rewrites_both_files(
        self,
        mock_chat: MagicMock,
        ai_submission: Path,
    ) -> None:
        (ai_submission / "instruction.md").write_text("old instruction")
        (ai_submission / "tests").mkdir(exist_ok=True)
        (ai_submission / "tests" / "test_outputs.py").write_text("def test_old(): pass\n")

        mock_chat.side_effect = [
            "corrected instruction",
            "def test_fixed(): assert True\n",
        ]
        _correction_pass(ai_submission, ["[coverage] missing edge case test"])

        assert "corrected instruction" in (ai_submission / "instruction.md").read_text()
        assert "test_fixed" in (ai_submission / "tests" / "test_outputs.py").read_text()
        assert mock_chat.call_count == 2

    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_empty_correction_preserves_original(
        self,
        mock_chat: MagicMock,
        ai_submission: Path,
    ) -> None:
        original = "original content"
        (ai_submission / "instruction.md").write_text(original)
        (ai_submission / "tests").mkdir(exist_ok=True)
        (ai_submission / "tests" / "test_outputs.py").write_text("def test_x(): pass\n")

        mock_chat.side_effect = ["", ""]
        _correction_pass(ai_submission, ["some issue"])

        assert (ai_submission / "instruction.md").read_text() == original


class TestMain:
    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_success_returns_zero(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
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

    @patch("scripts.generate_tests.final_review", return_value=FINAL_PASS)
    @patch("scripts.generate_tests.multi_reviewer_check", return_value=REVIEW_PASS)
    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    @patch("scripts.generate_tests.skill_loader.fetch_skill", return_value=None)
    @patch("scripts.generate_tests.llm_client.chat_completion")
    def test_max_retries_cli_arg(
        self,
        mock_chat: MagicMock,
        mock_fetch: MagicMock,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        mock_review: MagicMock,
        mock_final: MagicMock,
        ai_submission: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_chat.side_effect = _four_step_responses()
        rc = main([str(ai_submission), "--max-retries", "1"])
        assert rc == 0


class TestMinioUpload:
    def test_skips_when_no_credentials(self, ai_submission: Path) -> None:
        """No MinIO env vars → returns immediately without error."""
        _upload_to_minio(ai_submission, "test-skill")

    @patch.dict(
        "os.environ",
        {
            "MINIO_ENDPOINT": "http://minio:9000",
            "MINIO_ACCESS_KEY": "key",
            "MINIO_SECRET_KEY": "secret",
            "PIPELINE_RUN_ID": "run-99",
        },
    )
    @patch("scripts.generate_tests.upload_generated_files")
    def test_calls_upload_when_configured(
        self,
        mock_upload: MagicMock,
        ai_submission: Path,
    ) -> None:
        mock_upload.return_value = "20260428_test-skill_run-99"
        _upload_to_minio(ai_submission, "test-skill")

        mock_upload.assert_called_once_with(
            submission_dir=ai_submission,
            submission_name="test-skill",
            pipeline_run_id="run-99",
            endpoint="http://minio:9000",
            access_key="key",
            secret_key="secret",
        )

    @patch.dict(
        "os.environ",
        {
            "MINIO_ENDPOINT": "http://minio:9000",
            "MINIO_ACCESS_KEY": "key",
            "MINIO_SECRET_KEY": "secret",
        },
    )
    @patch("scripts.generate_tests.upload_generated_files", side_effect=Exception("conn refused"))
    def test_upload_failure_is_non_fatal(
        self,
        mock_upload: MagicMock,
        ai_submission: Path,
    ) -> None:
        _upload_to_minio(ai_submission, "test-skill")


class TestOracleMode:
    @pytest.fixture()
    def oracle_submission(self, tmp_path: Path) -> Path:
        sub = tmp_path / "submissions" / "oracle-skill"
        sub.mkdir(parents=True)
        (sub / "metadata.yaml").write_text(yaml.dump(AI_METADATA | {"name": "oracle-skill"}))
        (sub / "skills").mkdir()
        (sub / "skills" / "SKILL.md").write_text(SKILL_CONTENT)
        oracle = sub / "oracle"
        oracle.mkdir()
        (oracle / "instruction.md").write_text("<!-- #ai-generated-oracle -->\n# Task\nDo it.\n")
        (oracle / "test_outputs.py").write_text(
            "# #ai-generated-oracle\ndef test_ok(): assert True\n"
        )
        (oracle / "llm_judge.py").write_text(
            "# #ai-generated-oracle\nscore = 0.9\n"
        )
        return sub

    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    def test_copies_all_three_files(
        self,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        oracle_submission: Path,
        tmp_path: Path,
    ) -> None:
        generated = generate(oracle_submission, tmp_path, agent_type="oracle")

        assert "instruction.md" in generated
        assert "tests/test_outputs.py" in generated
        assert "tests/llm_judge.py" in generated
        assert (oracle_submission / "instruction.md").is_file()
        assert (oracle_submission / "tests" / "test_outputs.py").is_file()
        assert (oracle_submission / "tests" / "llm_judge.py").is_file()

    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    def test_skips_judge_when_metadata_flag_set(
        self,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        tmp_path: Path,
    ) -> None:
        sub = tmp_path / "submissions" / "no-judge-oracle"
        sub.mkdir(parents=True)
        meta = {**AI_METADATA, "name": "no-judge-oracle", "skip_llm_judge": True}
        (sub / "metadata.yaml").write_text(yaml.dump(meta))
        (sub / "skills").mkdir()
        (sub / "skills" / "SKILL.md").write_text(SKILL_CONTENT)
        oracle = sub / "oracle"
        oracle.mkdir()
        (oracle / "instruction.md").write_text("# Task\n")
        (oracle / "test_outputs.py").write_text("def test_ok(): assert True\n")
        (oracle / "llm_judge.py").write_text("score = 0.9\n")

        generated = generate(sub, tmp_path, agent_type="oracle")

        assert "tests/llm_judge.py" not in generated
        assert not (sub / "tests" / "llm_judge.py").is_file()

    def test_missing_oracle_dir_raises(
        self,
        tmp_path: Path,
    ) -> None:
        sub = tmp_path / "submissions" / "no-oracle"
        sub.mkdir(parents=True)
        (sub / "metadata.yaml").write_text(yaml.dump(AI_METADATA | {"name": "no-oracle"}))
        (sub / "skills").mkdir()
        (sub / "skills" / "SKILL.md").write_text(SKILL_CONTENT)

        with pytest.raises(ValueError, match="oracle/ directory"):
            generate(sub, tmp_path, agent_type="oracle")

    @patch("scripts.generate_tests.pytest_collect_check", return_value=[])
    @patch("scripts.generate_tests.structural_check", return_value=[])
    def test_no_llm_calls_made(
        self,
        mock_struct: MagicMock,
        mock_collect: MagicMock,
        oracle_submission: Path,
        tmp_path: Path,
    ) -> None:
        with patch("scripts.generate_tests.llm_client.chat_completion") as mock_chat:
            generate(oracle_submission, tmp_path, agent_type="oracle")
            mock_chat.assert_not_called()
