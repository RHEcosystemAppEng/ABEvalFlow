"""Tests for scripts/ai_review.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.ai_review import main, review_submission

VALID_METADATA = {
    "name": "review-skill",
    "description": "A skill to review",
    "generation_mode": "manual",
}

MOCK_PASS_RESPONSE = json.dumps({
    "dimensions": {
        "coherence": {"score": 0.9, "finding": "Instruction matches skill well."},
        "coverage": {"score": 0.8, "finding": "Tests cover main requirements."},
        "clarity": {"score": 0.85, "finding": "Clear and unambiguous."},
        "feasibility": {"score": 0.9, "finding": "Achievable in one session."},
        "robustness": {"score": 0.7, "finding": "Some edge cases missing."},
    },
    "overall_score": 0.83,
    "recommendation": "pass",
    "summary": "Good quality submission ready for evaluation.",
})

MOCK_FAIL_RESPONSE = json.dumps({
    "dimensions": {
        "coherence": {"score": 0.3, "finding": "Instruction does not match skill."},
        "coverage": {"score": 0.2, "finding": "Tests are trivial."},
        "clarity": {"score": 0.5, "finding": "Some ambiguity."},
        "feasibility": {"score": 0.4, "finding": "May be too complex."},
        "robustness": {"score": 0.2, "finding": "No edge cases."},
    },
    "overall_score": 0.32,
    "recommendation": "fail",
    "summary": "Significant quality issues prevent evaluation.",
})


@pytest.fixture()
def review_submission_dir(tmp_path: Path) -> Path:
    sub = tmp_path / "review-skill"
    sub.mkdir()
    (sub / "metadata.yaml").write_text(yaml.dump(VALID_METADATA))
    (sub / "skills").mkdir()
    (sub / "skills" / "SKILL.md").write_text("# Skill\nDo code review.\n")
    (sub / "instruction.md").write_text("# Task\nReview the code.\n")
    (sub / "tests").mkdir()
    (sub / "tests" / "test_outputs.py").write_text("def test_review(): assert True\n")
    return sub


class TestReviewSubmission:
    @patch("scripts.ai_review.llm_client.chat_completion")
    def test_passing_review(
        self,
        mock_chat: MagicMock,
        review_submission_dir: Path,
    ) -> None:
        mock_chat.return_value = MOCK_PASS_RESPONSE

        result = review_submission(review_submission_dir)

        assert result["passed"] is True
        assert result["recommendation"] == "pass"
        assert result["overall_score"] > 0.6
        assert "dimensions" in result
        assert "coherence" in result["dimensions"]

    @patch("scripts.ai_review.llm_client.chat_completion")
    def test_failing_review(
        self,
        mock_chat: MagicMock,
        review_submission_dir: Path,
    ) -> None:
        mock_chat.return_value = MOCK_FAIL_RESPONSE

        result = review_submission(review_submission_dir)

        assert result["passed"] is False
        assert result["recommendation"] == "fail"
        assert result["overall_score"] < 0.4

    @patch("scripts.ai_review.llm_client.chat_completion")
    def test_missing_overall_score_computed(
        self,
        mock_chat: MagicMock,
        review_submission_dir: Path,
    ) -> None:
        response_no_score = {
            "dimensions": {
                "coherence": {"score": 0.8, "finding": "Good."},
                "coverage": {"score": 0.7, "finding": "Ok."},
                "clarity": {"score": 0.9, "finding": "Clear."},
                "feasibility": {"score": 0.8, "finding": "Doable."},
                "robustness": {"score": 0.6, "finding": "Basic."},
            },
        }
        mock_chat.return_value = json.dumps(response_no_score)

        result = review_submission(review_submission_dir)

        assert "overall_score" in result
        assert 0.7 < result["overall_score"] < 0.9
        assert result["recommendation"] == "pass"
        assert result["passed"] is True

    @patch("scripts.ai_review.llm_client.chat_completion")
    def test_includes_llm_judge_when_present(
        self,
        mock_chat: MagicMock,
        review_submission_dir: Path,
    ) -> None:
        (review_submission_dir / "tests" / "llm_judge.py").write_text("score = 0.9\n")
        mock_chat.return_value = MOCK_PASS_RESPONSE

        review_submission(review_submission_dir)

        call_args = mock_chat.call_args
        messages = call_args.args[0] if call_args.args else call_args.kwargs["messages"]
        user_msg = messages[1]["content"]
        assert "llm_judge.py" in user_msg

    @patch("scripts.ai_review.llm_client.chat_completion")
    def test_handles_missing_files(
        self,
        mock_chat: MagicMock,
        review_submission_dir: Path,
    ) -> None:
        (review_submission_dir / "instruction.md").unlink()
        mock_chat.return_value = MOCK_PASS_RESPONSE

        result = review_submission(review_submission_dir)
        assert result["passed"] is True

    @patch("scripts.ai_review.llm_client.chat_completion")
    def test_invalid_json_response_raises(
        self,
        mock_chat: MagicMock,
        review_submission_dir: Path,
    ) -> None:
        mock_chat.return_value = "Not JSON at all, no braces here"

        with pytest.raises(ValueError, match="not valid JSON"):
            review_submission(review_submission_dir)


class TestMain:
    @patch("scripts.ai_review.llm_client.chat_completion")
    def test_pass_returns_zero(
        self,
        mock_chat: MagicMock,
        review_submission_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_chat.return_value = MOCK_PASS_RESPONSE
        rc = main([str(review_submission_dir)])
        assert rc == 0
        output = json.loads(capsys.readouterr().out)
        assert output["passed"] is True

    @patch("scripts.ai_review.llm_client.chat_completion")
    def test_fail_returns_one(
        self,
        mock_chat: MagicMock,
        review_submission_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_chat.return_value = MOCK_FAIL_RESPONSE
        rc = main([str(review_submission_dir)])
        assert rc == 1
        output = json.loads(capsys.readouterr().out)
        assert output["passed"] is False

    def test_nonexistent_dir_returns_one(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        rc = main([str(tmp_path / "nope")])
        assert rc == 1
        output = json.loads(capsys.readouterr().out)
        assert output["passed"] is False

    @patch("scripts.ai_review.llm_client.chat_completion")
    def test_api_error_returns_one(
        self,
        mock_chat: MagicMock,
        review_submission_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        mock_chat.side_effect = RuntimeError("API down")
        rc = main([str(review_submission_dir)])
        assert rc == 1
        output = json.loads(capsys.readouterr().out)
        assert "error" in output
