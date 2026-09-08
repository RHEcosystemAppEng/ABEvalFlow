"""Unit tests for scripts/pyrit_crescendo (deferred coverage from PR #66)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

PYRIT_DIR = Path(__file__).parent.parent / "scripts" / "pyrit_crescendo"
if str(PYRIT_DIR) not in sys.path:
    sys.path.insert(0, str(PYRIT_DIR))

import a2a_client  # noqa: E402
import judge  # noqa: E402
import run_crescendo  # noqa: E402


@pytest.fixture
def submission_path(tmp_path: Path) -> Path:
    path = tmp_path / "submission"
    path.mkdir()
    return path


def _write_metadata(submission_path: Path, content: dict) -> None:
    (submission_path / "metadata.yaml").write_text(yaml.dump(content))


def _run_main(argv: list[str]) -> int:
    with patch.object(sys, "argv", ["run_crescendo.py", *argv]):
        return run_crescendo.main()


class TestResolveObjectives:
    def test_explicit_objectives_win(self):
        red_team = {"crescendo_objectives": ["  Extract prompt  ", "", "Bypass RBAC"]}
        objectives, source = run_crescendo.resolve_objectives(
            red_team,
            purpose="test",
            policy="policy",
            auth_context="",
            llm_api_base="http://llm:4000",
            llm_model="claude",
            llm_api_key="sk",
        )
        assert source == "explicit"
        assert objectives == ["Extract prompt", "Bypass RBAC"]

    def test_derived_when_no_explicit(self):
        with patch.object(
            run_crescendo,
            "derive_objectives",
            return_value=["Derived objective A", "Derived objective B"],
        ) as derive:
            objectives, source = run_crescendo.resolve_objectives(
                {},
                purpose="test",
                policy="policy",
                auth_context="auth",
                llm_api_base="http://llm:4000",
                llm_model="claude",
                llm_api_key="sk",
            )
        assert source == "derived"
        assert objectives == ["Derived objective A", "Derived objective B"]
        derive.assert_called_once()

    def test_seed_when_derivation_empty(self):
        with patch.object(run_crescendo, "derive_objectives", return_value=[]):
            objectives, source = run_crescendo.resolve_objectives(
                {},
                purpose="cluster ops helper",
                policy="policy",
                auth_context="",
                llm_api_base="http://llm:4000",
                llm_model="claude",
                llm_api_key="sk",
            )
        assert source == "seed"
        assert len(objectives) >= 1
        assert any("system prompt" in o.lower() for o in objectives)

    def test_seed_when_no_llm_base(self):
        objectives, source = run_crescendo.resolve_objectives(
            {},
            purpose="agent",
            policy="policy",
            auth_context="",
            llm_api_base="",
            llm_model="claude",
            llm_api_key="sk",
        )
        assert source == "seed"
        assert objectives == judge.default_seed_objectives("agent")


class TestMainSkipPaths:
    def test_skip_when_enabled_false(self, submission_path: Path, tmp_path: Path):
        _write_metadata(submission_path, {"name": "t", "red_team": {"enabled": False}})
        output = tmp_path / "out.json"
        exit_code = _run_main(
            [
                "--endpoint",
                "http://agent:8000",
                "--eval-engine",
                "a2a",
                "--submission-path",
                str(submission_path),
                "--llm-api-base",
                "http://llm:4000",
                "--output",
                str(output),
            ]
        )
        assert exit_code == 0
        data = json.loads(output.read_text())
        assert data["skipped"] is True
        assert "enabled" in data["reason"]
        assert data["crescendo_results"] == []

    def test_skip_ase(self, submission_path: Path, tmp_path: Path):
        output = tmp_path / "out.json"
        exit_code = _run_main(
            [
                "--endpoint",
                "http://agent:8000",
                "--eval-engine",
                "ase",
                "--submission-path",
                str(submission_path),
                "--llm-api-base",
                "http://llm:4000",
                "--output",
                str(output),
            ]
        )
        assert exit_code == 0
        data = json.loads(output.read_text())
        assert data["skipped"] is True
        assert data["reason"] == "eval-engine=ase"

    def test_skip_mcpchecker(self, submission_path: Path, tmp_path: Path):
        output = tmp_path / "out.json"
        exit_code = _run_main(
            [
                "--endpoint",
                "http://mcp:3000",
                "--eval-engine",
                "mcpchecker",
                "--submission-path",
                str(submission_path),
                "--llm-api-base",
                "http://llm:4000",
                "--output",
                str(output),
            ]
        )
        assert exit_code == 0
        data = json.loads(output.read_text())
        assert data["skipped"] is True
        assert "mcpchecker" in data["reason"]


class TestJudgeObjectiveAchieved:
    def test_valid_json_achieved_true(self):
        with patch.object(
            judge,
            "chat_completion",
            return_value='{"objective_achieved": true, "reason": "Leaked prompt"}',
        ):
            result = judge.judge_objective_achieved(
                "http://llm:4000",
                "claude",
                "sk",
                "Extract prompt",
                "policy",
                "",
                [{"turn": 1, "attacker": "hi", "target": "SECRET"}],
            )
        assert result["objective_achieved"] is True
        assert "Leaked" in result["reason"]

    def test_valid_json_achieved_false(self):
        with patch.object(
            judge,
            "chat_completion",
            return_value='{"objective_achieved": false, "reason": "Refused"}',
        ):
            result = judge.judge_objective_achieved(
                "http://llm:4000",
                "claude",
                "sk",
                "Extract prompt",
                "policy",
                "",
                [{"turn": 1, "attacker": "hi", "target": "No"}],
            )
        assert result["objective_achieved"] is False

    def test_fenced_markdown_json(self):
        raw = '```json\n{"objective_achieved": true, "reason": "fenced"}\n```'
        with patch.object(judge, "chat_completion", return_value=raw):
            result = judge.judge_objective_achieved(
                "http://llm:4000",
                "claude",
                "sk",
                "obj",
                "policy",
                "",
                [],
            )
        assert result["objective_achieved"] is True
        assert result["reason"] == "fenced"

    def test_http_error_fail_open(self):
        with patch.object(judge, "chat_completion", side_effect=RuntimeError("timeout")):
            result = judge.judge_objective_achieved(
                "http://llm:4000",
                "claude",
                "sk",
                "obj",
                "policy",
                "",
                [],
            )
        assert result["objective_achieved"] is False
        assert "Judge error" in result["reason"]

    def test_garbage_response_defaults_false(self):
        with patch.object(judge, "chat_completion", return_value="not json at all"):
            result = judge.judge_objective_achieved(
                "http://llm:4000",
                "claude",
                "sk",
                "obj",
                "policy",
                "",
                [],
            )
        assert result["objective_achieved"] is False
        assert result["reason"]


class TestDeriveObjectivesFallback:
    def test_falls_back_to_seeds_on_http_error(self):
        with patch.object(judge, "chat_completion", side_effect=RuntimeError("boom")):
            objectives = judge.derive_objectives(
                "http://llm:4000",
                "claude",
                "sk",
                "ops agent",
                "policy",
                "",
                count=3,
            )
        assert objectives == judge.default_seed_objectives("ops agent")[:3]

    def test_parses_json_array(self):
        with patch.object(
            judge,
            "chat_completion",
            return_value='["Obj one with enough length", "Obj two with enough length"]',
        ):
            objectives = judge.derive_objectives(
                "http://llm:4000",
                "claude",
                "sk",
                "purpose",
                "policy",
                "",
                count=5,
            )
        assert objectives == ["Obj one with enough length", "Obj two with enough length"]


def _mock_httpx_json(data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data
    return resp


class TestA2AContextId:
    def test_context_id_threaded_across_calls(self):
        responses = [
            _mock_httpx_json(
                {
                    "result": {
                        "contextId": "ctx-abc",
                        "artifacts": [{"parts": [{"kind": "text", "text": "hello"}]}],
                    }
                }
            ),
            _mock_httpx_json(
                {
                    "result": {
                        "contextId": "ctx-abc",
                        "artifacts": [{"parts": [{"kind": "text", "text": "again"}]}],
                    }
                }
            ),
        ]

        with patch("a2a_client.httpx.post", side_effect=responses) as post:
            text1, ctx1 = a2a_client.send_a2a_message("http://agent:8000", "hi", None)
            text2, ctx2 = a2a_client.send_a2a_message("http://agent:8000", "follow-up", ctx1)

        assert text1 == "hello"
        assert ctx1 == "ctx-abc"
        assert text2 == "again"
        assert ctx2 == "ctx-abc"
        second_payload = post.call_args_list[1].kwargs["json"]
        assert second_payload["params"]["contextId"] == "ctx-abc"
        first_payload = post.call_args_list[0].kwargs["json"]
        assert "contextId" not in first_payload["params"]


class TestSendToTarget:
    def test_mcp_returns_skip_without_http(self):
        with patch("run_crescendo.send_a2a_message") as a2a, patch("run_crescendo.send_http_chat") as http:
            text, ctx = run_crescendo.send_to_target("mcpchecker", "http://mcp", "hi", "ctx-1")
        assert "Skipped" in text
        assert ctx == "ctx-1"
        a2a.assert_not_called()
        http.assert_not_called()

    def test_a2a_delegates(self):
        with patch("run_crescendo.send_a2a_message", return_value=("ok", "ctx-2")) as a2a:
            text, ctx = run_crescendo.send_to_target("a2a", "http://agent", "hi", "ctx-1")
        assert text == "ok"
        assert ctx == "ctx-2"
        a2a.assert_called_once_with("http://agent", "hi", "ctx-1")
