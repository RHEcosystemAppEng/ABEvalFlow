"""Tests for configurable blocking/TLS behavior on A2AAgent.

Covers three related fixes to abevalflow.harbor_agents.a2a_adapter.A2AAgent:

1. Message parts sent in `message/send` now include an explicit "kind": "text"
   discriminator, matching the A2A spec's Part schema.
2. `configuration.blocking` is now sent on every `message/send` request and is
   configurable via the `blocking` kwarg (default True). Without it, some A2A
   servers return before the agent has finished, yielding an empty response.
3. TLS verification is configurable via the `verify_ssl` kwarg (default False,
   preserving the prior hardcoded `ssl=False` behavior) instead of being
   permanently hardcoded with no way to opt into verification.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from abevalflow.harbor_agents.a2a_adapter import A2AAgent


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    path = tmp_path / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _patched_session(mock_response: AsyncMock) -> tuple[AsyncMock, patch]:
    """Build a mocked aiohttp.ClientSession context and the patch to install it."""
    mock_post_ctx = AsyncMock()
    mock_post_ctx.__aenter__.return_value = mock_response

    mock_session = AsyncMock()
    mock_session.post = MagicMock(return_value=mock_post_ctx)

    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__.return_value = mock_session

    patcher = patch(
        "abevalflow.harbor_agents.a2a_adapter.aiohttp.ClientSession",
        return_value=mock_session_ctx,
    )
    return mock_session, patcher


def _mock_response(result: dict | None = None) -> AsyncMock:
    mock_response = AsyncMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = AsyncMock(return_value={"result": result or {}})
    return mock_response


class TestA2ABlockingConfiguration:
    def test_blocking_defaults_to_true(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com")
        assert agent.blocking is True

    def test_blocking_can_be_disabled(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com", blocking=False)
        assert agent.blocking is False

    @pytest.mark.asyncio
    async def test_run_sends_blocking_configuration_and_kind(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com")
        mock_session, patcher = _patched_session(_mock_response())

        with patcher:
            await agent.run("do the thing", MagicMock(), MagicMock())

        _, kwargs = mock_session.post.call_args
        payload = kwargs["json"]
        assert payload["params"]["configuration"] == {"blocking": True}
        assert payload["params"]["message"]["parts"] == [{"kind": "text", "text": "do the thing"}]

    @pytest.mark.asyncio
    async def test_run_respects_blocking_false(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com", blocking=False)
        mock_session, patcher = _patched_session(_mock_response())

        with patcher:
            await agent.run("do the thing", MagicMock(), MagicMock())

        _, kwargs = mock_session.post.call_args
        assert kwargs["json"]["params"]["configuration"] == {"blocking": False}


class TestA2AVerifySslConfiguration:
    def test_verify_ssl_defaults_to_false(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com")
        assert agent.verify_ssl is False

    def test_verify_ssl_can_be_enabled(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com", verify_ssl=True)
        assert agent.verify_ssl is True

    @pytest.mark.asyncio
    async def test_send_request_passes_verify_ssl_false_by_default(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com")
        mock_session, patcher = _patched_session(_mock_response())

        with patcher:
            await agent._send_request({"jsonrpc": "2.0", "id": "1"})

        _, kwargs = mock_session.post.call_args
        assert kwargs["ssl"] is False

    @pytest.mark.asyncio
    async def test_send_request_passes_verify_ssl_true_when_enabled(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com", verify_ssl=True)
        mock_session, patcher = _patched_session(_mock_response())

        with patcher:
            await agent._send_request({"jsonrpc": "2.0", "id": "1"})

        _, kwargs = mock_session.post.call_args
        assert kwargs["ssl"] is True
