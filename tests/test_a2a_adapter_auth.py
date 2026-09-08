"""Tests for optional Bearer auth on abevalflow.harbor_agents.a2a_adapter.A2AAgent."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from abevalflow.harbor_agents.a2a_adapter import A2AAgent


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


class TestA2AAuthTokenResolution:
    def test_auth_token_kwarg(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com", auth_token="jwt-kwarg")
        assert agent._auth_token == "jwt-kwarg"

    def test_extra_env_agent_auth_token(self, logs_dir: Path) -> None:
        agent = A2AAgent(
            logs_dir,
            "https://agent.example.com",
            extra_env={"AGENT_AUTH_TOKEN": "jwt-extra"},
        )
        assert agent._auth_token == "jwt-extra"

    def test_env_agent_auth_token(self, logs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_AUTH_TOKEN", "jwt-env")
        agent = A2AAgent(logs_dir, "https://agent.example.com")
        assert agent._auth_token == "jwt-env"

    def test_kwarg_overrides_env(self, logs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_AUTH_TOKEN", "jwt-env")
        agent = A2AAgent(logs_dir, "https://agent.example.com", auth_token="jwt-kwarg")
        assert agent._auth_token == "jwt-kwarg"

    def test_no_token(self, logs_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("AGENT_AUTH_TOKEN", raising=False)
        agent = A2AAgent(logs_dir, "https://agent.example.com")
        assert agent._auth_token == ""


class TestA2ASendRequestHeaders:
    @pytest.mark.asyncio
    async def test_includes_bearer_header_when_token_set(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com", auth_token="secret-jwt")

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"result": {}})

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_session

        with patch(
            "abevalflow.harbor_agents.a2a_adapter.aiohttp.ClientSession",
            return_value=mock_session_ctx,
        ):
            await agent._send_request({"jsonrpc": "2.0", "id": "1"})

        _, kwargs = mock_session.post.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer secret-jwt"
        assert kwargs["headers"]["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_omits_authorization_without_token(self, logs_dir: Path) -> None:
        agent = A2AAgent(logs_dir, "https://agent.example.com")

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json = AsyncMock(return_value={"result": {}})

        mock_post_ctx = AsyncMock()
        mock_post_ctx.__aenter__.return_value = mock_response

        mock_session = AsyncMock()
        mock_session.post = MagicMock(return_value=mock_post_ctx)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_session

        with patch(
            "abevalflow.harbor_agents.a2a_adapter.aiohttp.ClientSession",
            return_value=mock_session_ctx,
        ):
            await agent._send_request({"jsonrpc": "2.0", "id": "1"})

        _, kwargs = mock_session.post.call_args
        assert "Authorization" not in kwargs["headers"]
        assert kwargs["headers"] == {"Content-Type": "application/json"}
