"""Tests for abevalflow/llm_client.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from abevalflow import llm_client
from abevalflow.llm_client import LLMResult


class TestResolveConfig:
    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        cfg = llm_client._resolve_config()
        assert cfg["base_url"] == llm_client.DEFAULT_BASE_URL
        assert cfg["api_key"] == "not-set"

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8080/v1")
        monkeypatch.setenv("LLM_API_KEY", "sk-test-123")
        cfg = llm_client._resolve_config()
        assert cfg["base_url"] == "http://localhost:8080/v1"
        assert cfg["api_key"] == "sk-test-123"


class TestGetModel:
    def test_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LLM_MODEL", raising=False)
        assert llm_client.get_model() == llm_client.DEFAULT_MODEL

    def test_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        assert llm_client.get_model() == "gpt-4o"


class TestChatCompletion:
    @patch("abevalflow.llm_client.get_client")
    def test_returns_content(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = llm_client.chat_completion(
            [{"role": "user", "content": "hi"}],
            model="test-model",
        )
        assert result == "Hello, world!"
        mock_client.chat.completions.create.assert_called_once()

    @patch("abevalflow.llm_client.get_client")
    def test_empty_content_returns_empty_string(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = llm_client.chat_completion(
            [{"role": "user", "content": "hi"}],
        )
        assert result == ""

    @patch("abevalflow.llm_client.get_client")
    def test_uses_default_model(
        self,
        mock_get_client: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("LLM_MODEL", "custom-model")
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        llm_client.chat_completion([{"role": "user", "content": "test"}])

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "custom-model"

    @patch("abevalflow.llm_client.get_client")
    def test_passes_temperature_and_max_tokens(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        llm_client.chat_completion(
            [{"role": "user", "content": "test"}],
            model="m",
            temperature=0.7,
            max_tokens=2048,
        )

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["temperature"] == 0.7
        assert call_kwargs.kwargs["max_tokens"] == 2048


class TestChatCompletionWithUsage:
    @patch("abevalflow.llm_client.get_client")
    def test_returns_llm_result(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = llm_client.chat_completion_with_usage(
            [{"role": "user", "content": "hi"}],
            model="test-model",
        )

        assert isinstance(result, LLMResult)
        assert result.content == "Hello!"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 50
        assert result.total_tokens == 150
        assert result.model == "test-model"

    @patch("abevalflow.llm_client.get_client")
    def test_handles_no_usage(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = llm_client.chat_completion_with_usage(
            [{"role": "user", "content": "hi"}],
            model="test-model",
        )

        assert result.content == "Hello!"
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0
        assert result.total_tokens == 0

    @patch("abevalflow.llm_client.get_client")
    def test_backward_compatible_chat_completion(self, mock_get_client: MagicMock) -> None:
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello!"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_client.chat.completions.create.return_value = mock_response
        mock_get_client.return_value = mock_client

        result = llm_client.chat_completion(
            [{"role": "user", "content": "hi"}],
            model="test-model",
        )

        assert isinstance(result, str)
        assert result == "Hello!"
