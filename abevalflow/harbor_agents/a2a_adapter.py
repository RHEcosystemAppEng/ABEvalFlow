"""A2A Agent Adapter for Harbor.

This module provides a Harbor-compatible agent that communicates with external
A2A-compliant services (like google-lightspeed-agent) via JSON-RPC.

Usage with Harbor CLI:
    harbor run -p tasks/my-eval \\
        --agent-import-path abevalflow.harbor_agents.a2a_adapter:A2AAgent \\
        --ak endpoint=https://my-agent.example.com \\
        --ak timeout=120

Usage in Harbor config YAML:
    agents:
      - import_path: "abevalflow.harbor_agents.a2a_adapter:A2AAgent"
        kwargs:
          endpoint: "https://my-agent.example.com"
          timeout: 120
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import aiohttp

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
    from harbor.models.trial.paths import EnvironmentPaths
except ImportError:
    BaseAgent = object
    BaseEnvironment = object
    AgentContext = object
    EnvironmentPaths = None

logger = logging.getLogger(__name__)

A2A_RESPONSE_FILE = "a2a_response.json"
A2A_RESPONSE_TEXT_FILE = "a2a_response.txt"


class A2AAgent(BaseAgent):
    """Harbor agent that sends tasks to an external A2A service.

    This agent implements the Harbor BaseAgent interface but instead of
    executing tasks locally, it forwards them to an external A2A-compliant
    agent via JSON-RPC over HTTP.

    The agent writes the response to the workspace so verifiers can check
    the results. Token usage metadata from the A2A response is captured
    in the AgentContext.

    Attributes:
        endpoint: The URL of the A2A agent service.
        timeout: Request timeout in seconds.
        context_id: Optional context ID for multi-turn conversations.
    """

    SUPPORTS_ATIF: bool = False

    def __init__(
        self,
        logs_dir: Path,
        endpoint: str,
        timeout: int = 120,
        context_id: str | None = None,
        model_name: str | None = None,
        extra_env: dict[str, str] | None = None,
        **kwargs,
    ):
        """Initialize the A2A agent adapter.

        Args:
            logs_dir: Directory for agent logs.
            endpoint: URL of the A2A agent service (e.g., https://my-agent.example.com).
            timeout: Request timeout in seconds (default: 120).
            context_id: Optional context ID for conversation continuity.
            model_name: Optional model name for logging/tracking.
            extra_env: Extra environment variables (unused but accepted for compatibility).
            **kwargs: Additional arguments passed to BaseAgent.
        """
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self.endpoint = endpoint.rstrip("/")
        self.timeout = timeout
        self.context_id = context_id
        self._extra_env = extra_env or {}

    @staticmethod
    def name() -> str:
        """Return the agent name."""
        return "a2a-adapter"

    def version(self) -> str | None:
        """Return the agent version."""
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup is a no-op for A2A adapter since agent runs externally."""
        self.logger.info(f"A2A adapter configured for endpoint: {self.endpoint}")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Send the instruction to the A2A agent and capture the response.

        Args:
            instruction: The task instruction to send to the agent.
            environment: The Harbor environment (used for writing response files).
            context: The agent context to populate with token usage.
        """
        self.logger.info(f"Sending A2A request to {self.endpoint}")

        message_id = f"msg-{uuid.uuid4().hex[:8]}"
        request_id = f"req-{uuid.uuid4().hex[:8]}"

        payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": message_id,
                    "role": "user",
                    "parts": [{"text": instruction}],
                }
            },
            "id": request_id,
        }

        if self.context_id:
            payload["params"]["contextId"] = self.context_id

        try:
            response_data = await self._send_request(payload)
            await self._process_response(response_data, environment, context)
        except asyncio.TimeoutError:
            self.logger.error(f"A2A request timed out after {self.timeout}s")
            await self._write_error_response(
                environment, f"Request timed out after {self.timeout} seconds"
            )
            raise
        except aiohttp.ClientError as e:
            self.logger.error(f"A2A request failed: {e}")
            await self._write_error_response(environment, str(e))
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in A2A request: {e}")
            await self._write_error_response(environment, str(e))
            raise

    async def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send the A2A JSON-RPC request.

        Args:
            payload: The JSON-RPC request payload.

        Returns:
            The JSON response from the A2A agent.
        """
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                self.endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                ssl=False,
            ) as response:
                response.raise_for_status()
                return await response.json()

    async def _process_response(
        self,
        response_data: dict[str, Any],
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Process the A2A response and write results to workspace.

        Args:
            response_data: The JSON response from the A2A agent.
            environment: The Harbor environment for file operations.
            context: The agent context to populate with metadata.
        """
        result = response_data.get("result", {})
        error = response_data.get("error")

        if error:
            error_msg = error.get("message", "Unknown error")
            self.logger.error(f"A2A agent returned error: {error_msg}")
            await self._write_error_response(environment, error_msg)
            return

        status = result.get("status", {})
        state = status.get("state", "unknown")

        if state == "failed":
            error_msg = status.get("message", {}).get("parts", [{}])[0].get(
                "text", "Agent task failed"
            )
            self.logger.error(f"A2A agent task failed: {error_msg}")

        agent_response_text = self._extract_response_text(result)
        self.logger.info(f"A2A agent response length: {len(agent_response_text)} chars")

        await self._write_response_files(environment, response_data, agent_response_text)
        self._populate_context(result, context)

        if self.context_id is None and "contextId" in result:
            self.context_id = result["contextId"]

    def _extract_response_text(self, result: dict[str, Any]) -> str:
        """Extract the final text response from the A2A result.

        Args:
            result: The A2A result object.

        Returns:
            The concatenated text response from the agent.
        """
        text_parts = []

        for artifact in result.get("artifacts", []):
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    metadata = part.get("metadata", {})
                    if not metadata.get("adk_thought"):
                        text_parts.append(part.get("text", ""))

        if not text_parts:
            history = result.get("history", [])
            for msg in reversed(history):
                if msg.get("role") == "agent":
                    for part in msg.get("parts", []):
                        if part.get("kind") == "text":
                            metadata = part.get("metadata", {})
                            if not metadata.get("adk_thought"):
                                text_parts.append(part.get("text", ""))
                    if text_parts:
                        break

        return "\n".join(text_parts).strip()

    async def _write_response_files(
        self,
        environment: BaseEnvironment,
        response_data: dict[str, Any],
        response_text: str,
    ) -> None:
        """Write response files to the agent logs directory.

        Args:
            environment: The Harbor environment.
            response_data: The full JSON response.
            response_text: The extracted text response.
        """
        host_json_path = self.logs_dir / A2A_RESPONSE_FILE
        host_text_path = self.logs_dir / A2A_RESPONSE_TEXT_FILE

        host_json_path.write_text(json.dumps(response_data, indent=2))
        host_text_path.write_text(response_text)

        if EnvironmentPaths is None:
            self.logger.warning("Harbor not available, skipping container file writes")
            return

        json_path_agent = str(EnvironmentPaths.agent_dir / A2A_RESPONSE_FILE)
        text_path_agent = str(EnvironmentPaths.agent_dir / A2A_RESPONSE_TEXT_FILE)

        if environment.is_mounted:
            pass
        else:
            try:
                await environment.upload_file(
                    source_path=str(host_json_path),
                    target_path=json_path_agent,
                )
                await environment.upload_file(
                    source_path=str(host_text_path),
                    target_path=text_path_agent,
                )
            except Exception as e:
                self.logger.warning(f"Failed to upload response files: {e}")

    async def _write_error_response(
        self,
        environment: BaseEnvironment,
        error_message: str,
    ) -> None:
        """Write an error response to the workspace.

        Args:
            environment: The Harbor environment.
            error_message: The error message to write.
        """
        error_response = {
            "error": True,
            "message": error_message,
        }
        await self._write_response_files(
            environment,
            error_response,
            f"ERROR: {error_message}",
        )

    def _populate_context(
        self,
        result: dict[str, Any],
        context: AgentContext,
    ) -> None:
        """Populate the agent context with token usage metadata.

        Args:
            result: The A2A result object.
            context: The agent context to populate.
        """
        metadata = result.get("metadata", {})
        usage = metadata.get("adk_usage_metadata", {})

        if usage:
            context.n_input_tokens = usage.get("promptTokenCount")
            context.n_output_tokens = usage.get("candidatesTokenCount")
            context.n_cache_tokens = usage.get("cachedContentTokenCount")

            context.metadata = context.metadata or {}
            context.metadata["a2a_usage"] = usage
            context.metadata["a2a_task_id"] = result.get("id")
            context.metadata["a2a_context_id"] = result.get("contextId")
            context.metadata["a2a_status"] = result.get("status", {}).get("state")
