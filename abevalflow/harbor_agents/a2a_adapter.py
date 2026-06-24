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
A2A_TRAJECTORY_FILE = "trajectory.json"


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

    SUPPORTS_ATIF: bool = True

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
            await self._process_response(response_data, environment, context, instruction)
        except TimeoutError:
            self.logger.error(f"A2A request timed out after {self.timeout}s")
            await self._write_error_response(environment, f"Request timed out after {self.timeout} seconds")
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
        instruction: str,
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
            error_msg = status.get("message", {}).get("parts", [{}])[0].get("text", "Agent task failed")
            self.logger.error(f"A2A agent task failed: {error_msg}")

        agent_response_text = self._extract_response_text(result)
        self.logger.info(f"A2A agent response length: {len(agent_response_text)} chars")

        await self._write_response_files(environment, response_data, agent_response_text)

        trajectory = self._build_trajectory(result, instruction)
        await self._write_trajectory_file(environment, trajectory)

        self._populate_context(result, context)

        if self.context_id is None and "contextId" in result:
            self.context_id = result["contextId"]

    @staticmethod
    def _is_thought_part(part: dict[str, Any]) -> bool:
        """Return True when an A2A part carries ADK thinking metadata."""
        metadata = part.get("metadata") or {}
        return bool(metadata.get("adk_thought"))

    @classmethod
    def _split_text_parts(cls, parts: list[dict[str, Any]]) -> tuple[str, str | None]:
        """Split text parts into visible message text and reasoning content."""
        message_parts: list[str] = []
        reasoning_parts: list[str] = []

        for part in parts:
            if part.get("kind") != "text":
                continue
            text = part.get("text", "")
            if not text:
                continue
            if cls._is_thought_part(part):
                reasoning_parts.append(text)
            else:
                message_parts.append(text)

        message_text = "\n".join(message_parts).strip()
        reasoning_content = "\n".join(reasoning_parts).strip() or None
        return message_text, reasoning_content

    @staticmethod
    def _classify_data_part(part: dict[str, Any]) -> str | None:
        """Return the ADK data-part type (function_call/function_response)."""
        metadata = part.get("metadata") or {}
        return metadata.get("adk_type") or metadata.get("type")

    def _history_message_to_agent_step(
        self,
        parts: list[dict[str, Any]],
        step_id: int,
    ) -> dict[str, Any]:
        """Convert an A2A agent history message into an ATIF agent step."""
        message_text, reasoning_content = self._split_text_parts(parts)

        tool_calls: list[dict[str, Any]] = []
        observation_results: list[dict[str, Any]] = []

        for part in parts:
            if part.get("kind") != "data":
                continue

            adk_type = self._classify_data_part(part)
            data = part.get("data") or {}

            if adk_type == "function_call":
                call_id = data.get("id") or f"call-{uuid.uuid4().hex[:8]}"
                tool_calls.append(
                    {
                        "tool_call_id": call_id,
                        "function_name": data.get("name", ""),
                        "arguments": data.get("args") or {},
                    }
                )
            elif adk_type == "function_response":
                response_content = data.get("response")
                if isinstance(response_content, (dict, list)):
                    content = json.dumps(response_content, ensure_ascii=False)
                else:
                    content = str(response_content or "")

                observation_results.append(
                    {
                        "source_call_id": data.get("id"),
                        "content": content,
                    }
                )

        step: dict[str, Any] = {
            "step_id": step_id,
            "source": "agent",
            "message": message_text,
        }
        if reasoning_content:
            step["reasoning_content"] = reasoning_content
        if tool_calls:
            step["tool_calls"] = tool_calls
        if observation_results:
            step["observation"] = {"results": observation_results}

        return step

    def _collect_response_parts(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect agent-visible parts from artifacts and status message."""
        parts: list[dict[str, Any]] = []

        for artifact in result.get("artifacts", []):
            parts.extend(artifact.get("parts", []))

        status_message = result.get("status", {}).get("message", {})
        parts.extend(status_message.get("parts", []))
        parts.extend(result.get("parts", []))

        return parts

    def _build_trajectory(self, result: dict[str, Any], instruction: str) -> dict[str, Any]:
        """Build an ATIF v1.7 trajectory from the A2A task result."""
        session_id = result.get("id") or result.get("contextId") or str(uuid.uuid4())

        steps: list[dict[str, Any]] = []
        step_id = 1
        history = result.get("history", [])

        if history:
            for message in history:
                role = message.get("role", "")
                parts = message.get("parts", [])

                if role == "user":
                    message_text, _ = self._split_text_parts(parts)
                    if not message_text and step_id == 1:
                        message_text = instruction
                    steps.append(
                        {
                            "step_id": step_id,
                            "source": "user",
                            "message": message_text,
                        }
                    )
                    step_id += 1
                elif role == "agent":
                    steps.append(self._history_message_to_agent_step(parts, step_id))
                    step_id += 1
        else:
            steps.append(
                {
                    "step_id": step_id,
                    "source": "user",
                    "message": instruction,
                }
            )
            step_id += 1

            response_parts = self._collect_response_parts(result)
            if response_parts:
                steps.append(self._history_message_to_agent_step(response_parts, step_id))
            else:
                steps.append(
                    {
                        "step_id": step_id,
                        "source": "agent",
                        "message": self._extract_response_text(result),
                    }
                )

        usage = result.get("metadata", {}).get("adk_usage_metadata", {})
        final_metrics: dict[str, Any] = {"total_steps": len(steps)}
        if usage:
            final_metrics["total_prompt_tokens"] = usage.get("promptTokenCount")
            final_metrics["total_completion_tokens"] = usage.get("candidatesTokenCount")
            final_metrics["total_cached_tokens"] = usage.get("cachedContentTokenCount")

        agent_info: dict[str, Any] = {
            "name": self.name(),
            "version": self.version() or "1.0.0",
        }
        if self.model_name:
            agent_info["model_name"] = self.model_name
        agent_info["extra"] = {"endpoint": self.endpoint}

        return {
            "schema_version": "ATIF-v1.7",
            "session_id": session_id,
            "agent": agent_info,
            "steps": steps,
            "final_metrics": final_metrics,
        }

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

        if environment.capabilities.mounted:
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

    async def _write_trajectory_file(
        self,
        environment: BaseEnvironment,
        trajectory: dict[str, Any],
    ) -> None:
        """Write ATIF trajectory.json to the agent logs directory."""
        host_path = self.logs_dir / A2A_TRAJECTORY_FILE
        host_path.write_text(
            json.dumps(trajectory, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        if EnvironmentPaths is None:
            self.logger.warning("Harbor not available, skipping trajectory container upload")
            return

        trajectory_path_agent = str(EnvironmentPaths.agent_dir / A2A_TRAJECTORY_FILE)

        if environment.capabilities.mounted:
            return

        try:
            await environment.upload_file(
                source_path=str(host_path),
                target_path=trajectory_path_agent,
            )
        except Exception as e:
            self.logger.warning(f"Failed to upload trajectory file: {e}")

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
