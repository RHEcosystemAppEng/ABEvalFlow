"""Base protocol for evaluation engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from abevalflow.gates.base import GateResult
from abevalflow.schemas import GatePolicy


class EvalEngine(ABC):
    """Abstract base class for evaluation engines.

    Engines read their evaluation results and convert them to a
    standardized GateResult for the unified scorecard.
    """

    name: str

    @abstractmethod
    def read_result(self, reports_dir: Path) -> dict[str, Any] | None:
        """Read the engine's result from the reports directory.

        Args:
            reports_dir: Path to reports/{submission-name}/

        Returns:
            Raw result dict, or None if not found
        """
        ...

    @abstractmethod
    def to_gate_result(
        self,
        raw_result: dict[str, Any],
        policy: GatePolicy,
    ) -> GateResult:
        """Convert engine result to standardized GateResult.

        Args:
            raw_result: Engine-specific result dict
            policy: Gate policy to apply

        Returns:
            Standardized GateResult
        """
        ...

    def get_default_threshold(self) -> float:
        """Get the engine's default pass/fail threshold."""
        return 0.0
