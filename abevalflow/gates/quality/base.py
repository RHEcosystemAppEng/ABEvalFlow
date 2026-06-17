"""Base protocol for quality gates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from abevalflow.gates.base import GateResult
from abevalflow.schemas import GatePolicy


class QualityGate(ABC):
    """Abstract base class for quality gates.

    Quality gates read review results and convert them to a
    standardized GateResult for the unified scorecard.
    """

    name: str

    @abstractmethod
    def evaluate(
        self,
        workspace_root: Path,
        policy: GatePolicy,
    ) -> GateResult:
        """Evaluate quality review results.

        Args:
            workspace_root: Path to workspace root (containing _ai_review.json)
            policy: Gate policy to apply

        Returns:
            Standardized GateResult
        """
        ...

    def get_default_threshold(self) -> float:
        """Get the gate's default pass threshold.

        Quality gates use score thresholds (e.g., 0.6 = all dimensions >= 0.6).
        """
        return 0.6
