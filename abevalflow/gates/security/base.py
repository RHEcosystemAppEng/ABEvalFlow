"""Base protocol for security gates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from abevalflow.gates.base import GateResult
from abevalflow.schemas import GatePolicy


class SecurityGate(ABC):
    """Abstract base class for security gates.

    Security gates read scan results and convert them to a
    standardized GateResult for the unified scorecard.
    """

    name: str

    @abstractmethod
    def evaluate(
        self,
        reports_dir: Path,
        policy: GatePolicy,
    ) -> GateResult:
        """Evaluate security scan results.

        Args:
            reports_dir: Path to reports/{submission-name}/
            policy: Gate policy to apply

        Returns:
            Standardized GateResult
        """
        ...

    def get_default_threshold(self) -> float:
        """Get the gate's default pass threshold.

        For security gates, this typically means the maximum allowed
        severity level (e.g., 0.0 = no high/critical, 1.0 = any allowed).
        """
        return 1.0
