"""Base class for behavioral gates."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from abevalflow.gates.base import GateResult
from abevalflow.schemas import GatePolicy

logger = logging.getLogger(__name__)


class BehavioralGate(ABC):
    """Abstract base class for behavioral gates.

    Behavioral gates evaluate skill behavior under stress, edge cases,
    and unusual conditions — beyond the happy path.
    """

    name: str

    @abstractmethod
    def evaluate(
        self,
        reports_dir: Path,
        policy: GatePolicy,
    ) -> GateResult:
        """Evaluate behavioral test results.

        Args:
            reports_dir: Path to reports/{submission-name}/
            policy: Gate policy to apply

        Returns:
            Standardized GateResult
        """
        ...

    def get_default_threshold(self) -> float:
        """Get the gate's default pass threshold."""
        return 0.5
