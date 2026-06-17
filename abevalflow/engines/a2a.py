"""A2A (Agent-to-Agent) evaluation engine adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from abevalflow.engines import register_engine
from abevalflow.engines.base import EvalEngine
from abevalflow.gates.base import GateResult, GateType
from abevalflow.schemas import GatePolicy

logger = logging.getLogger(__name__)


@register_engine("a2a")
class A2AEngine(EvalEngine):
    """A2A (Agent-to-Agent) evaluation engine.

    Reads report.json produced by scripts/analyze.py with AnalysisResult schema.
    Unlike Harbor/ASE, A2A is single-variant (no control). Pass/fail is based
    on mean_reward >= threshold (default 0.5).
    """

    name = "a2a"

    def read_result(self, reports_dir: Path) -> dict[str, Any] | None:
        """Read A2A's report.json."""
        report_path = reports_dir / "report.json"
        if not report_path.exists():
            logger.warning("A2A report not found: %s", report_path)
            return None

        try:
            return json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read A2A report: %s", e)
            return None

    def to_gate_result(
        self,
        raw_result: dict[str, Any],
        policy: GatePolicy,
    ) -> GateResult:
        """Convert A2A AnalysisResult to GateResult.

        Note:
            Pass/fail is determined by comparing mean_reward against the
            policy threshold. This is a simple numeric gate and does NOT
            replicate the full statistical logic of analyze.py. The upstream
            recommendation is included in the message for transparency but
            does not affect the gate decision.
        """
        gate_policy = policy.get_gate_policy(self.name)
        threshold = gate_policy.threshold if gate_policy.threshold is not None else 0.5

        summary = raw_result.get("summary", {})
        treatment = summary.get("treatment", {})

        mean_reward = treatment.get("mean_reward", 0.0) or 0.0
        pass_rate = treatment.get("pass_rate", 0.0) or 0.0

        score = min(1.0, max(0.0, mean_reward))

        # Compute pass/fail from policy threshold, not upstream recommendation
        passed = mean_reward >= threshold

        upstream_recommendation = summary.get("recommendation", "unknown")
        message = (
            f"A2A: mean_reward={mean_reward:.3f} >= threshold={threshold:.3f} -> "
            f"{'PASS' if passed else 'FAIL'} (upstream: {upstream_recommendation}, pass_rate={pass_rate:.3f})"
        )

        return GateResult(
            gate_type=GateType.ENGINE,
            gate_name=self.name,
            passed=passed,
            score=score,
            mode=gate_policy.mode,
            threshold=threshold,
            findings=[],
            details=raw_result,
            message=message,
        )

    def get_default_threshold(self) -> float:
        return 0.5
