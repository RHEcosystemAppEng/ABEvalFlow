"""Harbor evaluation engine adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from abevalflow.engines import register_engine
from abevalflow.engines.base import EvalEngine
from abevalflow.gates.base import GateMode, GateResult, GateType
from abevalflow.schemas import GatePolicy

logger = logging.getLogger(__name__)


@register_engine("harbor")
class HarborEngine(EvalEngine):
    """Harbor A/B evaluation engine.

    Reads report.json produced by scripts/analyze.py with AnalysisResult schema.
    Pass/fail is based on mean_reward_gap >= threshold (default 0.0).
    """

    name = "harbor"

    def read_result(self, reports_dir: Path) -> dict[str, Any] | None:
        """Read Harbor's report.json."""
        report_path = reports_dir / "report.json"
        if not report_path.exists():
            logger.warning("Harbor report not found: %s", report_path)
            return None

        try:
            return json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read Harbor report: %s", e)
            return None

    def to_gate_result(
        self,
        raw_result: dict[str, Any],
        policy: GatePolicy,
    ) -> GateResult:
        """Convert Harbor AnalysisResult to GateResult."""
        gate_policy = policy.get_gate_policy(self.name)
        threshold = gate_policy.threshold if gate_policy.threshold is not None else 0.0

        summary = raw_result.get("summary", {})
        treatment = summary.get("treatment", {})

        mean_reward_gap = summary.get("mean_reward_gap")
        if mean_reward_gap is None:
            mean_reward_gap = summary.get("uplift", 0.0)

        recommendation = summary.get("recommendation", "fail")
        passed = recommendation == "pass"

        treatment_mean = treatment.get("mean_reward", 0.0) or 0.0
        score = min(1.0, max(0.0, treatment_mean))

        message = (
            f"Harbor A/B: treatment mean_reward={treatment_mean:.3f}, "
            f"gap={mean_reward_gap:.3f}, threshold={threshold:.3f}"
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
        return 0.0
