"""Edge case behavioral gate.

Evaluates how skills perform under edge case instructions
(empty input, adversarial input, boundary values, etc.).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from abevalflow.gates.base import GateMode, GateResult, GateType
from abevalflow.gates.behavioral import register_behavioral_gate
from abevalflow.gates.behavioral.base import BehavioralGate
from abevalflow.schemas import GatePolicy

logger = logging.getLogger(__name__)


@register_behavioral_gate("edge-case")
class EdgeCaseGate(BehavioralGate):
    """Gate that evaluates edge case test results.

    Reads edge case results from report.json and computes a score
    based on the fraction of edge cases that passed verification.
    """

    name = "edge-case"

    def evaluate(
        self,
        reports_dir: Path,
        policy: GatePolicy,
    ) -> GateResult:
        gate_policy = policy.get_gate_policy("behavioral")

        if gate_policy.mode == GateMode.DISABLED:
            return GateResult(
                gate_type=GateType.BEHAVIORAL,
                gate_name=self.name,
                policy_key=self.name,
                passed=True,
                score=1.0,
                mode=GateMode.DISABLED,
                message="Edge case behavioral gate disabled",
            )

        report_path = reports_dir / "report.json"
        if not report_path.exists():
            return GateResult(
                gate_type=GateType.BEHAVIORAL,
                gate_name=self.name,
                policy_key=self.name,
                passed=True,
                score=1.0,
                mode=gate_policy.mode,
                message="No report.json found (edge case testing skipped)",
                details={"status": "not_found"},
            )

        try:
            report_data = json.loads(report_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read report.json: %s", e)
            return GateResult(
                gate_type=GateType.BEHAVIORAL,
                gate_name=self.name,
                policy_key=self.name,
                passed=False,
                score=0.0,
                mode=gate_policy.mode,
                message=f"Failed to parse report.json: {e}",
            )

        edge_case_results = report_data.get("edge_case_results", [])
        if not edge_case_results:
            return GateResult(
                gate_type=GateType.BEHAVIORAL,
                gate_name=self.name,
                policy_key=self.name,
                passed=True,
                score=1.0,
                mode=gate_policy.mode,
                message="No edge case results in report",
                details={"status": "no_edge_cases"},
            )

        total = len(edge_case_results)
        passed_count = sum(1 for ec in edge_case_results if ec.get("passed", False))
        score = passed_count / total if total > 0 else 0.0

        threshold = gate_policy.threshold or self.get_default_threshold()
        passed = score >= threshold

        if gate_policy.mode == GateMode.WARN:
            passed = True

        return GateResult(
            gate_type=GateType.BEHAVIORAL,
            gate_name=self.name,
            policy_key=self.name,
            passed=passed,
            score=score,
            mode=gate_policy.mode,
            threshold=threshold,
            message=(
                f"Edge cases: {passed_count}/{total} passed "
                f"(score {score:.2f}, threshold {threshold})"
            ),
            details={
                "total": total,
                "passed": passed_count,
                "failed": total - passed_count,
                "results": edge_case_results,
            },
        )
