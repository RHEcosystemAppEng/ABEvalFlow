"""MCPChecker evaluation engine adapter."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from abevalflow.engines import register_engine
from abevalflow.engines.base import EvalEngine
from abevalflow.gates.base import Finding, GateResult, GateType, Severity
from abevalflow.schemas import GatePolicy

logger = logging.getLogger(__name__)


@register_engine("mcpchecker")
class MCPCheckerEngine(EvalEngine):
    """MCPChecker evaluation engine.

    Reads mcpchecker-report.json produced by scripts/aggregate_mcpchecker.py.
    Unlike A/B engines, MCPChecker is task-based with a 70% pass threshold.
    """

    name = "mcpchecker"

    def read_result(self, reports_dir: Path) -> dict[str, Any] | None:
        """Read MCPChecker's report.json or mcpchecker-report.json."""
        for filename in ["mcpchecker-report.json", "report.json"]:
            report_path = reports_dir / filename
            if report_path.exists():
                try:
                    return json.loads(report_path.read_text())
                except (json.JSONDecodeError, OSError) as e:
                    logger.error("Failed to read MCPChecker report: %s", e)
                    return None

        logger.warning("MCPChecker report not found in: %s", reports_dir)
        return None

    def to_gate_result(
        self,
        raw_result: dict[str, Any],
        policy: GatePolicy,
    ) -> GateResult:
        """Convert MCPCheckerResult to GateResult."""
        gate_policy = policy.get_gate_policy("evaluation")
        threshold = gate_policy.threshold if gate_policy.threshold is not None else 0.7

        overall_score = raw_result.get("overall_score", 0.0)
        passed_tasks = raw_result.get("passed_tasks", 0)
        _failed_tasks = raw_result.get("failed_tasks", 0)  # Reserved for future use
        _error_tasks = raw_result.get("error_tasks", 0)  # Reserved for future use
        total_tasks = raw_result.get("total_tasks", 0)

        _recommendation = raw_result.get("recommendation", "fail")  # Reserved for future use
        passed = overall_score >= threshold

        findings = []
        for task in raw_result.get("tasks", []):
            status = task.get("status", "unknown")
            if status in ("failed", "error"):
                severity = Severity.HIGH if status == "error" else Severity.MEDIUM
                findings.append(
                    Finding(
                        severity=severity,
                        message=task.get("error_message") or f"Task {status}: {task.get('task_name', 'unknown')}",
                        location=task.get("task_id"),
                        rule_id=f"mcpchecker-{status}",
                    )
                )

        message = (
            f"MCPChecker: {passed_tasks}/{total_tasks} tasks passed "
            f"(score={overall_score:.2f}, threshold={threshold:.2f})"
        )

        return GateResult(
            gate_type=GateType.ENGINE,
            gate_name="evaluation",
            policy_key=self.name,
            passed=passed,
            score=overall_score,
            mode=gate_policy.mode,
            threshold=threshold,
            findings=findings,
            details={
                "engine": self.name,
                **raw_result,
            },
            message=message,
        )

    def get_default_threshold(self) -> float:
        return 0.7
