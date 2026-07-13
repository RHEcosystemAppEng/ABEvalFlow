"""Deterministic SKILL.md quality gate.

Reads skillmd-quality-scan.json produced by the quality scan step.
Checks description quality, broken references, file completeness,
imprecise instructions, unfinished content, and generic advice.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from abevalflow.gates.base import Finding, GateMode, GateResult, GateType, Severity
from abevalflow.gates.quality import register_quality_gate
from abevalflow.gates.quality.base import QualityGate
from abevalflow.schemas import GatePolicy

logger = logging.getLogger(__name__)


@register_quality_gate("skillmd-quality")
class SkillMdQualityGate(QualityGate):
    """Deterministic quality checks for SKILL.md content.

    Reads skillmd-quality-scan.json. Pass/fail depends on mode:
    - warn: always passes (findings are advisory)
    - block: fails if HIGH or CRITICAL findings exist
    """

    name = "skillmd-quality"

    def evaluate(
        self,
        workspace_root: Path,
        policy: GatePolicy,
    ) -> GateResult:
        """Evaluate deterministic quality scan results."""
        gate_policy = policy.get_gate_policy("quality")

        if gate_policy.mode == GateMode.DISABLED:
            return GateResult(
                gate_type=GateType.QUALITY,
                gate_name="quality",
                policy_key=self.name,
                passed=True,
                score=1.0,
                mode=GateMode.DISABLED,
                findings=[],
                details={"scanner": self.name},
                message="SKILL.md quality scan disabled",
            )

        scan_path = workspace_root / "skillmd-quality-scan.json"
        if not scan_path.exists():
            if gate_policy.mode == GateMode.BLOCK:
                return GateResult(
                    gate_type=GateType.QUALITY,
                    gate_name="quality",
                    policy_key=self.name,
                    passed=False,
                    score=0.0,
                    mode=gate_policy.mode,
                    findings=[],
                    details={
                        "scanner": self.name,
                        "scan_path": str(scan_path),
                        "status": "not_found",
                    },
                    message=("FAIL: skillmd-quality-scan.json missing (required in block mode)"),
                )
            return GateResult(
                gate_type=GateType.QUALITY,
                gate_name="quality",
                policy_key=self.name,
                passed=True,
                score=1.0,
                mode=gate_policy.mode,
                findings=[],
                details={
                    "scanner": self.name,
                    "scan_path": str(scan_path),
                    "status": "not_found",
                },
                message=("No skillmd-quality-scan.json found (scan may have been skipped)"),
            )

        try:
            scan_data = json.loads(scan_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read quality scan: %s", e)
            return GateResult(
                gate_type=GateType.QUALITY,
                gate_name="quality",
                policy_key=self.name,
                passed=False,
                score=0.0,
                mode=gate_policy.mode,
                findings=[],
                details={"scanner": self.name, "error": str(e)},
                message=f"Failed to parse skillmd-quality-scan.json: {e}",
            )

        findings: list[Finding] = []
        severity_counts = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "info": 0,
        }

        for f in scan_data.get("findings", []):
            sev_str = f.get("severity", "info").lower().strip()
            try:
                severity = Severity(sev_str)
            except ValueError:
                severity = Severity.INFO

            if severity.value in severity_counts:
                severity_counts[severity.value] += 1

            findings.append(
                Finding(
                    severity=severity,
                    message=f.get("message", ""),
                    location=f.get("file_path"),
                    rule_id=f.get("rule_id", "unknown"),
                    details=f,
                )
            )

        high_or_critical = severity_counts["critical"] + severity_counts["high"]

        if gate_policy.mode == GateMode.BLOCK:
            passed = high_or_critical == 0
        else:
            passed = True

        total_findings = len(findings)
        if total_findings == 0:
            score = 1.0
        else:
            weighted_score = (
                severity_counts["critical"] * 0.0
                + severity_counts["high"] * 0.25
                + severity_counts["medium"] * 0.5
                + severity_counts["low"] * 0.75
                + severity_counts["info"] * 0.9
            )
            score = weighted_score / total_findings

        message = (
            f"SKILL.md quality: {total_findings} findings "
            f"(critical={severity_counts['critical']},"
            f" high={severity_counts['high']},"
            f" medium={severity_counts['medium']},"
            f" low={severity_counts['low']})"
        )

        return GateResult(
            gate_type=GateType.QUALITY,
            gate_name="quality",
            policy_key=self.name,
            passed=passed,
            score=score,
            mode=gate_policy.mode,
            findings=findings,
            details={
                "scanner": self.name,
                "severity_counts": severity_counts,
                "scan_path": str(scan_path),
            },
            message=message,
        )
