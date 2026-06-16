"""Cisco security scanner gate."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from abevalflow.gates.base import Finding, GateMode, GateResult, GateType, Severity
from abevalflow.gates.security import register_security_gate
from abevalflow.gates.security.base import SecurityGate
from abevalflow.schemas import GatePolicy

logger = logging.getLogger(__name__)


@register_security_gate("cisco")
class CiscoGate(SecurityGate):
    """Cisco AI Skill Scanner security gate.

    Reads security-scan.json produced by the test phase's security scan step.
    Pass/fail depends on mode:
    - warn: always passes (findings are advisory)
    - block: fails if HIGH or CRITICAL findings exist
    """

    name = "cisco"

    def evaluate(
        self,
        reports_dir: Path,
        policy: GatePolicy,
    ) -> GateResult:
        """Evaluate Cisco security scan results."""
        gate_policy = policy.get_gate_policy(self.name)

        if gate_policy.mode == GateMode.DISABLED:
            return GateResult(
                gate_type=GateType.SECURITY,
                gate_name=self.name,
                passed=True,
                score=1.0,
                mode=GateMode.DISABLED,
                findings=[],
                details={},
                message="Cisco security scan disabled",
            )

        scan_path = reports_dir / "security-scan.json"
        if not scan_path.exists():
            return GateResult(
                gate_type=GateType.SECURITY,
                gate_name=self.name,
                passed=True,
                score=1.0,
                mode=gate_policy.mode,
                findings=[],
                details={"scan_path": str(scan_path), "status": "not_found"},
                message="No security-scan.json found (scan may have been skipped)",
            )

        try:
            scan_data = json.loads(scan_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read security scan: %s", e)
            return GateResult(
                gate_type=GateType.SECURITY,
                gate_name=self.name,
                passed=False,
                score=0.0,
                mode=gate_policy.mode,
                findings=[],
                details={"error": str(e)},
                message=f"Failed to parse security-scan.json: {e}",
            )

        findings = []
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0}

        for f in scan_data.get("findings", []):
            sev_str = f.get("severity", "info").lower()
            try:
                severity = Severity(sev_str)
            except ValueError:
                severity = Severity.INFO

            if severity.value in severity_counts:
                severity_counts[severity.value] += 1

            findings.append(Finding(
                severity=severity,
                message=f.get("message", f.get("description", "")),
                location=f.get("file_path", f.get("location", {}).get("file")),
                rule_id=f.get("rule_id", f.get("id", "unknown")),
                details=f,
            ))

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
                severity_counts["critical"] * 0.0 +
                severity_counts["high"] * 0.25 +
                severity_counts["medium"] * 0.5 +
                severity_counts["low"] * 0.75 +
                severity_counts["info"] * 0.9
            )
            score = weighted_score / total_findings if total_findings > 0 else 1.0

        message = (
            f"Cisco scan: {total_findings} findings "
            f"(critical={severity_counts['critical']}, high={severity_counts['high']}, "
            f"medium={severity_counts['medium']}, low={severity_counts['low']})"
        )

        return GateResult(
            gate_type=GateType.SECURITY,
            gate_name=self.name,
            passed=passed,
            score=score,
            mode=gate_policy.mode,
            findings=findings,
            details={
                "severity_counts": severity_counts,
                "scan_path": str(scan_path),
            },
            message=message,
        )
