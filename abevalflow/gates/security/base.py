"""Base class for security gates."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from abevalflow.gates.base import Finding, GateMode, GateResult, GateType, Severity
from abevalflow.schemas import GatePolicy

logger = logging.getLogger(__name__)


class SecurityGate(ABC):
    """Abstract base class for security gates.

    Security gates read scan results and convert them to a
    standardized GateResult for the unified scorecard.

    Subclasses only need to set ``name`` and ``scan_filename``.
    The shared evaluation logic (read JSON, parse findings, compute
    score, build GateResult) lives in ``evaluate_scan_json``.
    """

    name: str
    scan_filename: str

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
        """Get the gate's default pass threshold."""
        return 1.0

    # ------------------------------------------------------------------
    # Shared implementation
    # ------------------------------------------------------------------

    def evaluate_scan_json(
        self,
        reports_dir: Path,
        policy: GatePolicy,
    ) -> GateResult:
        """Read a JSON scan report and produce a GateResult.

        Handles disabled mode, missing/corrupt files, finding extraction,
        severity counting, score calculation, and pass/fail logic.
        Subclasses call this from their ``evaluate`` method.
        """
        gate_policy = policy.get_gate_policy("security")

        if gate_policy.mode == GateMode.DISABLED:
            return GateResult(
                gate_type=GateType.SECURITY,
                gate_name="security",
                policy_key=self.name,
                passed=True,
                score=1.0,
                mode=GateMode.DISABLED,
                findings=[],
                details={"scanner": self.name},
                message=f"{self.name} security scan disabled",
            )

        scan_path = reports_dir / self.scan_filename
        if not scan_path.exists():
            if gate_policy.mode == GateMode.BLOCK:
                return GateResult(
                    gate_type=GateType.SECURITY,
                    gate_name="security",
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
                    message=(f"FAIL: {self.scan_filename} missing (required in block mode)"),
                )
            return GateResult(
                gate_type=GateType.SECURITY,
                gate_name="security",
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
                message=(f"No {self.scan_filename} found (scan may have been skipped)"),
            )

        try:
            scan_data = json.loads(scan_path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to read security scan: %s", e)
            return GateResult(
                gate_type=GateType.SECURITY,
                gate_name="security",
                policy_key=self.name,
                passed=False,
                score=0.0,
                mode=gate_policy.mode,
                findings=[],
                details={"scanner": self.name, "error": str(e)},
                message=f"Failed to parse {self.scan_filename}: {e}",
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
                    message=f.get("message", f.get("description", "")),
                    location=f.get("file_path", f.get("location", {}).get("file")),
                    rule_id=f.get("rule_id", f.get("id", "unknown")),
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
            f"{self.name} scan: {total_findings} findings "
            f"(critical={severity_counts['critical']},"
            f" high={severity_counts['high']},"
            f" medium={severity_counts['medium']},"
            f" low={severity_counts['low']})"
        )

        return GateResult(
            gate_type=GateType.SECURITY,
            gate_name="security",
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
