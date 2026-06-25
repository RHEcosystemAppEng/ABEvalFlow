"""SKILL.md security scanner gate.

Reads skillmd-security-scan.json produced by the skillmd-security-scan
pipeline task. Patterns ported from harness-eval-lab (setup-eval).
"""

from __future__ import annotations

from pathlib import Path

from abevalflow.gates.base import GateResult
from abevalflow.gates.security import register_security_gate
from abevalflow.gates.security.base import SecurityGate
from abevalflow.schemas import GatePolicy


@register_security_gate("skillmd-scanner")
class SkillMdScannerGate(SecurityGate):
    """SKILL.md content security scanner gate.

    Reads skillmd-security-scan.json produced by the test phase's scan step.
    Pass/fail depends on mode:
    - warn: always passes (findings are advisory)
    - block: fails if HIGH or CRITICAL findings exist
    """

    name = "skillmd-scanner"
    scan_filename = "skillmd-security-scan.json"

    def evaluate(
        self,
        reports_dir: Path,
        policy: GatePolicy,
    ) -> GateResult:
        """Evaluate SKILL.md security scan results."""
        return self.evaluate_scan_json(reports_dir, policy)
