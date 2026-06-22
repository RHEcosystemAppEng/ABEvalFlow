"""Certification level computation for skill submissions.

Defines three certification levels (Foundational, Trusted, Certified) based on
which checks pass. Each level has required checks that must all pass for the
level to be achieved.

Levels are hierarchical: Certified requires Trusted, Trusted requires Foundational.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, computed_field

from abevalflow.gates.base import GateResult, GateType

if TYPE_CHECKING:
    from abevalflow.schemas import CertificationPolicy

logger = logging.getLogger(__name__)


class CertificationLevel(StrEnum):
    """Certification levels in ascending order of rigor."""

    NONE = "none"
    FOUNDATIONAL = "foundational"
    TRUSTED = "trusted"
    CERTIFIED = "certified"


class CheckId(StrEnum):
    """Identifiers for all certification checks.

    Checks are grouped by certification level.
    """

    # Foundational checks (6-10 in the spec)
    VALID_SKILL_STRUCTURE = "valid_skill_structure"
    BASIC_SECURITY_VALIDATION = "basic_security_validation"
    BASIC_EXECUTION_VALIDATION = "basic_execution_validation"
    CONTENT_QUALITY_REVIEW = "content_quality_review"
    METADATA_COMPLIANCE = "metadata_compliance"

    # Trusted checks (11-18 in the spec)
    EVALUATION_ASSETS = "evaluation_assets"
    ADVANCED_SECURITY_VALIDATION = "advanced_security_validation"
    FUNCTIONAL_VALIDATION = "functional_validation"
    INSTRUCTION_QUALITY = "instruction_quality"
    REGISTRY_GOVERNANCE = "registry_governance"
    OPERATIONAL_POLICY_COMPLIANCE = "operational_policy_compliance"
    EFFICIENCY_COST_PROFILING = "efficiency_cost_profiling"
    DATA_PRIVACY_PII_HANDLING = "data_privacy_pii_handling"

    # Certified checks (1-7 in the spec)
    ENTERPRISE_STRUCTURE_VALIDATION = "enterprise_structure_validation"
    ENTERPRISE_SECURITY_REVIEW = "enterprise_security_review"
    ENTERPRISE_BEHAVIORAL_TESTING = "enterprise_behavioral_testing"
    ADVANCED_AGENT_VALIDATION = "advanced_agent_validation"
    CONTINUOUS_OPTIMIZATION = "continuous_optimization"
    SAFETY_TOXICITY_BIAS_GUARDRAILS = "safety_toxicity_bias_guardrails"
    RESILIENCE_CHAOS_TESTING = "resilience_chaos_testing"


FOUNDATIONAL_CHECKS = [
    CheckId.VALID_SKILL_STRUCTURE,
    CheckId.BASIC_SECURITY_VALIDATION,
    CheckId.BASIC_EXECUTION_VALIDATION,
    CheckId.CONTENT_QUALITY_REVIEW,
    CheckId.METADATA_COMPLIANCE,
]

TRUSTED_CHECKS = [
    CheckId.EVALUATION_ASSETS,
    CheckId.ADVANCED_SECURITY_VALIDATION,
    CheckId.FUNCTIONAL_VALIDATION,
    CheckId.INSTRUCTION_QUALITY,
    # CheckId.REGISTRY_GOVERNANCE,  # Not yet implemented
    # CheckId.OPERATIONAL_POLICY_COMPLIANCE,  # Not yet implemented
]

CERTIFIED_CHECKS = [
    CheckId.ENTERPRISE_STRUCTURE_VALIDATION,
    CheckId.ENTERPRISE_SECURITY_REVIEW,
    # CheckId.ENTERPRISE_BEHAVIORAL_TESTING,  # Not yet implemented
    CheckId.ADVANCED_AGENT_VALIDATION,
    # CheckId.CONTINUOUS_OPTIMIZATION,  # Not yet implemented
]


class CheckResult(BaseModel):
    """Result of a single certification check."""

    check_id: CheckId = Field(description="Unique identifier for the check")
    name: str = Field(description="Human-readable check name")
    passed: bool = Field(description="Whether the check passed")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Score from 0-1")
    message: str = Field(default="", description="Details or reason for result")
    source_gate: str | None = Field(
        default=None,
        description="Gate that provided this result (e.g. 'security', 'evaluation')",
    )
    details: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional details from the check",
    )


class LevelResult(BaseModel):
    """Result for a single certification level."""

    level: CertificationLevel = Field(description="The certification level")
    passed: bool = Field(description="Whether all required checks passed")
    checks: list[CheckResult] = Field(
        default_factory=list,
        description="Results of checks required for this level",
    )

    @computed_field
    @property
    def checks_passed(self) -> int:
        """Number of checks that passed."""
        return sum(1 for c in self.checks if c.passed)

    @computed_field
    @property
    def checks_total(self) -> int:
        """Total number of checks for this level."""
        return len(self.checks)

    @computed_field
    @property
    def overall_score(self) -> float:
        """Average score across all checks."""
        if not self.checks:
            return 0.0
        return sum(c.score for c in self.checks) / len(self.checks)


class CertificationResult(BaseModel):
    """Complete certification result across all levels."""

    foundational: LevelResult = Field(description="Foundational level result")
    trusted: LevelResult = Field(description="Trusted level result")
    certified: LevelResult = Field(description="Certified level result")

    @computed_field
    @property
    def highest_level(self) -> CertificationLevel:
        """Highest certification level achieved.

        Levels are hierarchical: Certified requires Trusted, Trusted requires Foundational.
        """
        if self.certified.passed and self.trusted.passed and self.foundational.passed:
            return CertificationLevel.CERTIFIED
        if self.trusted.passed and self.foundational.passed:
            return CertificationLevel.TRUSTED
        if self.foundational.passed:
            return CertificationLevel.FOUNDATIONAL
        return CertificationLevel.NONE

    def get_level_result(self, level: CertificationLevel) -> LevelResult | None:
        """Get the result for a specific level."""
        if level == CertificationLevel.FOUNDATIONAL:
            return self.foundational
        elif level == CertificationLevel.TRUSTED:
            return self.trusted
        elif level == CertificationLevel.CERTIFIED:
            return self.certified
        return None


DEFAULT_THRESHOLDS: dict[CheckId, float] = {
    CheckId.ADVANCED_AGENT_VALIDATION: 0.8,
    CheckId.ADVANCED_SECURITY_VALIDATION: 0.9,
    CheckId.INSTRUCTION_QUALITY: 0.7,
}


def _get_threshold(
    check_id: CheckId,
    threshold_overrides: dict[str, float] | None = None,
) -> float:
    """Get threshold for a check, using override if provided."""
    if threshold_overrides and check_id.value in threshold_overrides:
        return threshold_overrides[check_id.value]
    return DEFAULT_THRESHOLDS.get(check_id, 0.0)


def _map_gate_to_checks(
    gate: GateResult,
    validation_passed: bool = True,
    threshold_overrides: dict[str, float] | None = None,
) -> list[CheckResult]:
    """Map a gate result to certification checks.

    Args:
        gate: The gate result to map
        validation_passed: Whether pre-evaluation validation passed
        threshold_overrides: Optional dict of check_id -> threshold overrides

    Returns:
        List of check results derived from this gate
    """
    checks: list[CheckResult] = []
    source_impl = gate.policy_key or gate.gate_name

    if gate.gate_type == GateType.ENGINE:
        checks.append(
            CheckResult(
                check_id=CheckId.BASIC_EXECUTION_VALIDATION,
                name="Basic Execution Validation",
                passed=gate.passed,
                score=gate.score,
                message=gate.message or "",
                source_gate=gate.gate_name,
                details={"source_implementation": source_impl},
            )
        )
        checks.append(
            CheckResult(
                check_id=CheckId.FUNCTIONAL_VALIDATION,
                name="Functional Validation",
                passed=gate.passed,
                score=gate.score,
                message=gate.message or "",
                source_gate=gate.gate_name,
                details={"source_implementation": source_impl},
            )
        )
        advanced_threshold = _get_threshold(
            CheckId.ADVANCED_AGENT_VALIDATION, threshold_overrides
        )
        if gate.score >= advanced_threshold:
            checks.append(
                CheckResult(
                    check_id=CheckId.ADVANCED_AGENT_VALIDATION,
                    name="Advanced Agent Validation",
                    passed=True,
                    score=gate.score,
                    message="High evaluation score indicates advanced validation passed",
                    source_gate=gate.gate_name,
                    details={"source_implementation": source_impl},
                )
            )
        else:
            checks.append(
                CheckResult(
                    check_id=CheckId.ADVANCED_AGENT_VALIDATION,
                    name="Advanced Agent Validation",
                    passed=False,
                    score=gate.score,
                    message=f"Score {gate.score:.2f} below {advanced_threshold:.2f} threshold for advanced validation",
                    source_gate=gate.gate_name,
                    details={"source_implementation": source_impl},
                )
            )

    elif gate.gate_type == GateType.SECURITY:
        checks.append(
            CheckResult(
                check_id=CheckId.BASIC_SECURITY_VALIDATION,
                name="Basic Security Validation",
                passed=gate.passed,
                score=gate.score,
                message=gate.message or "",
                source_gate=gate.gate_name,
                details={
                    "source_implementation": source_impl,
                    "findings": [f.model_dump() for f in gate.findings],
                },
            )
        )
        adv_security_threshold = _get_threshold(
            CheckId.ADVANCED_SECURITY_VALIDATION, threshold_overrides
        )
        high_score = gate.score >= adv_security_threshold and gate.passed
        checks.append(
            CheckResult(
                check_id=CheckId.ADVANCED_SECURITY_VALIDATION,
                name="Advanced Security Validation",
                passed=high_score,
                score=gate.score,
                message="No high/critical findings" if high_score else "Security concerns present",
                source_gate=gate.gate_name,
                details={"source_implementation": source_impl},
            )
        )
        enterprise_passed = high_score and len(gate.findings) == 0
        if len(gate.findings) == 0 and not high_score:
            enterprise_message = f"Score {gate.score:.2f} below threshold {adv_security_threshold}"
        elif len(gate.findings) == 0:
            enterprise_message = "No security findings"
        else:
            enterprise_message = f"{len(gate.findings)} findings require review"
        checks.append(
            CheckResult(
                check_id=CheckId.ENTERPRISE_SECURITY_REVIEW,
                name="Enterprise Security Review",
                passed=enterprise_passed,
                score=1.0 if len(gate.findings) == 0 else max(0, 1 - len(gate.findings) * 0.1),
                message=enterprise_message,
                source_gate=gate.gate_name,
                details={"source_implementation": source_impl},
            )
        )

    elif gate.gate_type == GateType.QUALITY:
        checks.append(
            CheckResult(
                check_id=CheckId.CONTENT_QUALITY_REVIEW,
                name="Content Quality Review",
                passed=gate.passed,
                score=gate.score,
                message=gate.message or "",
                source_gate=gate.gate_name,
                details={"source_implementation": source_impl},
            )
        )
        instruction_threshold = _get_threshold(
            CheckId.INSTRUCTION_QUALITY, threshold_overrides
        )
        checks.append(
            CheckResult(
                check_id=CheckId.INSTRUCTION_QUALITY,
                name="Instruction Quality",
                passed=gate.score >= instruction_threshold,
                score=gate.score,
                message="Quality review passed" if gate.score >= instruction_threshold else "Quality below threshold",
                source_gate=gate.gate_name,
                details={"source_implementation": source_impl},
            )
        )

    return checks


def _validate_check_ids(check_ids: list[str]) -> list[CheckId]:
    """Validate and convert string check IDs to CheckId enums.

    Args:
        check_ids: List of check ID strings

    Returns:
        List of validated CheckId enums

    Raises:
        ValueError: If any check ID is not a valid CheckId
    """
    valid_ids = {c.value for c in CheckId}
    result = []
    for check_id in check_ids:
        if check_id not in valid_ids:
            raise ValueError(
                f"Invalid check ID '{check_id}'. Valid IDs are: {sorted(valid_ids)}"
            )
        result.append(CheckId(check_id))
    return result


def _collect_threshold_overrides(
    policy: "CertificationPolicy | None",
) -> dict[str, float]:
    """Collect all threshold overrides from a certification policy."""
    if policy is None:
        return {}

    overrides: dict[str, float] = {}
    for level_policy in [policy.foundational, policy.trusted, policy.certified]:
        if level_policy is not None and level_policy.thresholds is not None:
            overrides.update(level_policy.thresholds)
    return overrides


def compute_certification(
    gates: list[GateResult],
    validation_passed: bool = True,
    metadata_valid: bool = True,
    has_eval_assets: bool = True,
    policy: "CertificationPolicy | None" = None,
) -> CertificationResult:
    """Compute certification levels from gate results.

    Args:
        gates: List of gate results from evaluation
        validation_passed: Whether structural validation passed
        metadata_valid: Whether metadata.yaml validation passed
        has_eval_assets: Whether evaluation assets (evals.json, tests) exist
        policy: Optional certification policy for custom checks and thresholds

    Returns:
        Complete certification result with all levels
    """
    threshold_overrides = _collect_threshold_overrides(policy)
    all_checks: dict[CheckId, CheckResult] = {}

    all_checks[CheckId.VALID_SKILL_STRUCTURE] = CheckResult(
        check_id=CheckId.VALID_SKILL_STRUCTURE,
        name="Valid Skill Structure",
        passed=validation_passed,
        score=1.0 if validation_passed else 0.0,
        message="Structure validation passed" if validation_passed else "Structure validation failed",
    )

    all_checks[CheckId.METADATA_COMPLIANCE] = CheckResult(
        check_id=CheckId.METADATA_COMPLIANCE,
        name="Metadata Compliance",
        passed=metadata_valid,
        score=1.0 if metadata_valid else 0.0,
        message="Metadata schema valid" if metadata_valid else "Metadata schema invalid",
    )

    all_checks[CheckId.EVALUATION_ASSETS] = CheckResult(
        check_id=CheckId.EVALUATION_ASSETS,
        name="Evaluation Assets",
        passed=has_eval_assets,
        score=1.0 if has_eval_assets else 0.0,
        message="Evaluation assets present" if has_eval_assets else "Missing evaluation assets",
    )

    for gate in gates:
        for check in _map_gate_to_checks(gate, validation_passed, threshold_overrides):
            if check.check_id not in all_checks:
                all_checks[check.check_id] = check
            elif not check.passed and all_checks[check.check_id].passed:
                # New check fails while existing passes - keep the failing one
                # (conservative: any failure from any source should block certification)
                all_checks[check.check_id] = check

    all_checks.setdefault(
        CheckId.ENTERPRISE_STRUCTURE_VALIDATION,
        CheckResult(
            check_id=CheckId.ENTERPRISE_STRUCTURE_VALIDATION,
            name="Enterprise Structure Validation",
            passed=validation_passed,
            score=1.0 if validation_passed else 0.0,
            message="Derived from structure validation",
        ),
    )

    all_checks.setdefault(
        CheckId.REGISTRY_GOVERNANCE,
        CheckResult(
            check_id=CheckId.REGISTRY_GOVERNANCE,
            name="Registry Governance",
            passed=False,
            score=0.0,
            message="Registry governance check not implemented",
        ),
    )

    all_checks.setdefault(
        CheckId.OPERATIONAL_POLICY_COMPLIANCE,
        CheckResult(
            check_id=CheckId.OPERATIONAL_POLICY_COMPLIANCE,
            name="Operational Policy Compliance",
            passed=False,
            score=0.0,
            message="Operational policy check not implemented",
        ),
    )

    all_checks.setdefault(
        CheckId.ENTERPRISE_BEHAVIORAL_TESTING,
        CheckResult(
            check_id=CheckId.ENTERPRISE_BEHAVIORAL_TESTING,
            name="Enterprise Behavioral Testing",
            passed=False,
            score=0.0,
            message="Behavioral testing not implemented",
        ),
    )

    all_checks.setdefault(
        CheckId.CONTINUOUS_OPTIMIZATION,
        CheckResult(
            check_id=CheckId.CONTINUOUS_OPTIMIZATION,
            name="Continuous Optimization",
            passed=False,
            score=0.0,
            message="Continuous optimization not implemented",
        ),
    )

    def get_checks_for_level(
        level_name: str,
        default_checks: list[CheckId],
    ) -> list[CheckId]:
        """Get check IDs for a level, using policy override if specified."""
        if policy is not None:
            custom_checks = policy.get_checks_for_level(level_name)
            if custom_checks is not None:
                return _validate_check_ids(custom_checks)
        return default_checks

    def build_level(
        level: CertificationLevel,
        level_name: str,
        default_check_ids: list[CheckId],
    ) -> LevelResult:
        check_ids = get_checks_for_level(level_name, default_check_ids)
        level_checks = []
        for cid in check_ids:
            if cid in all_checks:
                level_checks.append(all_checks[cid])
            else:
                level_checks.append(
                    CheckResult(
                        check_id=cid,
                        name=cid.value.replace("_", " ").title(),
                        passed=False,
                        score=0.0,
                        message="Check not evaluated",
                    )
                )
        if len(level_checks) == 0:
            logger.warning(
                "Certification level '%s' has no checks configured; level will not pass",
                level_name,
            )
        all_passed = len(level_checks) > 0 and all(c.passed for c in level_checks)
        return LevelResult(level=level, passed=all_passed, checks=level_checks)

    return CertificationResult(
        foundational=build_level(
            CertificationLevel.FOUNDATIONAL, "foundational", FOUNDATIONAL_CHECKS
        ),
        trusted=build_level(CertificationLevel.TRUSTED, "trusted", TRUSTED_CHECKS),
        certified=build_level(
            CertificationLevel.CERTIFIED, "certified", CERTIFIED_CHECKS
        ),
    )
