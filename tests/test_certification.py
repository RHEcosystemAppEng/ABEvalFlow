"""Tests for certification level computation."""

from __future__ import annotations

import pytest

from abevalflow.certification import (
    CERTIFIED_CHECKS,
    DEFAULT_THRESHOLDS,
    FOUNDATIONAL_CHECKS,
    TRUSTED_CHECKS,
    CertificationLevel,
    CertificationResult,
    CheckId,
    CheckResult,
    LevelResult,
    _validate_check_ids,
    compute_certification,
)
from abevalflow.gates.base import Finding, GateMode, GateResult, GateType, Severity
from abevalflow.schemas import CertificationLevelPolicy, CertificationPolicy


class TestCheckResult:
    """Tests for CheckResult model."""

    def test_basic_check_result(self) -> None:
        result = CheckResult(
            check_id=CheckId.VALID_SKILL_STRUCTURE,
            name="Valid Skill Structure",
            passed=True,
            score=1.0,
            message="All checks passed",
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_failed_check_result(self) -> None:
        result = CheckResult(
            check_id=CheckId.BASIC_SECURITY_VALIDATION,
            name="Basic Security Validation",
            passed=False,
            score=0.3,
            message="High severity findings detected",
            source_gate="security",
        )
        assert result.passed is False
        assert result.source_gate == "security"


class TestLevelResult:
    """Tests for LevelResult model."""

    def test_all_checks_passed(self) -> None:
        checks = [
            CheckResult(
                check_id=CheckId.VALID_SKILL_STRUCTURE,
                name="Test 1",
                passed=True,
                score=1.0,
            ),
            CheckResult(
                check_id=CheckId.BASIC_SECURITY_VALIDATION,
                name="Test 2",
                passed=True,
                score=0.8,
            ),
        ]
        level = LevelResult(
            level=CertificationLevel.FOUNDATIONAL,
            passed=True,
            checks=checks,
        )
        assert level.checks_passed == 2
        assert level.checks_total == 2
        assert level.overall_score == 0.9

    def test_some_checks_failed(self) -> None:
        checks = [
            CheckResult(
                check_id=CheckId.VALID_SKILL_STRUCTURE,
                name="Test 1",
                passed=True,
                score=1.0,
            ),
            CheckResult(
                check_id=CheckId.BASIC_SECURITY_VALIDATION,
                name="Test 2",
                passed=False,
                score=0.4,
            ),
        ]
        level = LevelResult(
            level=CertificationLevel.FOUNDATIONAL,
            passed=False,
            checks=checks,
        )
        assert level.checks_passed == 1
        assert level.checks_total == 2
        assert level.overall_score == 0.7


class TestCertificationResult:
    """Tests for CertificationResult model."""

    def test_highest_level_certified(self) -> None:
        result = CertificationResult(
            foundational=LevelResult(level=CertificationLevel.FOUNDATIONAL, passed=True),
            trusted=LevelResult(level=CertificationLevel.TRUSTED, passed=True),
            certified=LevelResult(level=CertificationLevel.CERTIFIED, passed=True),
        )
        assert result.highest_level == CertificationLevel.CERTIFIED

    def test_highest_level_trusted(self) -> None:
        result = CertificationResult(
            foundational=LevelResult(level=CertificationLevel.FOUNDATIONAL, passed=True),
            trusted=LevelResult(level=CertificationLevel.TRUSTED, passed=True),
            certified=LevelResult(level=CertificationLevel.CERTIFIED, passed=False),
        )
        assert result.highest_level == CertificationLevel.TRUSTED

    def test_highest_level_foundational(self) -> None:
        result = CertificationResult(
            foundational=LevelResult(level=CertificationLevel.FOUNDATIONAL, passed=True),
            trusted=LevelResult(level=CertificationLevel.TRUSTED, passed=False),
            certified=LevelResult(level=CertificationLevel.CERTIFIED, passed=False),
        )
        assert result.highest_level == CertificationLevel.FOUNDATIONAL

    def test_highest_level_none(self) -> None:
        result = CertificationResult(
            foundational=LevelResult(level=CertificationLevel.FOUNDATIONAL, passed=False),
            trusted=LevelResult(level=CertificationLevel.TRUSTED, passed=False),
            certified=LevelResult(level=CertificationLevel.CERTIFIED, passed=False),
        )
        assert result.highest_level == CertificationLevel.NONE

    def test_hierarchical_requirement_certified_needs_trusted(self) -> None:
        result = CertificationResult(
            foundational=LevelResult(level=CertificationLevel.FOUNDATIONAL, passed=True),
            trusted=LevelResult(level=CertificationLevel.TRUSTED, passed=False),
            certified=LevelResult(level=CertificationLevel.CERTIFIED, passed=True),
        )
        assert result.highest_level == CertificationLevel.FOUNDATIONAL

    def test_get_level_result(self) -> None:
        foundational = LevelResult(level=CertificationLevel.FOUNDATIONAL, passed=True)
        trusted = LevelResult(level=CertificationLevel.TRUSTED, passed=False)
        certified = LevelResult(level=CertificationLevel.CERTIFIED, passed=False)
        result = CertificationResult(
            foundational=foundational,
            trusted=trusted,
            certified=certified,
        )
        assert result.get_level_result(CertificationLevel.FOUNDATIONAL) == foundational
        assert result.get_level_result(CertificationLevel.TRUSTED) == trusted
        assert result.get_level_result(CertificationLevel.CERTIFIED) == certified
        assert result.get_level_result(CertificationLevel.NONE) is None


class TestComputeCertification:
    """Tests for compute_certification function."""

    def test_no_gates_minimal_certification(self) -> None:
        result = compute_certification(
            gates=[],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
        )
        assert result.foundational.passed is False
        assert result.highest_level == CertificationLevel.NONE

    def test_engine_gate_provides_execution_checks(self) -> None:
        engine_gate = GateResult(
            gate_name="evaluation",
            gate_type=GateType.ENGINE,
            passed=True,
            score=0.85,
            mode=GateMode.BLOCK,
            message="85% pass rate",
        )
        result = compute_certification(
            gates=[engine_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
        )
        foundational_check_ids = [c.check_id for c in result.foundational.checks]
        assert CheckId.BASIC_EXECUTION_VALIDATION in foundational_check_ids

        trusted_check_ids = [c.check_id for c in result.trusted.checks]
        assert CheckId.FUNCTIONAL_VALIDATION in trusted_check_ids

    def test_security_gate_provides_security_checks(self) -> None:
        security_gate = GateResult(
            gate_name="security",
            gate_type=GateType.SECURITY,
            passed=True,
            score=0.95,
            mode=GateMode.BLOCK,
            message="No critical findings",
            findings=[],
        )
        result = compute_certification(
            gates=[security_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
        )
        foundational_check_ids = [c.check_id for c in result.foundational.checks]
        assert CheckId.BASIC_SECURITY_VALIDATION in foundational_check_ids

        trusted_check_ids = [c.check_id for c in result.trusted.checks]
        assert CheckId.ADVANCED_SECURITY_VALIDATION in trusted_check_ids

    def test_quality_gate_provides_quality_checks(self) -> None:
        quality_gate = GateResult(
            gate_name="quality",
            gate_type=GateType.QUALITY,
            passed=True,
            score=0.8,
            mode=GateMode.WARN,
            message="Quality review passed",
        )
        result = compute_certification(
            gates=[quality_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
        )
        foundational_check_ids = [c.check_id for c in result.foundational.checks]
        assert CheckId.CONTENT_QUALITY_REVIEW in foundational_check_ids

    def test_validation_failure_affects_certification(self) -> None:
        result = compute_certification(
            gates=[],
            validation_passed=False,
            metadata_valid=True,
            has_eval_assets=True,
        )
        valid_structure_check = next(
            c for c in result.foundational.checks
            if c.check_id == CheckId.VALID_SKILL_STRUCTURE
        )
        assert valid_structure_check.passed is False

    def test_missing_metadata_affects_certification(self) -> None:
        result = compute_certification(
            gates=[],
            validation_passed=True,
            metadata_valid=False,
            has_eval_assets=True,
        )
        metadata_check = next(
            c for c in result.foundational.checks
            if c.check_id == CheckId.METADATA_COMPLIANCE
        )
        assert metadata_check.passed is False

    def test_missing_eval_assets_affects_trusted(self) -> None:
        result = compute_certification(
            gates=[],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=False,
        )
        eval_assets_check = next(
            c for c in result.trusted.checks
            if c.check_id == CheckId.EVALUATION_ASSETS
        )
        assert eval_assets_check.passed is False

    def test_high_score_engine_provides_advanced_validation(self) -> None:
        engine_gate = GateResult(
            gate_name="evaluation",
            gate_type=GateType.ENGINE,
            passed=True,
            score=0.90,
            mode=GateMode.BLOCK,
        )
        result = compute_certification(
            gates=[engine_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
        )
        advanced_check = next(
            c for c in result.certified.checks
            if c.check_id == CheckId.ADVANCED_AGENT_VALIDATION
        )
        assert advanced_check.passed is True

    def test_low_score_engine_fails_advanced_validation(self) -> None:
        engine_gate = GateResult(
            gate_name="evaluation",
            gate_type=GateType.ENGINE,
            passed=True,
            score=0.60,
            mode=GateMode.BLOCK,
        )
        result = compute_certification(
            gates=[engine_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
        )
        advanced_check = next(
            c for c in result.certified.checks
            if c.check_id == CheckId.ADVANCED_AGENT_VALIDATION
        )
        assert advanced_check.passed is False

    def test_security_findings_affect_enterprise_review(self) -> None:
        findings = [
            Finding(
                rule_id="test-rule",
                severity=Severity.HIGH,
                message="Test finding",
            )
        ]
        security_gate = GateResult(
            gate_name="security",
            gate_type=GateType.SECURITY,
            passed=False,
            score=0.5,
            mode=GateMode.BLOCK,
            findings=findings,
        )
        result = compute_certification(
            gates=[security_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
        )
        enterprise_check = next(
            c for c in result.certified.checks
            if c.check_id == CheckId.ENTERPRISE_SECURITY_REVIEW
        )
        assert enterprise_check.passed is False

    def test_default_checks_are_valid_check_ids(self) -> None:
        """Verify all default checks are valid CheckId values."""
        all_level_checks = set(FOUNDATIONAL_CHECKS + TRUSTED_CHECKS + CERTIFIED_CHECKS)
        all_check_ids = set(CheckId)
        # All default checks should be valid CheckIds (subset)
        assert all_level_checks.issubset(all_check_ids)


class TestCertificationLevelConstants:
    """Tests for certification level check constants."""

    def test_foundational_has_expected_checks(self) -> None:
        assert CheckId.VALID_SKILL_STRUCTURE in FOUNDATIONAL_CHECKS
        assert CheckId.BASIC_SECURITY_VALIDATION in FOUNDATIONAL_CHECKS
        assert CheckId.BASIC_EXECUTION_VALIDATION in FOUNDATIONAL_CHECKS
        assert CheckId.CONTENT_QUALITY_REVIEW in FOUNDATIONAL_CHECKS
        assert CheckId.METADATA_COMPLIANCE in FOUNDATIONAL_CHECKS
        assert len(FOUNDATIONAL_CHECKS) == 5

    def test_trusted_has_expected_checks(self) -> None:
        assert CheckId.EVALUATION_ASSETS in TRUSTED_CHECKS
        assert CheckId.ADVANCED_SECURITY_VALIDATION in TRUSTED_CHECKS
        assert CheckId.FUNCTIONAL_VALIDATION in TRUSTED_CHECKS
        assert CheckId.INSTRUCTION_QUALITY in TRUSTED_CHECKS
        # registry_governance and operational_policy_compliance not yet implemented
        assert len(TRUSTED_CHECKS) == 4

    def test_certified_has_expected_checks(self) -> None:
        assert CheckId.ENTERPRISE_STRUCTURE_VALIDATION in CERTIFIED_CHECKS
        assert CheckId.ENTERPRISE_SECURITY_REVIEW in CERTIFIED_CHECKS
        assert CheckId.ADVANCED_AGENT_VALIDATION in CERTIFIED_CHECKS
        # behavioral_testing and continuous_optimization not yet implemented
        assert len(CERTIFIED_CHECKS) == 3


class TestCertificationPolicy:
    """Tests for CertificationPolicy configuration."""

    def test_empty_policy_uses_defaults(self) -> None:
        """Empty policy should use default checks for all levels."""
        policy = CertificationPolicy()
        result = compute_certification(
            gates=[],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        assert len(result.foundational.checks) == len(FOUNDATIONAL_CHECKS)
        assert len(result.trusted.checks) == len(TRUSTED_CHECKS)
        assert len(result.certified.checks) == len(CERTIFIED_CHECKS)

    def test_none_policy_uses_defaults(self) -> None:
        """None policy should use default checks for all levels."""
        result = compute_certification(
            gates=[],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=None,
        )
        assert len(result.foundational.checks) == len(FOUNDATIONAL_CHECKS)
        assert len(result.trusted.checks) == len(TRUSTED_CHECKS)
        assert len(result.certified.checks) == len(CERTIFIED_CHECKS)

    def test_custom_checks_for_foundational(self) -> None:
        """Custom check list for foundational level."""
        policy = CertificationPolicy(
            foundational=CertificationLevelPolicy(
                checks=[
                    "valid_skill_structure",
                    "basic_security_validation",
                ]
            )
        )
        result = compute_certification(
            gates=[],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        assert len(result.foundational.checks) == 2
        check_ids = [c.check_id for c in result.foundational.checks]
        assert CheckId.VALID_SKILL_STRUCTURE in check_ids
        assert CheckId.BASIC_SECURITY_VALIDATION in check_ids
        assert CheckId.METADATA_COMPLIANCE not in check_ids

    def test_custom_checks_for_certified(self) -> None:
        """Custom check list for certified level (single check)."""
        policy = CertificationPolicy(
            certified=CertificationLevelPolicy(
                checks=["advanced_agent_validation"]
            )
        )
        engine_gate = GateResult(
            gate_name="evaluation",
            gate_type=GateType.ENGINE,
            passed=True,
            score=0.85,
            mode=GateMode.BLOCK,
        )
        result = compute_certification(
            gates=[engine_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        assert len(result.certified.checks) == 1
        assert result.certified.checks[0].check_id == CheckId.ADVANCED_AGENT_VALIDATION
        assert result.certified.passed is True

    def test_threshold_override_for_advanced_agent_validation(self) -> None:
        """Override threshold for advanced_agent_validation."""
        policy = CertificationPolicy(
            certified=CertificationLevelPolicy(
                thresholds={"advanced_agent_validation": 0.5}
            )
        )
        engine_gate = GateResult(
            gate_name="evaluation",
            gate_type=GateType.ENGINE,
            passed=True,
            score=0.6,
            mode=GateMode.BLOCK,
        )
        result = compute_certification(
            gates=[engine_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        advanced_check = next(
            c for c in result.certified.checks
            if c.check_id == CheckId.ADVANCED_AGENT_VALIDATION
        )
        assert advanced_check.passed is True

    def test_threshold_override_fails_when_below(self) -> None:
        """Score below overridden threshold should fail."""
        policy = CertificationPolicy(
            certified=CertificationLevelPolicy(
                thresholds={"advanced_agent_validation": 0.9}
            )
        )
        engine_gate = GateResult(
            gate_name="evaluation",
            gate_type=GateType.ENGINE,
            passed=True,
            score=0.85,
            mode=GateMode.BLOCK,
        )
        result = compute_certification(
            gates=[engine_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        advanced_check = next(
            c for c in result.certified.checks
            if c.check_id == CheckId.ADVANCED_AGENT_VALIDATION
        )
        assert advanced_check.passed is False
        assert "0.90" in advanced_check.message

    def test_partial_policy_only_foundational(self) -> None:
        """Partial policy with only foundational specified."""
        policy = CertificationPolicy(
            foundational=CertificationLevelPolicy(
                checks=[
                    "valid_skill_structure",
                    "metadata_compliance",
                ]
            )
        )
        result = compute_certification(
            gates=[],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        assert len(result.foundational.checks) == 2
        assert result.foundational.passed is True
        assert len(result.trusted.checks) == len(TRUSTED_CHECKS)
        assert len(result.certified.checks) == len(CERTIFIED_CHECKS)

    def test_invalid_check_id_raises_error(self) -> None:
        """Invalid check ID should raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _validate_check_ids(["invalid_check_id"])
        assert "Invalid check ID 'invalid_check_id'" in str(exc_info.value)

    def test_threshold_override_for_instruction_quality(self) -> None:
        """Override threshold for instruction_quality check."""
        policy = CertificationPolicy(
            trusted=CertificationLevelPolicy(
                thresholds={"instruction_quality": 0.5}
            )
        )
        quality_gate = GateResult(
            gate_name="quality",
            gate_type=GateType.QUALITY,
            passed=True,
            score=0.6,
            mode=GateMode.WARN,
        )
        result = compute_certification(
            gates=[quality_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        instruction_check = next(
            c for c in result.trusted.checks
            if c.check_id == CheckId.INSTRUCTION_QUALITY
        )
        assert instruction_check.passed is True

    def test_threshold_override_for_advanced_security(self) -> None:
        """Override threshold for advanced_security_validation check."""
        policy = CertificationPolicy(
            trusted=CertificationLevelPolicy(
                thresholds={"advanced_security_validation": 0.7}
            )
        )
        security_gate = GateResult(
            gate_name="security",
            gate_type=GateType.SECURITY,
            passed=True,
            score=0.75,
            mode=GateMode.BLOCK,
            findings=[],
        )
        result = compute_certification(
            gates=[security_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        adv_security_check = next(
            c for c in result.trusted.checks
            if c.check_id == CheckId.ADVANCED_SECURITY_VALIDATION
        )
        assert adv_security_check.passed is True

    def test_custom_checks_empty_list(self) -> None:
        """Empty check list should result in empty level."""
        policy = CertificationPolicy(
            foundational=CertificationLevelPolicy(checks=[])
        )
        result = compute_certification(
            gates=[],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        assert len(result.foundational.checks) == 0
        assert result.foundational.passed is True

    def test_combined_custom_checks_and_thresholds(self) -> None:
        """Custom checks and thresholds together."""
        policy = CertificationPolicy(
            foundational=CertificationLevelPolicy(
                checks=["basic_execution_validation"],
                thresholds={"basic_execution_validation": 0.3},
            )
        )
        engine_gate = GateResult(
            gate_name="evaluation",
            gate_type=GateType.ENGINE,
            passed=True,
            score=0.4,
            mode=GateMode.BLOCK,
        )
        result = compute_certification(
            gates=[engine_gate],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            policy=policy,
        )
        assert len(result.foundational.checks) == 1
        assert result.foundational.checks[0].check_id == CheckId.BASIC_EXECUTION_VALIDATION
        assert result.foundational.passed is True


class TestDefaultThresholds:
    """Tests for default threshold constants."""

    def test_default_thresholds_exist(self) -> None:
        """Verify default thresholds are defined."""
        assert CheckId.ADVANCED_AGENT_VALIDATION in DEFAULT_THRESHOLDS
        assert CheckId.ADVANCED_SECURITY_VALIDATION in DEFAULT_THRESHOLDS
        assert CheckId.INSTRUCTION_QUALITY in DEFAULT_THRESHOLDS

    def test_default_threshold_values(self) -> None:
        """Verify default threshold values."""
        assert DEFAULT_THRESHOLDS[CheckId.ADVANCED_AGENT_VALIDATION] == 0.8
        assert DEFAULT_THRESHOLDS[CheckId.ADVANCED_SECURITY_VALIDATION] == 0.9
        assert DEFAULT_THRESHOLDS[CheckId.INSTRUCTION_QUALITY] == 0.7
