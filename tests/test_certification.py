"""Tests for certification level computation."""

from __future__ import annotations

import pytest

from abevalflow.certification import (
    CERTIFIED_CHECKS,
    FOUNDATIONAL_CHECKS,
    TRUSTED_CHECKS,
    CertificationLevel,
    CertificationResult,
    CheckId,
    CheckResult,
    LevelResult,
    compute_certification,
)
from abevalflow.gates.base import Finding, GateMode, GateResult, GateType, Severity


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

    def test_all_check_ids_covered(self) -> None:
        """Verify all check IDs are accounted for in level definitions."""
        all_level_checks = set(FOUNDATIONAL_CHECKS + TRUSTED_CHECKS + CERTIFIED_CHECKS)
        all_check_ids = set(CheckId)
        assert all_check_ids == all_level_checks


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
        assert len(TRUSTED_CHECKS) == 6

    def test_certified_has_expected_checks(self) -> None:
        assert CheckId.ENTERPRISE_STRUCTURE_VALIDATION in CERTIFIED_CHECKS
        assert CheckId.ENTERPRISE_SECURITY_REVIEW in CERTIFIED_CHECKS
        assert CheckId.ADVANCED_AGENT_VALIDATION in CERTIFIED_CHECKS
        assert len(CERTIFIED_CHECKS) == 5
