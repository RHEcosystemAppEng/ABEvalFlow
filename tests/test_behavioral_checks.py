"""Tests for behavioral testing certification checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abevalflow.certification import (
    CheckId,
    CheckResult,
    _check_consistency,
    _check_edge_case_results,
    _check_failure_mode,
    _check_stability,
    _compute_behavioral_testing_check,
    compute_certification,
)
from abevalflow.gates.base import GateMode, GateResult, GateType
from abevalflow.gates.behavioral.edge_case import EdgeCaseGate
from abevalflow.harbor_agents.verifiers.llm_judge import CRITERIA_DESCRIPTIONS
from abevalflow.report import EdgeCaseResult, VariantSummary
from abevalflow.schemas import GatePolicy, GatePolicyItem


class TestConsistencyCheck:
    """Tests for _check_consistency() — trial variance detection."""

    def test_low_variance_passes(self) -> None:
        result = _check_consistency({"std_reward": 0.1})
        assert result.passed is True
        assert result.score > 0.5

    def test_high_variance_fails(self) -> None:
        result = _check_consistency({"std_reward": 0.5})
        assert result.passed is False
        assert result.score < 0.5

    def test_zero_variance_passes(self) -> None:
        result = _check_consistency({"std_reward": 0.0})
        assert result.passed is True
        assert result.score == 1.0

    def test_at_threshold_score_is_half(self) -> None:
        result = _check_consistency({"std_reward": 0.3})
        assert result.passed is True
        assert result.score == pytest.approx(0.5)

    def test_none_variance_passes(self) -> None:
        result = _check_consistency({"std_reward": None})
        assert result.passed is True
        assert result.score == 1.0

    def test_missing_variance_passes(self) -> None:
        result = _check_consistency({})
        assert result.passed is True
        assert result.score == 1.0

    def test_custom_threshold(self) -> None:
        result = _check_consistency({"std_reward": 0.15}, threshold=0.1)
        assert result.passed is False

        result = _check_consistency({"std_reward": 0.05}, threshold=0.1)
        assert result.passed is True

    def test_score_clamped(self) -> None:
        result = _check_consistency({"std_reward": 1.0})
        assert 0.0 <= result.score <= 1.0

    def test_details_contain_threshold_and_status(self) -> None:
        result = _check_consistency({"std_reward": 0.2})
        assert result.details["std_reward"] == 0.2
        assert "status" not in result.details

    def test_no_data_has_status_sentinel(self) -> None:
        result = _check_consistency({})
        assert result.details["status"] == "no_data"

    def test_low_mean_low_variance_passes_consistency(self) -> None:
        """Consistency only measures reliability, not quality."""
        result = _check_consistency({"std_reward": 0.02})
        assert result.passed is True

    def test_high_mean_high_variance_fails_consistency(self) -> None:
        """Good average doesn't help if results are unpredictable."""
        result = _check_consistency({"std_reward": 0.4})
        assert result.passed is False


class TestStabilityCheck:
    """Tests for _check_stability() — score drift detection."""

    def test_stable_scores_pass(self) -> None:
        result = _check_stability(
            {
                "stability": {"score_variance": 0.01, "run_count": 5},
            }
        )
        assert result.passed is True
        assert result.score > 0.5

    def test_drifting_scores_fail(self) -> None:
        result = _check_stability(
            {
                "stability": {"score_variance": 0.2, "run_count": 5},
            }
        )
        assert result.passed is False
        assert result.score < 0.5

    def test_insufficient_history_passes(self) -> None:
        result = _check_stability(
            {
                "stability": {"score_variance": 0.0, "run_count": 2},
            }
        )
        assert result.passed is True
        assert "Insufficient history" in result.message

    def test_no_stability_data_passes(self) -> None:
        result = _check_stability({})
        assert result.passed is True
        assert result.score == 1.0

    def test_none_stability_passes(self) -> None:
        result = _check_stability({"stability": None})
        assert result.passed is True

    def test_at_threshold_score_is_half(self) -> None:
        result = _check_stability(
            {
                "stability": {"score_variance": 0.1, "run_count": 5},
            }
        )
        assert result.passed is True
        assert result.score == pytest.approx(0.5)

    def test_no_data_has_status_sentinel(self) -> None:
        result = _check_stability({})
        assert result.details["status"] == "no_data"

    def test_insufficient_history_has_status_sentinel(self) -> None:
        result = _check_stability(
            {
                "stability": {"score_variance": 0.0, "run_count": 1},
            }
        )
        assert result.details["status"] == "no_data"


class TestEdgeCaseCheck:
    """Tests for _check_edge_case_results() — edge case pass rate."""

    def test_all_passed(self) -> None:
        result = _check_edge_case_results(
            {
                "edge_cases": {"total": 3, "passed": 3},
            }
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_some_failed(self) -> None:
        result = _check_edge_case_results(
            {
                "edge_cases": {"total": 4, "passed": 1},
            }
        )
        assert result.passed is False
        assert result.score == 0.25

    def test_all_failed(self) -> None:
        result = _check_edge_case_results(
            {
                "edge_cases": {"total": 3, "passed": 0},
            }
        )
        assert result.passed is False
        assert result.score == 0.0

    def test_no_edge_cases_defined(self) -> None:
        result = _check_edge_case_results(
            {
                "edge_cases": {"total": 0, "passed": 0},
            }
        )
        assert result.passed is True
        assert result.score == 1.0

    def test_no_edge_case_data(self) -> None:
        result = _check_edge_case_results({})
        assert result.passed is True
        assert result.score == 1.0

    def test_details_contain_counts(self) -> None:
        result = _check_edge_case_results(
            {
                "edge_cases": {"total": 5, "passed": 3},
            }
        )
        assert result.details["total"] == 5
        assert result.details["passed"] == 3
        assert result.details["pass_rate"] == 0.6


class TestFailureModeCheck:
    """Tests for _check_failure_mode() — failure handling scores."""

    def test_good_score_passes(self) -> None:
        result = _check_failure_mode(
            {
                "failure_mode": {"score": 0.8, "threshold": 0.5},
            }
        )
        assert result.passed is True
        assert result.score == 0.8

    def test_low_score_fails(self) -> None:
        result = _check_failure_mode(
            {
                "failure_mode": {"score": 0.3, "threshold": 0.5},
            }
        )
        assert result.passed is False
        assert result.score == 0.3

    def test_no_data_passes(self) -> None:
        result = _check_failure_mode({})
        assert result.passed is True
        assert result.score == 1.0


class TestBehavioralTestingCombined:
    """Tests for _compute_behavioral_testing_check() — combined check."""

    def test_all_sub_checks_pass(self) -> None:
        data = {
            "std_reward": 0.1,
            "edge_cases": {"total": 3, "passed": 3},
            "stability": {"score_variance": 0.01, "run_count": 5},
            "failure_mode": {"score": 0.8, "threshold": 0.5},
        }
        result = _compute_behavioral_testing_check(data)
        assert result.passed is True
        assert result.score > 0.6

    def test_consistency_fails_means_overall_fails(self) -> None:
        data = {
            "std_reward": 0.9,
            "edge_cases": {"total": 3, "passed": 3},
            "stability": {"score_variance": 0.01, "run_count": 5},
            "failure_mode": {"score": 0.8, "threshold": 0.5},
        }
        result = _compute_behavioral_testing_check(data)
        assert result.passed is False

    def test_single_sub_check_insufficient(self) -> None:
        data = {"std_reward": 0.1}
        result = _compute_behavioral_testing_check(data)
        assert result.passed is False
        assert "Insufficient behavioral coverage" in result.message

    def test_two_sub_checks_sufficient(self) -> None:
        data = {
            "std_reward": 0.1,
            "edge_cases": {"total": 3, "passed": 3},
        }
        result = _compute_behavioral_testing_check(data)
        assert result.passed is True
        assert result.score > 0.0

    def test_no_data_fails(self) -> None:
        result = _compute_behavioral_testing_check({})
        assert result.passed is False
        assert result.score == 0.0
        assert "No behavioral testing data available" in result.message

    def test_details_contain_sub_checks(self) -> None:
        data = {
            "std_reward": 0.1,
            "edge_cases": {"total": 3, "passed": 3},
        }
        result = _compute_behavioral_testing_check(data)
        assert "sub_checks" in result.details
        assert "weights" in result.details

    def test_at_threshold_sub_checks_still_pass_composite(self) -> None:
        data = {
            "std_reward": 0.3,
            "edge_cases": {"total": 2, "passed": 1},
        }
        result = _compute_behavioral_testing_check(data)
        assert result.passed is True
        assert result.score == pytest.approx(0.5)


class TestCertificationWithBehavioralData:
    """Tests for compute_certification() with behavioral_data parameter."""

    def _make_all_gates(self) -> list[GateResult]:
        return [
            GateResult(
                gate_name="evaluation",
                gate_type=GateType.ENGINE,
                passed=True,
                score=0.9,
                mode=GateMode.BLOCK,
            ),
            GateResult(
                gate_name="security",
                gate_type=GateType.SECURITY,
                passed=True,
                score=1.0,
                mode=GateMode.BLOCK,
            ),
            GateResult(
                gate_name="quality",
                gate_type=GateType.QUALITY,
                passed=True,
                score=0.85,
                mode=GateMode.WARN,
            ),
        ]

    def test_behavioral_data_produces_passing_check(self) -> None:
        behavioral_data = {
            "std_reward": 0.1,
            "edge_cases": {"total": 3, "passed": 3},
        }
        result = _compute_behavioral_testing_check(behavioral_data)
        assert result.passed is True
        assert result.check_id == CheckId.ENTERPRISE_BEHAVIORAL_TESTING

    def test_no_behavioral_data_produces_failing_check(self) -> None:
        result = _compute_behavioral_testing_check({})
        assert result.passed is False

    def test_behavioral_check_in_certified_checks(self) -> None:
        from abevalflow.certification import CERTIFIED_CHECKS

        assert CheckId.ENTERPRISE_BEHAVIORAL_TESTING in CERTIFIED_CHECKS

    def _passing_operational_policy(self) -> CheckResult:
        return CheckResult(
            check_id=CheckId.OPERATIONAL_POLICY_COMPLIANCE,
            name="Operational Policy Compliance",
            passed=True,
            score=1.0,
            message="Passed",
        )

    def test_behavioral_data_required_for_certified(self) -> None:
        behavioral_data = {
            "std_reward": 0.1,
            "edge_cases": {"total": 3, "passed": 3},
        }
        result = compute_certification(
            gates=self._make_all_gates(),
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            operational_policy_result=self._passing_operational_policy(),
            behavioral_data=behavioral_data,
        )
        assert result.certified.passed is True

    def test_no_behavioral_data_blocks_certified(self) -> None:
        result = compute_certification(
            gates=self._make_all_gates(),
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            operational_policy_result=self._passing_operational_policy(),
        )
        assert result.foundational.passed is True
        assert result.trusted.passed is True
        assert result.certified.passed is False

    def test_low_mean_high_consistency_blocked_by_engine_gate(self) -> None:
        """A skill with low variance but poor absolute scores still fails Certified.

        Consistency passes (low std_reward), but ADVANCED_AGENT_VALIDATION
        requires engine score >= 0.8. This proves the two checks work together
        as a safety net.
        """
        low_score_engine = GateResult(
            gate_name="evaluation",
            gate_type=GateType.ENGINE,
            passed=True,
            score=0.3,
            mode=GateMode.BLOCK,
        )
        behavioral_data = {
            "std_reward": 0.02,
            "edge_cases": {"total": 3, "passed": 3},
        }
        result = compute_certification(
            gates=[
                low_score_engine,
                GateResult(
                    gate_name="security",
                    gate_type=GateType.SECURITY,
                    passed=True,
                    score=1.0,
                    mode=GateMode.BLOCK,
                ),
                GateResult(
                    gate_name="quality",
                    gate_type=GateType.QUALITY,
                    passed=True,
                    score=0.85,
                    mode=GateMode.WARN,
                ),
            ],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            operational_policy_result=self._passing_operational_policy(),
            behavioral_data=behavioral_data,
        )
        behavioral_check = next(
            (c for c in result.certified.checks if c.check_id == CheckId.ENTERPRISE_BEHAVIORAL_TESTING),
            None,
        )
        assert behavioral_check is not None
        assert behavioral_check.passed is True

        agent_check = next(
            (c for c in result.certified.checks if c.check_id == CheckId.ADVANCED_AGENT_VALIDATION),
            None,
        )
        assert agent_check is not None
        assert agent_check.passed is False

        assert result.certified.passed is False


class TestEdgeCaseGate:
    """Tests for EdgeCaseGate behavioral gate."""

    def test_no_report_json(self, tmp_path: Path) -> None:
        policy = GatePolicy()
        gate = EdgeCaseGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True
        assert result.gate_type == GateType.BEHAVIORAL

    def test_no_report_block_mode_fails(self, tmp_path: Path) -> None:
        policy = GatePolicy(gates={"behavioral": GatePolicyItem(mode=GateMode.BLOCK)})
        gate = EdgeCaseGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is False
        assert result.score == 0.0

    def test_disabled_mode(self, tmp_path: Path) -> None:
        policy = GatePolicy(gates={"behavioral": GatePolicyItem(mode=GateMode.DISABLED)})
        gate = EdgeCaseGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True
        assert result.mode == GateMode.DISABLED

    def test_all_edge_cases_pass(self, tmp_path: Path) -> None:
        report = {
            "edge_case_results": [
                {"name": "empty_input", "passed": True},
                {"name": "adversarial", "passed": True},
            ],
        }
        (tmp_path / "report.json").write_text(json.dumps(report))

        policy = GatePolicy(gates={"behavioral": GatePolicyItem(mode=GateMode.BLOCK)})
        gate = EdgeCaseGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True
        assert result.score == 1.0

    def test_some_edge_cases_fail(self, tmp_path: Path) -> None:
        report = {
            "edge_case_results": [
                {"name": "empty_input", "passed": True},
                {"name": "adversarial", "passed": False},
                {"name": "boundary", "passed": False},
            ],
        }
        (tmp_path / "report.json").write_text(json.dumps(report))

        policy = GatePolicy(gates={"behavioral": GatePolicyItem(mode=GateMode.BLOCK, threshold=0.5)})
        gate = EdgeCaseGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is False
        assert abs(result.score - 1 / 3) < 0.01

    def test_no_edge_case_results_in_report(self, tmp_path: Path) -> None:
        report = {"summary": {"treatment": {}}}
        (tmp_path / "report.json").write_text(json.dumps(report))

        policy = GatePolicy()
        gate = EdgeCaseGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True

    def test_warn_mode_always_passes(self, tmp_path: Path) -> None:
        report = {
            "edge_case_results": [
                {"name": "test", "passed": False},
            ],
        }
        (tmp_path / "report.json").write_text(json.dumps(report))

        policy = GatePolicy(gates={"behavioral": GatePolicyItem(mode=GateMode.WARN)})
        gate = EdgeCaseGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True
        assert result.score == 0.0


class TestEdgeCaseValidation:
    """Tests for edge case directory validation."""

    def test_no_edge_cases_dir_passes(self, tmp_path: Path) -> None:
        from scripts.validate import _check_edge_cases

        errors = _check_edge_cases(tmp_path)
        assert errors == []

    def test_valid_edge_cases(self, tmp_path: Path) -> None:
        from scripts.validate import _check_edge_cases

        edge_dir = tmp_path / "edge_cases"
        edge_dir.mkdir()
        (edge_dir / "empty_input.md").write_text("Test with empty input")
        (edge_dir / "adversarial.md").write_text("Test with adversarial prompt")
        errors = _check_edge_cases(tmp_path)
        assert errors == []

    def test_empty_md_file(self, tmp_path: Path) -> None:
        from scripts.validate import _check_edge_cases

        edge_dir = tmp_path / "edge_cases"
        edge_dir.mkdir()
        (edge_dir / "empty_input.md").write_text("")
        errors = _check_edge_cases(tmp_path)
        assert len(errors) == 1
        assert "empty" in errors[0]

    def test_non_md_files(self, tmp_path: Path) -> None:
        from scripts.validate import _check_edge_cases

        edge_dir = tmp_path / "edge_cases"
        edge_dir.mkdir()
        (edge_dir / "valid.md").write_text("Valid content")
        (edge_dir / "not_md.txt").write_text("Wrong extension")
        errors = _check_edge_cases(tmp_path)
        assert any("non-.md" in e for e in errors)

    def test_empty_dir(self, tmp_path: Path) -> None:
        from scripts.validate import _check_edge_cases

        edge_dir = tmp_path / "edge_cases"
        edge_dir.mkdir()
        errors = _check_edge_cases(tmp_path)
        assert any("no .md files" in e for e in errors)


class TestExtractBehavioralData:
    """Tests for _extract_behavioral_data() glue function."""

    def test_extracts_std_reward(self, tmp_path: Path) -> None:
        from scripts.aggregate_scorecard import _extract_behavioral_data

        report = {"summary": {"treatment": {"std_reward": 0.15}}}
        (tmp_path / "report.json").write_text(json.dumps(report))
        data = _extract_behavioral_data(tmp_path, gates=[])
        assert data is not None
        assert data["std_reward"] == 0.15

    def test_extracts_edge_cases_from_gate(self, tmp_path: Path) -> None:
        from scripts.aggregate_scorecard import _extract_behavioral_data

        gate = GateResult(
            gate_type=GateType.BEHAVIORAL,
            gate_name="edge-case",
            passed=True,
            score=0.5,
            details={"total": 4, "passed": 2},
        )
        data = _extract_behavioral_data(tmp_path, gates=[gate])
        assert data is not None
        assert data["edge_cases"] == {"total": 4, "passed": 2}

    def test_returns_none_for_empty_report(self, tmp_path: Path) -> None:
        from scripts.aggregate_scorecard import _extract_behavioral_data

        report = {"summary": {"treatment": {}}}
        (tmp_path / "report.json").write_text(json.dumps(report))
        data = _extract_behavioral_data(tmp_path, gates=[])
        assert data is None

    def test_returns_none_when_no_report_and_no_gates(self, tmp_path: Path) -> None:
        from scripts.aggregate_scorecard import _extract_behavioral_data

        data = _extract_behavioral_data(tmp_path, gates=[])
        assert data is None

    def test_combines_std_reward_and_gate_edge_cases(self, tmp_path: Path) -> None:
        from scripts.aggregate_scorecard import _extract_behavioral_data

        report = {"summary": {"treatment": {"std_reward": 0.2}}}
        (tmp_path / "report.json").write_text(json.dumps(report))
        gate = GateResult(
            gate_type=GateType.BEHAVIORAL,
            gate_name="edge-case",
            passed=True,
            score=1.0,
            details={"total": 3, "passed": 3},
        )
        data = _extract_behavioral_data(tmp_path, gates=[gate])
        assert data is not None
        assert data["std_reward"] == 0.2
        assert data["edge_cases"] == {"total": 3, "passed": 3}


class TestGenerateEdgeCaseEvals:
    """Tests for generate_edge_case_evals.py."""

    def test_generates_evals_from_edge_cases(self, tmp_path: Path) -> None:
        from scripts.generate_edge_case_evals import generate_edge_case_evals

        sub = tmp_path / "my-skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text("name: my-skill\n")
        edge_dir = sub / "edge_cases"
        edge_dir.mkdir()
        (edge_dir / "empty_input.md").write_text("Test with empty input")
        (edge_dir / "adversarial.md").write_text("Try to break the skill")

        result = generate_edge_case_evals(sub)
        assert result is not None
        assert result["skill_name"] == "my-skill"
        assert len(result["evals"]) == 2
        assert result["evals"][0]["id"] == "edge-adversarial"
        assert result["evals"][1]["id"] == "edge-empty_input"
        assert "assertions" in result["evals"][0]

    def test_returns_none_without_edge_cases(self, tmp_path: Path) -> None:
        from scripts.generate_edge_case_evals import generate_edge_case_evals

        sub = tmp_path / "my-skill"
        sub.mkdir()
        result = generate_edge_case_evals(sub)
        assert result is None

    def test_skips_empty_files(self, tmp_path: Path) -> None:
        from scripts.generate_edge_case_evals import generate_edge_case_evals

        sub = tmp_path / "my-skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text("name: my-skill\n")
        edge_dir = sub / "edge_cases"
        edge_dir.mkdir()
        (edge_dir / "empty.md").write_text("")
        (edge_dir / "valid.md").write_text("A valid edge case")

        result = generate_edge_case_evals(sub)
        assert result is not None
        assert len(result["evals"]) == 1
        assert result["evals"][0]["id"] == "edge-valid"

    def test_scaffold_does_not_create_edge_case_dirs(self, tmp_path: Path) -> None:
        from scripts.scaffold import scaffold_submission

        sub = tmp_path / "my-skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text("name: my-skill\n")
        (sub / "instruction.md").write_text("Main instruction")
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text("# My Skill")
        tests = sub / "tests"
        tests.mkdir()
        (tests / "test_outputs.py").write_text("pass")
        edge_dir = sub / "edge_cases"
        edge_dir.mkdir()
        (edge_dir / "empty_input.md").write_text("Test with empty input")

        output = tmp_path / "output"
        scaffold_submission(sub, output)

        edge_dirs = list(output.glob("tasks-treatment-edge-*"))
        assert edge_dirs == []


class TestEdgeCaseAggregation:
    """Tests for ASE-based edge case aggregation."""

    def test_edge_case_result_model_with_score(self) -> None:
        result = EdgeCaseResult(name="empty_input", passed=True, score=0.8)
        assert result.name == "empty_input"
        assert result.passed is True
        assert result.summary is None
        assert result.score == 0.8

    def test_edge_case_result_model_with_summary(self) -> None:
        result = EdgeCaseResult(
            name="empty_input",
            summary=VariantSummary(n_trials=5, n_passed=4, pass_rate=0.8, mean_reward=0.7),
            passed=True,
        )
        assert result.summary is not None
        assert result.summary.pass_rate == 0.8

    def test_aggregate_per_edge_case_results(self, tmp_path: Path) -> None:
        from scripts.aggregate_edge_case_evals import aggregate_edge_case_results

        edge_dir = tmp_path / "empty_input" / "iteration-1" / "eval-skill" / "with_skill"
        edge_dir.mkdir(parents=True)
        (edge_dir / "grading.json").write_text(
            json.dumps(
                {
                    "summary": {"passed": 3, "failed": 0, "total": 3, "pass_rate": 1.0},
                }
            )
        )

        results = aggregate_edge_case_results(tmp_path)
        assert len(results) == 1
        assert results[0]["name"] == "empty_input"
        assert results[0]["passed"] is True

    def test_missing_grading_counted_as_failure(self, tmp_path: Path) -> None:
        from scripts.aggregate_edge_case_evals import aggregate_edge_case_results

        (tmp_path / "empty_input").mkdir()
        results = aggregate_edge_case_results(tmp_path)
        assert len(results) == 1
        assert results[0]["name"] == "empty_input"
        assert results[0]["passed"] is False


class TestLLMJudgeCriteria:
    """Tests for failure mode LLM judge criteria."""

    def test_failure_handling_criterion_exists(self) -> None:
        assert "failure_handling" in CRITERIA_DESCRIPTIONS
        assert "error" in CRITERIA_DESCRIPTIONS["failure_handling"].lower()

    def test_uncertainty_acknowledgment_criterion_exists(self) -> None:
        assert "uncertainty_acknowledgment" in CRITERIA_DESCRIPTIONS
        assert "uncertainty" in CRITERIA_DESCRIPTIONS["uncertainty_acknowledgment"].lower()

    def test_all_criteria_have_descriptions(self) -> None:
        for name, desc in CRITERIA_DESCRIPTIONS.items():
            assert isinstance(desc, str), f"Criterion {name} has non-string description"
            assert len(desc) > 10, f"Criterion {name} has too short description"


class TestEdgeCaseEndToEnd:
    """End-to-end: generate evals → mock results → certification."""

    def test_generate_evals_and_certify(self, tmp_path: Path) -> None:
        from scripts.generate_edge_case_evals import generate_edge_case_evals

        # 1. Create submission with edge cases
        sub = tmp_path / "e2e-skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text("name: e2e-skill\n")
        edge_cases = sub / "edge_cases"
        edge_cases.mkdir()
        (edge_cases / "empty_input.md").write_text("Test with empty input")
        (edge_cases / "adversarial.md").write_text("Test with adversarial prompt")

        # 2. Generate edge case evals
        evals = generate_edge_case_evals(sub)
        assert evals is not None
        assert len(evals["evals"]) == 2

        # 3. Simulate edge case results (as pipeline would produce)
        behavioral_data = {
            "std_reward": 0.05,
            "edge_cases": {
                "total": len(evals["evals"]),
                "passed": len(evals["evals"]),
            },
        }

        # 4. Verify behavioral check passes
        behavioral_check = _compute_behavioral_testing_check(behavioral_data)
        assert behavioral_check.passed is True
        assert behavioral_check.score > 0.0


class TestBehavioralGateRegistry:
    """Tests for behavioral gate registry."""

    def test_edge_case_gate_registered(self) -> None:
        from abevalflow.gates.behavioral import get_all_behavioral_gates, get_behavioral_gate

        gate = get_behavioral_gate("edge-case")
        assert isinstance(gate, EdgeCaseGate)

        all_gates = get_all_behavioral_gates()
        assert len(all_gates) >= 1
        assert any(isinstance(g, EdgeCaseGate) for g in all_gates)

    def test_unknown_gate_raises(self) -> None:
        from abevalflow.gates.behavioral import get_behavioral_gate

        with pytest.raises(KeyError, match="Unknown behavioral gate"):
            get_behavioral_gate("nonexistent")

    def test_gate_type_is_behavioral(self) -> None:
        assert GateType.BEHAVIORAL == "behavioral"
