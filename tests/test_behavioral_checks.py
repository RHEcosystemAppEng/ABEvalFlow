"""Tests for behavioral testing certification checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abevalflow.certification import (
    CheckId,
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

    def test_behavioral_data_enables_certified_check(self) -> None:
        behavioral_data = {
            "std_reward": 0.1,
            "edge_cases": {"total": 3, "passed": 3},
        }
        result = compute_certification(
            gates=self._make_all_gates(),
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            behavioral_data=behavioral_data,
        )
        behavioral_check = next(
            (c for c in result.certified.checks if c.check_id == CheckId.ENTERPRISE_BEHAVIORAL_TESTING),
            None,
        )
        assert behavioral_check is not None
        assert behavioral_check.passed is True

    def test_no_behavioral_data_check_fails(self) -> None:
        result = compute_certification(
            gates=self._make_all_gates(),
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
        )
        behavioral_check = next(
            (c for c in result.certified.checks if c.check_id == CheckId.ENTERPRISE_BEHAVIORAL_TESTING),
            None,
        )
        assert behavioral_check is not None
        assert behavioral_check.passed is False

    def test_behavioral_check_in_certified_checks(self) -> None:
        from abevalflow.certification import CERTIFIED_CHECKS

        assert CheckId.ENTERPRISE_BEHAVIORAL_TESTING in CERTIFIED_CHECKS

    def test_backward_compatible_no_behavioral_data(self) -> None:
        result = compute_certification(
            gates=self._make_all_gates(),
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
        )
        assert result.foundational.passed is True
        assert result.trusted.passed is True


class TestEdgeCaseGate:
    """Tests for EdgeCaseGate behavioral gate."""

    def test_no_report_json(self, tmp_path: Path) -> None:
        policy = GatePolicy()
        gate = EdgeCaseGate()
        result = gate.evaluate(tmp_path, policy)
        assert result.passed is True
        assert result.gate_type == GateType.BEHAVIORAL

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


class TestEdgeCaseScaffolding:
    """Tests for edge case scaffolding."""

    def _make_submission(self, tmp_path: Path) -> Path:
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

        edge_cases = sub / "edge_cases"
        edge_cases.mkdir()
        (edge_cases / "empty_input.md").write_text("Test with empty input")
        (edge_cases / "adversarial.md").write_text("Test with adversarial prompt")
        return sub

    def test_edge_case_dirs_created(self, tmp_path: Path) -> None:
        from scripts.scaffold import scaffold_submission

        sub = self._make_submission(tmp_path)
        output = tmp_path / "output"
        scaffold_submission(sub, output)

        assert (output / "tasks-treatment-edge-adversarial" / "my-skill").is_dir()
        assert (output / "tasks-treatment-edge-empty_input" / "my-skill").is_dir()

    def test_edge_case_uses_alternative_instruction(self, tmp_path: Path) -> None:
        from scripts.scaffold import scaffold_submission

        sub = self._make_submission(tmp_path)
        output = tmp_path / "output"
        scaffold_submission(sub, output)

        edge_dir = output / "tasks-treatment-edge-empty_input" / "my-skill"
        instruction = (edge_dir / "instruction.md").read_text()
        assert instruction == "Test with empty input"

    def test_no_edge_cases_no_extra_dirs(self, tmp_path: Path) -> None:
        from scripts.scaffold import scaffold_submission

        sub = tmp_path / "no-edge"
        sub.mkdir()
        (sub / "metadata.yaml").write_text("name: no-edge\n")
        (sub / "instruction.md").write_text("Main instruction")
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text("# My Skill")
        tests = sub / "tests"
        tests.mkdir()
        (tests / "test_outputs.py").write_text("pass")

        output = tmp_path / "output"
        scaffold_submission(sub, output)

        edge_dirs = list(output.glob("tasks-treatment-edge-*"))
        assert edge_dirs == []


class TestEdgeCaseAnalysis:
    """Tests for edge case analysis in analyze.py."""

    def test_edge_case_result_model(self) -> None:
        result = EdgeCaseResult(
            name="empty_input",
            summary=VariantSummary(n_trials=5, n_passed=4, pass_rate=0.8, mean_reward=0.7),
            passed=True,
        )
        assert result.name == "empty_input"
        assert result.passed is True
        assert result.summary.pass_rate == 0.8

    def test_build_analysis_includes_edge_cases(self, tmp_path: Path) -> None:
        from scripts.analyze import build_analysis

        results = tmp_path / "results"
        treatment = results / "treatment" / "job" / "task__001"
        treatment.mkdir(parents=True)
        (treatment / "result.json").write_text(
            json.dumps(
                {
                    "verifier_result": {"reward": 0.8},
                }
            )
        )
        control = results / "control" / "job" / "task__001"
        control.mkdir(parents=True)
        (control / "result.json").write_text(
            json.dumps(
                {
                    "verifier_result": {"reward": 0.3},
                }
            )
        )

        edge_root = tmp_path / "edge_results"
        edge_dir = edge_root / "tasks-treatment-edge-empty_input" / "job" / "task__001"
        edge_dir.mkdir(parents=True)
        (edge_dir / "result.json").write_text(
            json.dumps(
                {
                    "verifier_result": {"reward": 0.6},
                }
            )
        )

        result = build_analysis(
            results_dir=results,
            submission_name="test-skill",
            edge_case_results_dir=edge_root,
        )
        assert len(result.edge_case_results) == 1
        assert result.edge_case_results[0].name == "empty_input"
        assert result.edge_case_results[0].passed is True

    def test_build_analysis_no_edge_cases(self, tmp_path: Path) -> None:
        from scripts.analyze import build_analysis

        results = tmp_path / "results"
        treatment = results / "treatment" / "job" / "task__001"
        treatment.mkdir(parents=True)
        (treatment / "result.json").write_text(
            json.dumps(
                {
                    "verifier_result": {"reward": 0.8},
                }
            )
        )
        control = results / "control" / "job" / "task__001"
        control.mkdir(parents=True)
        (control / "result.json").write_text(
            json.dumps(
                {
                    "verifier_result": {"reward": 0.3},
                }
            )
        )

        result = build_analysis(
            results_dir=results,
            submission_name="test-skill",
        )
        assert result.edge_case_results == []


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
    """End-to-end: scaffold → mock results → analyze → certification."""

    def test_scaffold_analyze_certify(self, tmp_path: Path) -> None:
        from scripts.analyze import build_analysis
        from scripts.scaffold import scaffold_submission

        # 1. Create submission with edge cases
        sub = tmp_path / "e2e-skill"
        sub.mkdir()
        (sub / "metadata.yaml").write_text("name: e2e-skill\n")
        (sub / "instruction.md").write_text("Main instruction")
        skills = sub / "skills"
        skills.mkdir()
        (skills / "SKILL.md").write_text("# E2E Skill")
        tests = sub / "tests"
        tests.mkdir()
        (tests / "test_outputs.py").write_text("pass")
        edge_cases = sub / "edge_cases"
        edge_cases.mkdir()
        (edge_cases / "empty_input.md").write_text("Test with empty input")
        (edge_cases / "adversarial.md").write_text("Test with adversarial prompt")

        # 2. Scaffold
        output = tmp_path / "scaffold_output"
        scaffold_submission(sub, output)
        assert (output / "tasks-treatment-edge-adversarial" / "e2e-skill").is_dir()
        assert (output / "tasks-treatment-edge-empty_input" / "e2e-skill").is_dir()

        # 3. Create mock Harbor results (2+ trials for std_reward computation)
        results = tmp_path / "results"
        for variant in ("treatment", "control"):
            for i, reward in enumerate([0.8, 0.85] if variant == "treatment" else [0.3, 0.35]):
                trial_dir = results / variant / "job" / f"task__{i:03d}"
                trial_dir.mkdir(parents=True)
                (trial_dir / "result.json").write_text(
                    json.dumps(
                        {
                            "verifier_result": {"reward": reward},
                        }
                    )
                )

        # 4. Create mock results for edge cases
        edge_results = tmp_path / "edge_results"
        for edge_name, reward in [("adversarial", 0.7), ("empty_input", 0.6)]:
            trial_dir = edge_results / f"tasks-treatment-edge-{edge_name}" / "job" / "task__001"
            trial_dir.mkdir(parents=True)
            (trial_dir / "result.json").write_text(
                json.dumps(
                    {
                        "verifier_result": {"reward": reward},
                    }
                )
            )

        # 5. Analyze with edge case results
        analysis = build_analysis(
            results_dir=results,
            submission_name="e2e-skill",
            edge_case_results_dir=edge_results,
        )
        assert len(analysis.edge_case_results) == 2
        assert all(ec.passed for ec in analysis.edge_case_results)

        # 6. Simulate the unified data path: std_reward from report + edge cases from gate
        behavioral_data = {
            "std_reward": analysis.summary.treatment.std_reward,
            "edge_cases": {
                "total": len(analysis.edge_case_results),
                "passed": sum(1 for ec in analysis.edge_case_results if ec.passed),
            },
        }

        # 7. Verify certification sees it
        result = compute_certification(
            gates=[
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
                    score=0.8,
                    mode=GateMode.WARN,
                ),
            ],
            validation_passed=True,
            metadata_valid=True,
            has_eval_assets=True,
            behavioral_data=behavioral_data,
        )
        behavioral_check = next(
            c for c in result.certified.checks if c.check_id == CheckId.ENTERPRISE_BEHAVIORAL_TESTING
        )
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
