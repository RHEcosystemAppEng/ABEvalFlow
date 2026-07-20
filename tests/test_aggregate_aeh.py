"""Tests for AEH report aggregation (scripts/aggregate_aeh.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from abevalflow.report import AnalysisResult
from scripts.aggregate_aeh import (
    _case_reward,
    _trials_from_per_case,
    aggregate_pairwise_run,
    aggregate_single_run,
)
from scripts.analyze import render_markdown


def _write_summary(run_dir: Path, summary: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.yaml").write_text(yaml.dump(summary))


class TestCaseReward:
    def test_boolean_pass(self):
        assert _case_reward({"exit_success": {"value": True}}) == 1.0

    def test_boolean_fail(self):
        assert _case_reward({"exit_success": {"value": False}}) == 0.0

    def test_numeric_mean(self):
        assert (
            _case_reward(
                {
                    "a": {"value": 0.5},
                    "b": {"value": 1.0},
                }
            )
            == 0.75
        )

    def test_top_level_reward(self):
        assert _case_reward({"reward": 0.9}) == 0.9

    def test_empty(self):
        assert _case_reward({}) is None


class TestTrialsFromPerCase:
    def test_maps_case_ids(self):
        trials = _trials_from_per_case(
            {
                "case-001": {"exit_success": {"value": True}},
                "case-002": {"exit_success": {"value": False}},
            }
        )
        assert len(trials) == 2
        assert trials[0]["trial_name"] == "case-001"
        assert trials[0]["reward"] == 1.0
        assert trials[1]["reward"] == 0.0


class TestAggregateSingleRun:
    def test_trials_match_n_trials(self, tmp_path: Path):
        run_dir = tmp_path / "aeh-hello" / "run-1"
        _write_summary(
            run_dir,
            {
                "run_id": "run-1",
                "mean_reward": 1.0,
                "per_case": {
                    "case-001": {"exit_success": {"value": True}},
                },
                "judges": {},
            },
        )
        report = aggregate_single_run(run_dir)
        assert report["summary"]["treatment"]["n_trials"] == 1
        assert len(report["trials"]["treatment"]) == 1
        assert report["trials"]["treatment"][0]["trial_name"] == "case-001"
        assert report["trials"]["control"] == []

        result = AnalysisResult.model_validate(report)
        md = render_markdown(result)
        assert "Treatment (1 trials)" in md
        assert "case-001" in md


class TestAggregatePairwiseRun:
    def test_trials_and_pairwise_section(self, tmp_path: Path):
        treatment = tmp_path / "skill" / "treatment-1"
        control = tmp_path / "skill" / "control-1"
        _write_summary(
            treatment,
            {
                "run_id": "treatment-1",
                "mean_reward": 1.0,
                "per_case": {"case-001": {"exit_success": {"value": True}}},
                "pairwise": {
                    "run_a": "treatment-1",
                    "run_b": "control-1",
                    "cases_compared": 1,
                    "wins_a": 1,
                    "wins_b": 0,
                    "ties": 0,
                    "errors": 0,
                    "per_case": [{"case_id": "case-001", "winner": "a"}],
                },
            },
        )
        _write_summary(
            control,
            {
                "run_id": "control-1",
                "mean_reward": 0.0,
                "per_case": {"case-001": {"exit_success": {"value": False}}},
            },
        )
        report = aggregate_pairwise_run(treatment, control)
        assert len(report["trials"]["treatment"]) == 1
        assert len(report["trials"]["control"]) == 1
        assert report["pairwise"]["wins_a"] == 1
        assert report["aeh_warnings"] == []

        result = AnalysisResult.model_validate(report)
        md = render_markdown(result)
        assert "## Pairwise Comparison" in md
        assert "**Treatment wins:** 1" in md
        assert "case-001" in md
        assert "Treatment (1 trials)" in md
        assert "Control (1 trials)" in md

    def test_missing_pairwise_warns(self, tmp_path: Path):
        treatment = tmp_path / "skill" / "treatment-1"
        control = tmp_path / "skill" / "control-1"
        _write_summary(
            treatment,
            {
                "run_id": "treatment-1",
                "mean_reward": 0.0,
                "per_case": {},
            },
        )
        _write_summary(control, {"run_id": "control-1", "mean_reward": 0.0, "per_case": {}})
        report = aggregate_pairwise_run(treatment, control)
        assert report["aeh_warnings"]
        assert report["pairwise"]["cases_compared"] == 0

    def test_all_ties_is_pass_with_rewards_from_run_result(self, tmp_path: Path):
        treatment = tmp_path / "skill" / "treatment-1"
        control = tmp_path / "skill" / "control-1"
        _write_summary(
            treatment,
            {
                "run_id": "treatment-1",
                "per_case": {"case-001": {"exit_success": {"value": True}}},
                "pairwise": {
                    "wins_a": 0,
                    "wins_b": 0,
                    "ties": 1,
                    "errors": 0,
                    "cases_compared": 1,
                    "per_case": [{"case_id": "case-001", "winner": "tie"}],
                },
            },
        )
        (treatment / "run_result.json").write_text(json.dumps({"mean_reward": 1.0}))
        _write_summary(
            control,
            {
                "run_id": "control-1",
                "per_case": {"case-001": {"exit_success": {"value": True}}},
            },
        )
        (control / "run_result.json").write_text(json.dumps({"mean_reward": 1.0}))
        report = aggregate_pairwise_run(treatment, control)
        assert report["summary"]["treatment"]["mean_reward"] == 1.0
        assert report["summary"]["control"]["mean_reward"] == 1.0
        assert report["pairwise"]["win_rate"] == 0.0  # ties are non-wins
        assert report["recommendation"] == "pass"  # all-ties is still pass

    def test_errors_lower_win_rate(self, tmp_path: Path):
        treatment = tmp_path / "skill" / "treatment-1"
        control = tmp_path / "skill" / "control-1"
        _write_summary(
            treatment,
            {
                "run_id": "treatment-1",
                "per_case": {},
                "pairwise": {
                    "wins_a": 1,
                    "wins_b": 0,
                    "ties": 0,
                    "errors": 9,
                    "cases_compared": 10,
                    "per_case": [],
                },
            },
        )
        _write_summary(control, {"run_id": "control-1", "per_case": {}})
        report = aggregate_pairwise_run(treatment, control)
        assert report["pairwise"]["win_rate"] == pytest.approx(0.1)
        assert report["recommendation"] == "fail"

    def test_submission_name_override(self, tmp_path: Path):
        run_dir = tmp_path / "skill-from-path" / "run-1"
        _write_summary(
            run_dir,
            {
                "run_id": "run-1",
                "mean_reward": 1.0,
                "per_case": {"case-001": {"exit_success": {"value": True}}},
                "judges": {},
            },
        )
        report = aggregate_single_run(run_dir, submission_name="pipeline-submission")
        assert report["submission_name"] == "pipeline-submission"

    def test_likert_one_is_not_a_pass(self, tmp_path: Path):
        run_dir = tmp_path / "skill" / "run-1"
        _write_summary(
            run_dir,
            {
                "run_id": "run-1",
                "mean_reward": 0.2,
                "per_case": {
                    "case-001": {"output_quality": {"value": 1}},
                },
                "judges": {},
            },
        )
        report = aggregate_single_run(run_dir)
        assert report["passed_cases"] == 0
        assert report["pass_rate"] == 0.0
