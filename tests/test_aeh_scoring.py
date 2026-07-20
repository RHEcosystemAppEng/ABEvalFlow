"""Unit tests for shared AEH scoring helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from abevalflow.aeh_scoring import (
    DEFAULT_AEH_THRESHOLD,
    numeric_judge_is_low,
    numeric_judge_passes,
    pairwise_outcome,
    resolve_evaluation_threshold,
)


class TestPairwiseOutcome:
    def test_includes_errors_in_denominator(self):
        out = pairwise_outcome(wins_a=1, wins_b=0, ties=0, errors=9)
        assert out["total"] == 10
        assert out["win_rate"] == 0.1
        assert out["passed"] is False

    def test_all_ties_passes(self):
        out = pairwise_outcome(wins_a=0, wins_b=0, ties=5, errors=0)
        assert out["all_ties"] is True
        assert out["passed"] is True
        assert out["win_rate"] == 0.0

    def test_all_errors_fails(self):
        out = pairwise_outcome(wins_a=0, wins_b=0, ties=0, errors=5)
        assert out["all_ties"] is False
        assert out["passed"] is False


class TestNumericJudge:
    def test_normalized_scale(self):
        assert numeric_judge_passes(0.5) is True
        assert numeric_judge_passes(0.49) is False

    def test_likert_int(self):
        assert numeric_judge_passes(1) is False
        assert numeric_judge_passes(3) is True
        assert numeric_judge_is_low(2) is True

    def test_likert_float(self):
        assert numeric_judge_passes(2.5) is False
        assert numeric_judge_passes(3.0) is True


class TestResolveEvaluationThreshold:
    def test_default_when_missing(self, tmp_path: Path):
        assert resolve_evaluation_threshold(tmp_path / "missing.yaml") == DEFAULT_AEH_THRESHOLD

    def test_reads_gate_policy(self, tmp_path: Path):
        meta = tmp_path / "metadata.yaml"
        meta.write_text(
            yaml.dump(
                {
                    "name": "demo",
                    "gate_policy": {"gates": {"evaluation": {"threshold": 0.6}}},
                }
            )
        )
        assert resolve_evaluation_threshold(meta) == 0.6
