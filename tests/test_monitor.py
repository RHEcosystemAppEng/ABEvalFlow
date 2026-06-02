"""Tests for scripts/monitor.py."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from abevalflow.db.models import Base, EvaluationRun
from scripts.monitor import MonitorResult, check_degradation


@pytest.fixture
def engine():
    """Create an in-memory SQLite database with schema."""
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session(engine):
    """Create a session factory bound to the test engine."""
    return sessionmaker(bind=engine)


def _create_run(
    session_factory,
    submission_name: str,
    pipeline_run_id: str,
    treatment_pass_rate: float,
    created_at: datetime,
) -> EvaluationRun:
    """Helper to insert a minimal EvaluationRun for testing."""
    run = EvaluationRun(
        submission_name=submission_name,
        pipeline_run_id=pipeline_run_id,
        treatment_pass_rate=treatment_pass_rate,
        control_pass_rate=0.5,
        treatment_n_trials=20,
        treatment_n_passed=int(20 * treatment_pass_rate),
        treatment_n_failed=20 - int(20 * treatment_pass_rate),
        treatment_n_errors=0,
        control_n_trials=20,
        control_n_passed=10,
        control_n_failed=10,
        control_n_errors=0,
        uplift=treatment_pass_rate - 0.5,
        recommendation="pass" if treatment_pass_rate > 0.5 else "fail",
        report_json={},
        created_at=created_at,
    )
    with session_factory() as sess:
        sess.add(run)
        sess.commit()
        sess.refresh(run)
    return run


class TestCheckDegradation:
    """Tests for check_degradation function."""

    def test_no_runs_returns_not_degraded(self, engine):
        """When no runs exist, should return not degraded with message."""
        result = check_degradation(engine, "nonexistent-skill")

        assert result.degraded is False
        assert result.current_score is None
        assert result.previous_score is None
        assert "Not enough data" in result.message
        assert "0 run(s)" in result.message

    def test_single_run_returns_not_degraded(self, engine, session):
        """When only one run exists, should return not degraded."""
        now = datetime.now(timezone.utc)
        _create_run(session, "test-skill", "run-001", 0.8, now)

        result = check_degradation(engine, "test-skill")

        assert result.degraded is False
        assert result.current_score == 0.8
        assert result.previous_score is None
        assert "Not enough data" in result.message
        assert "1 run(s)" in result.message

    def test_healthy_no_degradation(self, engine, session):
        """When current score is similar to previous, no degradation."""
        now = datetime.now(timezone.utc)
        _create_run(session, "test-skill", "run-001", 0.80, now - timedelta(days=1))
        _create_run(session, "test-skill", "run-002", 0.82, now)

        result = check_degradation(engine, "test-skill", threshold=0.85)

        assert result.degraded is False
        assert result.current_score == 0.82
        assert result.previous_score == 0.80
        assert result.ratio == pytest.approx(1.025, rel=0.01)
        assert "No degradation" in result.message

    def test_degradation_detected(self, engine, session):
        """When score drops significantly, degradation is detected."""
        now = datetime.now(timezone.utc)
        _create_run(session, "test-skill", "run-001", 0.90, now - timedelta(days=1))
        _create_run(session, "test-skill", "run-002", 0.70, now)

        result = check_degradation(engine, "test-skill", threshold=0.85)

        assert result.degraded is True
        assert result.current_score == 0.70
        assert result.previous_score == 0.90
        assert result.ratio == pytest.approx(0.778, rel=0.01)
        assert "Degradation detected" in result.message
        assert result.current_run_id == "run-002"
        assert result.previous_run_id == "run-001"

    def test_exact_threshold_not_degraded(self, engine, session):
        """When ratio equals threshold exactly, no degradation."""
        now = datetime.now(timezone.utc)
        _create_run(session, "test-skill", "run-001", 1.0, now - timedelta(days=1))
        _create_run(session, "test-skill", "run-002", 0.85, now)

        result = check_degradation(engine, "test-skill", threshold=0.85)

        assert result.degraded is False
        assert result.ratio == 0.85

    def test_just_below_threshold_is_degraded(self, engine, session):
        """When ratio is just below threshold, degradation is detected."""
        now = datetime.now(timezone.utc)
        _create_run(session, "test-skill", "run-001", 1.0, now - timedelta(days=1))
        _create_run(session, "test-skill", "run-002", 0.84, now)

        result = check_degradation(engine, "test-skill", threshold=0.85)

        assert result.degraded is True
        assert result.ratio == 0.84

    def test_previous_score_zero(self, engine, session):
        """Edge case: previous score is zero."""
        now = datetime.now(timezone.utc)
        _create_run(session, "test-skill", "run-001", 0.0, now - timedelta(days=1))
        _create_run(session, "test-skill", "run-002", 0.5, now)

        result = check_degradation(engine, "test-skill", threshold=0.85)

        assert result.degraded is False
        assert result.ratio == float("inf")

    def test_both_scores_zero(self, engine, session):
        """Edge case: both scores are zero."""
        now = datetime.now(timezone.utc)
        _create_run(session, "test-skill", "run-001", 0.0, now - timedelta(days=1))
        _create_run(session, "test-skill", "run-002", 0.0, now)

        result = check_degradation(engine, "test-skill", threshold=0.85)

        assert result.degraded is True
        assert result.ratio == 0.0

    def test_filters_by_submission_name(self, engine, session):
        """Should only consider runs for the specified submission."""
        now = datetime.now(timezone.utc)
        _create_run(session, "skill-a", "run-a1", 0.9, now - timedelta(days=2))
        _create_run(session, "skill-a", "run-a2", 0.85, now - timedelta(days=1))
        _create_run(session, "skill-b", "run-b1", 0.5, now)

        result = check_degradation(engine, "skill-a", threshold=0.85)

        assert result.degraded is False
        assert result.current_score == 0.85
        assert result.previous_score == 0.9
        assert result.current_run_id == "run-a2"

    def test_custom_threshold(self, engine, session):
        """Custom threshold should be respected."""
        now = datetime.now(timezone.utc)
        _create_run(session, "test-skill", "run-001", 1.0, now - timedelta(days=1))
        _create_run(session, "test-skill", "run-002", 0.75, now)

        result_low = check_degradation(engine, "test-skill", threshold=0.70)
        result_high = check_degradation(engine, "test-skill", threshold=0.80)

        assert result_low.degraded is False
        assert result_high.degraded is True


class TestMonitorResult:
    """Tests for MonitorResult dataclass."""

    def test_dataclass_fields(self):
        """MonitorResult should have all expected fields."""
        result = MonitorResult(
            submission_name="test",
            degraded=True,
            current_score=0.7,
            previous_score=0.9,
            ratio=0.78,
            threshold=0.85,
            message="Test message",
            current_run_id="run-002",
            previous_run_id="run-001",
        )

        assert result.submission_name == "test"
        assert result.degraded is True
        assert result.current_score == 0.7
        assert result.previous_score == 0.9
        assert result.ratio == 0.78
        assert result.threshold == 0.85
        assert result.message == "Test message"
        assert result.current_run_id == "run-002"
        assert result.previous_run_id == "run-001"
