"""Tests for observability DB models (scorecards, gate_results, certifications, observability_metrics)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from abevalflow.db.models import (
    Base,
    CertificationRow,
    GateResultRow,
    ObservabilityMetricsRow,
    ScorecardRow,
)


@pytest.fixture()
def engine():
    eng = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture()
def session(engine) -> Session:
    factory = sessionmaker(bind=engine)
    with factory() as s:
        yield s


def _make_scorecard(**overrides) -> ScorecardRow:
    defaults = {
        "pipeline_run_id": f"run-{uuid.uuid4().hex[:8]}",
        "submission_name": "test-skill",
        "eval_engine": "harbor",
        "recommendation": "pass",
        "recommendation_reason": "All gates passed",
        "combination_mode": "all_pass",
        "gates_passed": 3,
        "gates_failed": 0,
        "blocking_gates_passed": 2,
        "blocking_gates_failed": 0,
        "highest_certification": "trusted",
        "scorecard_json": {"recommendation": "pass", "gates": []},
    }
    defaults.update(overrides)
    return ScorecardRow(**defaults)


def _make_gate_result(scorecard: ScorecardRow, **overrides) -> GateResultRow:
    defaults = {
        "scorecard": scorecard,
        "gate_name": "evaluation",
        "gate_type": "engine",
        "policy_key": "harbor",
        "passed": True,
        "score": 0.85,
        "mode": "block",
        "threshold": 0.0,
        "findings_count": 0,
    }
    defaults.update(overrides)
    return GateResultRow(**defaults)


def _make_certification(scorecard: ScorecardRow, **overrides) -> CertificationRow:
    defaults = {
        "scorecard": scorecard,
        "level": "foundational",
        "passed": True,
        "checks_total": 5,
        "checks_passed": 5,
        "checks_failed": 0,
    }
    defaults.update(overrides)
    return CertificationRow(**defaults)


def _make_metrics(**overrides) -> ObservabilityMetricsRow:
    defaults = {
        "pipeline_run_id": f"run-{uuid.uuid4().hex[:8]}",
        "submission_name": "test-skill",
        "model_name": "claude-sonnet-4-6",
        "total_prompt_tokens": 15000,
        "total_completion_tokens": 3500,
        "total_tokens": 18500,
        "estimated_cost_usd": 0.097500,
        "attempt_number": 1,
    }
    defaults.update(overrides)
    return ObservabilityMetricsRow(**defaults)


class TestScorecardRow:
    def test_create_and_read(self, session: Session) -> None:
        sc = _make_scorecard()
        session.add(sc)
        session.commit()

        loaded = session.execute(select(ScorecardRow)).scalar_one()
        assert loaded.submission_name == "test-skill"
        assert loaded.recommendation == "pass"
        assert loaded.highest_certification == "trusted"

    def test_created_at_default(self, session: Session) -> None:
        sc = _make_scorecard()
        session.add(sc)
        session.commit()

        loaded = session.execute(select(ScorecardRow)).scalar_one()
        assert loaded.created_at is not None
        assert (datetime.now(UTC) - loaded.created_at.replace(tzinfo=UTC)).total_seconds() < 5

    def test_unique_pipeline_run_id(self, session: Session) -> None:
        run_id = "run-duplicate"
        session.add(_make_scorecard(pipeline_run_id=run_id))
        session.commit()

        session.add(_make_scorecard(pipeline_run_id=run_id))
        with pytest.raises(Exception):
            session.commit()

    def test_scorecard_json_round_trip(self, session: Session) -> None:
        data = {"recommendation": "pass", "gates": [{"name": "security", "passed": True}]}
        sc = _make_scorecard(scorecard_json=data)
        session.add(sc)
        session.commit()

        loaded = session.execute(select(ScorecardRow)).scalar_one()
        assert loaded.scorecard_json == data

    def test_repr(self) -> None:
        sc = _make_scorecard()
        assert "test-skill" in repr(sc)
        assert "pass" in repr(sc)


class TestGateResultRow:
    def test_create_linked_to_scorecard(self, session: Session) -> None:
        sc = _make_scorecard()
        gate = _make_gate_result(sc)
        session.add(sc)
        session.add(gate)
        session.commit()

        loaded = session.execute(select(GateResultRow)).scalar_one()
        assert loaded.gate_name == "evaluation"
        assert loaded.passed is True
        assert loaded.score == 0.85

    def test_nullable_fields(self, session: Session) -> None:
        sc = _make_scorecard()
        gate = _make_gate_result(
            sc,
            threshold=None,
            message=None,
            details_json=None,
            duration_ms=None,
            prompt_tokens=None,
            completion_tokens=None,
            evaluated_at=None,
        )
        session.add(sc)
        session.add(gate)
        session.commit()

        loaded = session.execute(select(GateResultRow)).scalar_one()
        assert loaded.threshold is None
        assert loaded.duration_ms is None
        assert loaded.prompt_tokens is None

    def test_cascade_delete(self, session: Session) -> None:
        sc = _make_scorecard()
        gate = _make_gate_result(sc)
        session.add(sc)
        session.add(gate)
        session.commit()

        session.delete(sc)
        session.commit()

        assert session.execute(select(GateResultRow)).scalar_one_or_none() is None

    def test_relationship_back_populates(self, session: Session) -> None:
        sc = _make_scorecard()
        g1 = _make_gate_result(sc, gate_name="evaluation")
        g2 = _make_gate_result(sc, gate_name="security", policy_key="cisco", score=0.92)
        session.add(sc)
        session.add_all([g1, g2])
        session.commit()

        loaded = session.execute(select(ScorecardRow)).scalar_one()
        assert len(loaded.gate_results) == 2


class TestCertificationRow:
    def test_create_all_three_levels(self, session: Session) -> None:
        sc = _make_scorecard()
        levels = [
            _make_certification(sc, level="foundational", passed=True),
            _make_certification(sc, level="trusted", passed=True, checks_passed=4, checks_failed=1),
            _make_certification(sc, level="certified", passed=False, checks_passed=2, checks_failed=3),
        ]
        session.add(sc)
        session.add_all(levels)
        session.commit()

        loaded = session.execute(select(CertificationRow)).scalars().all()
        assert len(loaded) == 3

    def test_cascade_delete(self, session: Session) -> None:
        sc = _make_scorecard()
        cert = _make_certification(sc)
        session.add(sc)
        session.add(cert)
        session.commit()

        session.delete(sc)
        session.commit()

        assert session.execute(select(CertificationRow)).scalar_one_or_none() is None

    def test_failed_checks_json(self, session: Session) -> None:
        sc = _make_scorecard()
        cert = _make_certification(
            sc,
            passed=False,
            checks_passed=3,
            checks_failed=2,
            failed_checks=["basic_security_validation", "metadata_compliance"],
        )
        session.add(sc)
        session.add(cert)
        session.commit()

        loaded = session.execute(select(CertificationRow)).scalar_one()
        assert loaded.failed_checks == ["basic_security_validation", "metadata_compliance"]


class TestObservabilityMetricsRow:
    def test_create_and_read(self, session: Session) -> None:
        m = _make_metrics()
        session.add(m)
        session.commit()

        loaded = session.execute(select(ObservabilityMetricsRow)).scalar_one()
        assert loaded.total_tokens == 18500
        assert loaded.model_name == "claude-sonnet-4-6"

    def test_unique_run_attempt(self, session: Session) -> None:
        run_id = "run-unique-test"
        session.add(_make_metrics(pipeline_run_id=run_id, attempt_number=1))
        session.commit()

        session.add(_make_metrics(pipeline_run_id=run_id, attempt_number=1))
        with pytest.raises(Exception):
            session.commit()

    def test_multiple_attempts_allowed(self, session: Session) -> None:
        run_id = "run-retry-test"
        session.add(_make_metrics(pipeline_run_id=run_id, attempt_number=1))
        session.add(_make_metrics(pipeline_run_id=run_id, attempt_number=2))
        session.commit()

        loaded = session.execute(select(ObservabilityMetricsRow)).scalars().all()
        assert len(loaded) == 2

    def test_nullable_token_fields(self, session: Session) -> None:
        m = _make_metrics(
            total_prompt_tokens=None,
            total_completion_tokens=None,
            total_tokens=None,
            estimated_cost_usd=None,
            pipeline_duration_ms=None,
        )
        session.add(m)
        session.commit()

        loaded = session.execute(select(ObservabilityMetricsRow)).scalar_one()
        assert loaded.total_tokens is None
        assert loaded.estimated_cost_usd is None

    def test_numeric_cost_precision(self, session: Session) -> None:
        m = _make_metrics(estimated_cost_usd=0.123456)
        session.add(m)
        session.commit()

        loaded = session.execute(select(ObservabilityMetricsRow)).scalar_one()
        assert abs(float(loaded.estimated_cost_usd) - 0.123456) < 1e-6
