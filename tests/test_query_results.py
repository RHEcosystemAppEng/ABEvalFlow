"""Tests for scripts/query_results.py."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from abevalflow.db.models import Base, EvaluationRun
from scripts.query_results import cmd_compare, cmd_history, cmd_latest, cmd_list


def _make_run(
    name: str = "test-sub",
    recommendation: str = "pass",
    uplift: float = 0.3,
    t_rate: float = 0.8,
    c_rate: float = 0.5,
    created_at: datetime | None = None,
    **kwargs,
) -> EvaluationRun:
    defaults = {
        "submission_name": name,
        "pipeline_run_id": f"run-{uuid.uuid4().hex[:8]}",
        "recommendation": recommendation,
        "uplift": uplift,
        "treatment_n_trials": 20,
        "treatment_n_passed": 16,
        "treatment_n_failed": 3,
        "treatment_n_errors": 1,
        "treatment_pass_rate": t_rate,
        "control_n_trials": 20,
        "control_n_passed": 10,
        "control_n_failed": 8,
        "control_n_errors": 2,
        "control_pass_rate": c_rate,
        "report_json": {"submission_name": name},
    }
    defaults.update(kwargs)
    run = EvaluationRun(**defaults)
    if created_at:
        run.created_at = created_at
    return run


@pytest.fixture()
def session_factory():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture()
def seeded_factory(session_factory):
    """Seed 3 runs: 2 for 'alpha', 1 for 'beta'."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    with session_factory() as session:
        session.add(
            _make_run(
                name="alpha",
                uplift=0.2,
                recommendation="pass",
                created_at=now - timedelta(days=2),
            )
        )
        session.add(
            _make_run(
                name="alpha",
                uplift=0.4,
                recommendation="pass",
                created_at=now - timedelta(days=1),
            )
        )
        session.add(
            _make_run(
                name="beta",
                uplift=-0.1,
                recommendation="fail",
                created_at=now,
            )
        )
        session.commit()
    return session_factory


class TestCmdList:
    def test_list_shows_submissions(self, seeded_factory, capsys):
        cmd_list(seeded_factory)
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "beta" in out

    def test_list_empty_db(self, session_factory, capsys):
        cmd_list(session_factory)
        out = capsys.readouterr().out
        assert "No evaluation runs found" in out


class TestCmdLatest:
    def test_latest_shows_details(self, seeded_factory, capsys):
        cmd_latest(seeded_factory, "alpha")
        out = capsys.readouterr().out
        assert "alpha" in out
        assert "Recommendation" in out
        assert "Treatment" in out
        assert "Control" in out

    def test_latest_not_found(self, seeded_factory, capsys):
        cmd_latest(seeded_factory, "nonexistent")
        out = capsys.readouterr().out
        assert "No runs found" in out

    def test_latest_returns_most_recent(self, seeded_factory, capsys):
        cmd_latest(seeded_factory, "alpha")
        out = capsys.readouterr().out
        assert "0.4000" in out


class TestCmdHistory:
    def test_history_shows_all_runs(self, seeded_factory, capsys):
        cmd_history(seeded_factory, "alpha")
        out = capsys.readouterr().out
        assert "2 runs" in out
        assert "0.2000" in out
        assert "0.4000" in out

    def test_history_not_found(self, seeded_factory, capsys):
        cmd_history(seeded_factory, "nonexistent")
        out = capsys.readouterr().out
        assert "No runs found" in out


class TestCmdCompare:
    def test_compare_shows_trend(self, seeded_factory, capsys):
        cmd_compare(seeded_factory, "alpha")
        out = capsys.readouterr().out
        assert "Trend" in out
        assert "2 runs" in out

    def test_compare_not_found(self, seeded_factory, capsys):
        cmd_compare(seeded_factory, "nonexistent")
        out = capsys.readouterr().out
        assert "No runs found" in out


class TestMainCLI:
    def test_list_via_main(self, tmp_path, monkeypatch, capsys):
        db_url = f"sqlite:///{tmp_path / 'query.db'}"
        engine = create_engine(db_url, connect_args={"check_same_thread": False})
        Base.metadata.create_all(engine)
        monkeypatch.setattr(
            "sys.argv",
            ["query_results.py", "--database-url", db_url, "list"],
        )
        from scripts.query_results import main

        main()
        out = capsys.readouterr().out
        assert "No evaluation runs found" in out

    def test_missing_command(self, monkeypatch):
        monkeypatch.setattr(
            "sys.argv",
            ["query_results.py", "--database-url", "sqlite://"],
        )
        from scripts.query_results import main

        with pytest.raises(SystemExit):
            main()
