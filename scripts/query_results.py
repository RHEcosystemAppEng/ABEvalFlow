"""Query historical A/B evaluation results from the database.

Usage::

    python scripts/query_results.py list
    python scripts/query_results.py latest my-submission
    python scripts/query_results.py history my-submission
    python scripts/query_results.py compare my-submission
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import desc, func, select

from abevalflow.db.engine import get_engine, make_session
from abevalflow.db.models import EvaluationRun

logger = logging.getLogger(__name__)

# Column widths for tabular output
_COL = {
    "name": 25,
    "rec": 6,
    "uplift": 8,
    "t_rate": 7,
    "c_rate": 7,
    "p_tt": 8,
    "p_fi": 8,
    "date": 20,
    "run_id": 20,
}


def _header() -> str:
    return (
        f"{'Submission':<{_COL['name']}} "
        f"{'Rec':<{_COL['rec']}} "
        f"{'Uplift':>{_COL['uplift']}} "
        f"{'T.Rate':>{_COL['t_rate']}} "
        f"{'C.Rate':>{_COL['c_rate']}} "
        f"{'p(tt)':>{_COL['p_tt']}} "
        f"{'p(fi)':>{_COL['p_fi']}} "
        f"{'Date':<{_COL['date']}}"
    )


def _row(r: EvaluationRun) -> str:
    def _fmt(v: float | None, width: int) -> str:
        return f"{v:{width}.4f}" if v is not None else f"{'—':>{width}}"

    ts = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "—"
    return (
        f"{r.submission_name:<{_COL['name']}} "
        f"{r.recommendation:<{_COL['rec']}} "
        f"{r.uplift:{_COL['uplift']}.4f} "
        f"{r.treatment_pass_rate:{_COL['t_rate']}.4f} "
        f"{r.control_pass_rate:{_COL['c_rate']}.4f} "
        f"{_fmt(r.ttest_p_value, _COL['p_tt'])} "
        f"{_fmt(r.fisher_p_value, _COL['p_fi'])} "
        f"{ts}"
    )


def cmd_list(session_factory) -> None:
    """List all submissions with their latest result."""
    subq = (
        select(
            EvaluationRun.submission_name,
            func.max(EvaluationRun.created_at).label("latest"),
        )
        .group_by(EvaluationRun.submission_name)
        .subquery()
    )

    stmt = (
        select(EvaluationRun)
        .join(
            subq,
            (EvaluationRun.submission_name == subq.c.submission_name)
            & (EvaluationRun.created_at == subq.c.latest),
        )
        .order_by(desc(EvaluationRun.created_at))
    )

    with session_factory() as session:
        runs = session.execute(stmt).scalars().all()
        if not runs:
            print("No evaluation runs found.")
            return
        print(_header())
        print("-" * 110)
        for r in runs:
            print(_row(r))


def cmd_latest(session_factory, name: str) -> None:
    """Show the latest run for a submission."""
    stmt = (
        select(EvaluationRun)
        .where(EvaluationRun.submission_name == name)
        .order_by(desc(EvaluationRun.created_at))
        .limit(1)
    )

    with session_factory() as session:
        run = session.execute(stmt).scalar_one_or_none()
        if not run:
            print(f"No runs found for '{name}'.")
            return

        print(f"Submission:    {run.submission_name}")
        print(f"Run ID:        {run.pipeline_run_id}")
        print(f"Date:          {run.created_at}")
        print(f"Recommendation:{run.recommendation}")
        print(f"Uplift:        {run.uplift:.4f}")
        print()
        print("Treatment:")
        print(f"  Trials: {run.treatment_n_trials}  "
              f"Pass: {run.treatment_n_passed}  "
              f"Fail: {run.treatment_n_failed}  "
              f"Errors: {run.treatment_n_errors}")
        print(f"  Pass rate: {run.treatment_pass_rate:.4f}  "
              f"Mean reward: {run.treatment_mean_reward or '—'}  "
              f"Std: {run.treatment_std_reward or '—'}")
        print()
        print("Control:")
        print(f"  Trials: {run.control_n_trials}  "
              f"Pass: {run.control_n_passed}  "
              f"Fail: {run.control_n_failed}  "
              f"Errors: {run.control_n_errors}")
        print(f"  Pass rate: {run.control_pass_rate:.4f}  "
              f"Mean reward: {run.control_mean_reward or '—'}  "
              f"Std: {run.control_std_reward or '—'}")
        print()
        print(f"t-test p-value:  {run.ttest_p_value or '—'}")
        print(f"Fisher p-value:  {run.fisher_p_value or '—'}")
        if run.commit_sha:
            print(f"Commit SHA:      {run.commit_sha}")


def cmd_history(session_factory, name: str) -> None:
    """Show all runs for a submission."""
    stmt = (
        select(EvaluationRun)
        .where(EvaluationRun.submission_name == name)
        .order_by(desc(EvaluationRun.created_at))
    )

    with session_factory() as session:
        runs = session.execute(stmt).scalars().all()
        if not runs:
            print(f"No runs found for '{name}'.")
            return
        print(f"History for '{name}' ({len(runs)} runs):\n")
        print(_header())
        print("-" * 110)
        for r in runs:
            print(_row(r))


def cmd_compare(session_factory, name: str) -> None:
    """Show pass_rate and uplift trend over time."""
    stmt = (
        select(EvaluationRun)
        .where(EvaluationRun.submission_name == name)
        .order_by(EvaluationRun.created_at)
    )

    with session_factory() as session:
        runs = session.execute(stmt).scalars().all()
        if not runs:
            print(f"No runs found for '{name}'.")
            return

        print(f"Trend for '{name}' ({len(runs)} runs, oldest first):\n")
        print(
            f"{'Date':<20} {'Rec':<6} {'Uplift':>8} "
            f"{'T.Rate':>7} {'C.Rate':>7} {'T.N':>4} {'C.N':>4}"
        )
        print("-" * 75)
        for r in runs:
            ts = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "—"
            print(
                f"{ts:<20} {r.recommendation:<6} {r.uplift:8.4f} "
                f"{r.treatment_pass_rate:7.4f} {r.control_pass_rate:7.4f} "
                f"{r.treatment_n_trials:4d} {r.control_n_trials:4d}"
            )


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(
        description="Query historical A/B evaluation results",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="SQLAlchemy database URL (default: DATABASE_URL env var)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all submissions with latest result")

    p_latest = sub.add_parser("latest", help="Show latest run for a submission")
    p_latest.add_argument("name", help="Submission name")

    p_history = sub.add_parser("history", help="Show all runs for a submission")
    p_history.add_argument("name", help="Submission name")

    p_compare = sub.add_parser("compare", help="Show trend over time")
    p_compare.add_argument("name", help="Submission name")

    args = parser.parse_args()

    try:
        engine = get_engine(args.database_url)
        session_factory = make_session(engine)
    except Exception:
        logger.exception("Failed to connect to database")
        sys.exit(1)

    if args.command == "list":
        cmd_list(session_factory)
    elif args.command == "latest":
        cmd_latest(session_factory, args.name)
    elif args.command == "history":
        cmd_history(session_factory, args.name)
    elif args.command == "compare":
        cmd_compare(session_factory, args.name)


if __name__ == "__main__":
    main()
