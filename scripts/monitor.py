#!/usr/bin/env python3
"""Monitor skill performance and detect degradation.

Queries the last N evaluation runs for a submission and compares scores
to detect performance degradation. Outputs a JSON payload indicating
whether degradation was detected.

Exit codes:
    0: No degradation or not enough data
    1: Degradation detected
    2: Error (DB connection, invalid args, etc.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from statistics import mean, variance as stat_variance
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

if TYPE_CHECKING:
    from sqlalchemy.engine import Engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MonitorResult:
    """Result of a monitoring check."""

    submission_name: str
    degraded: bool
    current_score: float | None
    previous_score: float | None
    ratio: float | None
    threshold: float
    message: str
    current_run_id: str | None = None
    previous_run_id: str | None = None


# TODO: Wire into aggregate_scorecard._extract_behavioral_data()
# to feed stability sub-check in behavioral testing certification.
def get_historical_variance(
    engine: Engine,
    submission_name: str,
    window: int = 5,
    eval_engine: str | None = None,
) -> dict:
    """Query last N evaluation runs and compute score variance.

    Args:
        engine: SQLAlchemy engine connected to the evaluation DB.
        submission_name: The skill/submission name to check.
        window: Number of recent runs to consider.
        eval_engine: Optional engine filter.

    Returns:
        Dict with score_variance, run_count, scores, mean_score.
    """
    from abevalflow.db.models import EvaluationRun

    with Session(engine) as session:
        stmt = select(EvaluationRun).where(EvaluationRun.submission_name == submission_name)
        if eval_engine is not None:
            stmt = stmt.where(EvaluationRun.eval_engine == eval_engine)
        stmt = stmt.order_by(EvaluationRun.created_at.desc()).limit(window)
        runs = list(session.execute(stmt).scalars().all())

    scores = [r.treatment_pass_rate for r in runs if r.treatment_pass_rate is not None]
    run_count = len(scores)

    if run_count < 2:
        return {
            "score_variance": 0.0,
            "run_count": run_count,
            "scores": scores,
            "mean_score": scores[0] if scores else None,
        }

    mean_score = mean(scores)
    variance = stat_variance(scores)

    return {
        "score_variance": variance,
        "run_count": run_count,
        "scores": scores,
        "mean_score": mean_score,
    }


def check_degradation(
    engine: Engine,
    submission_name: str,
    threshold: float = 0.85,
    current_score: float | None = None,
    eval_engine: str | None = None,
    run_id: str | None = None,
) -> MonitorResult:
    """Check if the most recent run shows degradation vs the previous run.

    Args:
        engine: SQLAlchemy engine connected to the evaluation DB.
        submission_name: The skill/submission name to check.
        threshold: Ratio threshold below which degradation is flagged.
            E.g., 0.85 means current score must be at least 85% of previous.
        current_score: Optional score for the current run (not yet in DB).
            When provided, only the most recent historical run is queried.
        eval_engine: Optional engine filter (e.g. ``harbor``, ``a2a``).

    Returns:
        MonitorResult with degradation status and scores.
    """
    from abevalflow.db.models import EvaluationRun

    external_current_score = current_score is not None
    limit = 1 if external_current_score else 2
    min_runs = 1 if external_current_score else 2

    with Session(engine) as session:
        stmt = select(EvaluationRun).where(EvaluationRun.submission_name == submission_name)
        if eval_engine is not None:
            stmt = stmt.where(EvaluationRun.eval_engine == eval_engine)
        stmt = stmt.order_by(EvaluationRun.created_at.desc()).limit(limit)
        runs = list(session.execute(stmt).scalars().all())

    if len(runs) < min_runs:
        return MonitorResult(
            submission_name=submission_name,
            degraded=False,
            current_score=(
                current_score if current_score is not None else (runs[0].treatment_pass_rate if runs else None)
            ),
            previous_score=None,
            ratio=None,
            threshold=threshold,
            message=(f"Not enough data: found {len(runs)} run(s), need at least {min_runs}"),
            current_run_id=None if external_current_score else (runs[0].pipeline_run_id if runs else None),
            previous_run_id=None,
        )

    if external_current_score:
        previous_run = runs[0]
        previous_score = previous_run.treatment_pass_rate
    else:
        current_run, previous_run = runs[0], runs[1]
        current_score = current_run.treatment_pass_rate
        previous_score = previous_run.treatment_pass_rate

    if previous_score == 0:
        ratio = 0.0 if current_score == 0 else float("inf")
    else:
        ratio = current_score / previous_score

    degraded = ratio < threshold

    if degraded:
        message = (
            f"Degradation detected: score dropped from {previous_score:.2%} "
            f"to {current_score:.2%} (ratio: {ratio:.2f} < {threshold})"
        )
    else:
        message = f"No degradation: score is {current_score:.2%} (ratio: {ratio:.2f} >= {threshold})"

    return MonitorResult(
        submission_name=submission_name,
        degraded=degraded,
        current_score=current_score,
        previous_score=previous_score,
        ratio=ratio,
        threshold=threshold,
        message=message,
        current_run_id=run_id if external_current_score else current_run.pipeline_run_id,
        previous_run_id=previous_run.pipeline_run_id,
    )


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Monitor skill performance and detect degradation",
    )
    parser.add_argument(
        "--submission-name",
        required=True,
        help="Skill/submission name to monitor",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Ratio threshold for degradation (default: 0.85)",
    )
    parser.add_argument(
        "--db-url",
        required=True,
        help="PostgreSQL connection URL",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--current-score",
        type=float,
        default=None,
        help=(
            "Current run score (not yet in DB). When set, only the most "
            "recent historical run is queried for comparison."
        ),
    )
    parser.add_argument(
        "--eval-engine",
        default=None,
        help="Filter historical runs by eval engine (e.g. harbor, a2a)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Current pipeline run ID (used as current_run_id in output when --current-score is set)",
    )

    args = parser.parse_args()

    try:
        engine = create_engine(args.db_url)
        result = check_degradation(
            engine=engine,
            submission_name=args.submission_name,
            threshold=args.threshold,
            current_score=args.current_score,
            eval_engine=args.eval_engine,
            run_id=args.run_id,
        )
    except Exception as e:
        logger.exception("Error during monitoring check")
        error_result = {
            "submission_name": args.submission_name,
            "degraded": False,
            "error": str(e),
            "message": f"Monitoring check failed: {e}",
        }
        print(json.dumps(error_result, indent=2))
        return 2

    output_json = json.dumps(asdict(result), indent=2)

    if args.output == "-":
        print(output_json)
    else:
        with open(args.output, "w") as f:
            f.write(output_json)
        logger.info("Wrote result to %s", args.output)

    logger.info(result.message)

    return 1 if result.degraded else 0


if __name__ == "__main__":
    sys.exit(main())
