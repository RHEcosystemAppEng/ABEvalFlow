"""Persist an A/B evaluation report to the results database.

Usage::

    python scripts/store_results.py \\
        --report-dir /path/to/report-dir \\
        --database-url postgresql+psycopg://user:pass@host:5432/abevalflow

The script reads ``{report-dir}/report.json``, validates it against the
``AnalysisResult`` Pydantic model, and inserts one ``EvaluationRun`` row
plus one ``Trial`` row per trial into the database.

Idempotency: if ``pipeline_run_id`` already exists, the insert is skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from abevalflow.db.engine import get_engine, init_db, make_session
from abevalflow.db.models import EvaluationRun, Trial
from abevalflow.db.observer import discover_observers, notify_observers
from abevalflow.report import AnalysisResult

logger = logging.getLogger(__name__)


def _compute_content_hash(data: bytes) -> str:
    """Deterministic fallback key when no pipeline_run_id is provided."""
    return f"content-{hashlib.sha256(data).hexdigest()[:16]}"


def map_result_to_run(result: AnalysisResult, run_id: str) -> EvaluationRun:
    """Flatten an AnalysisResult into an EvaluationRun row."""
    s = result.summary
    p = result.provenance
    t = s.treatment
    c = s.control

    return EvaluationRun(
        submission_name=result.submission_name,
        pipeline_run_id=run_id,
        commit_sha=p.commit_sha,
        treatment_image_ref=p.treatment_image_ref,
        control_image_ref=p.control_image_ref,
        harbor_fork_revision=p.harbor_fork_revision,
        recommendation=s.recommendation.value,
        uplift=s.uplift,
        mean_reward_gap=s.mean_reward_gap,
        ttest_p_value=s.ttest_p_value,
        fisher_p_value=s.fisher_p_value,
        treatment_n_trials=t.n_trials,
        treatment_n_passed=t.n_passed,
        treatment_n_failed=t.n_failed,
        treatment_n_errors=t.n_errors,
        treatment_pass_rate=t.pass_rate,
        treatment_mean_reward=t.mean_reward,
        treatment_median_reward=t.median_reward,
        treatment_std_reward=t.std_reward,
        control_n_trials=c.n_trials,
        control_n_passed=c.n_passed,
        control_n_failed=c.n_failed,
        control_n_errors=c.n_errors,
        control_pass_rate=c.pass_rate,
        control_mean_reward=c.mean_reward,
        control_median_reward=c.median_reward,
        control_std_reward=c.std_reward,
        report_json=json.loads(result.model_dump_json()),
    )


def map_trials(result: AnalysisResult, run: EvaluationRun) -> list[Trial]:
    """Create Trial rows from the per-variant trial lists."""
    trials: list[Trial] = []
    for variant, trial_list in result.trials.items():
        for tr in trial_list:
            dumped = tr.model_dump()
            trials.append(
                Trial(
                    run=run,
                    variant=variant,
                    trial_name=dumped["trial_name"],
                    reward=dumped["reward"],
                    passed=dumped["passed"],
                )
            )
    return trials


def store(
    report_dir: Path,
    database_url: str | None = None,
    run_id: str | None = None,
) -> bool:
    """Load, validate, and persist a report. Returns True on success."""
    report_path = report_dir / "report.json"
    if not report_path.exists():
        logger.error("Report not found: %s", report_path)
        return False

    raw = report_path.read_bytes()
    try:
        result = AnalysisResult.model_validate_json(raw)
    except Exception:
        logger.exception("Failed to validate report JSON")
        return False

    effective_run_id = run_id or _compute_content_hash(raw)
    logger.info("Run ID: %s", effective_run_id)

    engine = get_engine(database_url)
    init_db(engine)
    session_factory = make_session(engine)

    with session_factory() as session:
        existing = session.execute(
            select(EvaluationRun).where(
                EvaluationRun.pipeline_run_id == effective_run_id
            )
        ).scalar_one_or_none()

        if existing is not None:
            logger.warning(
                "Run %s already exists (id=%s) — skipping",
                effective_run_id,
                existing.id,
            )
            return True

        ev_run = map_result_to_run(result, effective_run_id)
        trials = map_trials(result, ev_run)

        session.add(ev_run)
        session.add_all(trials)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            logger.warning(
                "Concurrent insert for run %s — treating as idempotent",
                effective_run_id,
            )
            return True

        logger.info(
            "Stored: submission=%s run_id=%s trials=%d recommendation=%s",
            result.submission_name,
            effective_run_id,
            len(trials),
            result.summary.recommendation.value,
        )

        observers = discover_observers()
        if observers:
            notify_observers(observers, result, ev_run.id)

    return True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Persist A/B evaluation results to the database",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        required=True,
        help="Directory containing report.json",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=None,
        help="SQLAlchemy database URL (default: DATABASE_URL env var)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Pipeline run ID for idempotency (default: content hash of report)",
    )
    args = parser.parse_args()

    ok = store(args.report_dir, args.database_url, args.run_id)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
