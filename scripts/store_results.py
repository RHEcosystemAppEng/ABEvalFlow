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
from abevalflow.db.models import EvaluationRun, MCPCheckerRun, MCPCheckerTask, SecurityScan, Trial
from abevalflow.db.observer import discover_observers, notify_observers
from abevalflow.mcpchecker_report import MCPCheckerResult
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
        eval_engine=p.eval_engine,
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


def map_security_scans(
    result: AnalysisResult,
    run_id: str,
) -> list[SecurityScan]:
    """Create SecurityScan rows from security_scans in the report."""
    scans: list[SecurityScan] = []
    for scan in result.security_scans:
        findings_list = [f.model_dump() for f in scan.findings]
        scans.append(
            SecurityScan(
                pipeline_run_id=run_id,
                submission_name=result.submission_name,
                scanner=scan.scanner,
                scan_mode=scan.scan_mode.value,
                passed=scan.passed,
                findings_count=len(scan.findings),
                critical_count=scan.severity_counts.get("critical", 0),
                high_count=scan.severity_counts.get("high", 0),
                findings_json=findings_list,
            )
        )
    return scans


def map_mcpchecker_result(result: MCPCheckerResult, run_id: str) -> MCPCheckerRun:
    """Create MCPCheckerRun row from MCPCheckerResult."""
    return MCPCheckerRun(
        pipeline_run_id=run_id,
        submission_name=result.submission_name,
        eval_name=result.eval_name,
        overall_score=result.overall_score,
        passed_tasks=result.passed_tasks,
        failed_tasks=result.failed_tasks,
        error_tasks=result.error_tasks,
        skipped_tasks=result.skipped_tasks,
        total_tasks=result.total_tasks,
        total_duration_ms=result.total_duration_ms,
        raw_output_json=json.loads(result.model_dump_json()),
    )


def map_mcpchecker_tasks(result: MCPCheckerResult, run: MCPCheckerRun) -> list[MCPCheckerTask]:
    """Create MCPCheckerTask rows from task results."""
    tasks: list[MCPCheckerTask] = []
    for task in result.tasks:
        llm_judge_passed = sum(1 for r in task.llm_judge_results if r.passed)
        llm_judge_total = len(task.llm_judge_results)

        tasks.append(
            MCPCheckerTask(
                run=run,
                task_id=task.task_id,
                task_name=task.task_name,
                status=task.status,
                tool_calls=task.tool_calls,
                duration_ms=task.duration_ms,
                error_message=task.error_message,
                llm_judge_passed=llm_judge_passed,
                llm_judge_total=llm_judge_total,
                details_json=task.model_dump() if task.llm_judge_results or task.tool_call_records else None,
            )
        )
    return tasks


def store_mcpchecker(
    report_dir: Path,
    database_url: str | None = None,
    run_id: str | None = None,
) -> bool:
    """Load, validate, and persist an MCPChecker report. Returns True on success."""
    report_path = report_dir / "mcpchecker-report.json"
    if not report_path.exists():
        logger.error("MCPChecker report not found: %s", report_path)
        return False

    raw = report_path.read_bytes()
    try:
        result = MCPCheckerResult.model_validate_json(raw)
    except Exception:
        logger.exception("Failed to validate MCPChecker report JSON")
        return False

    effective_run_id = run_id or _compute_content_hash(raw)
    logger.info("MCPChecker Run ID: %s", effective_run_id)

    engine = get_engine(database_url)
    init_db(engine)
    session_factory = make_session(engine)

    with session_factory() as session:
        existing = session.execute(
            select(MCPCheckerRun).where(MCPCheckerRun.pipeline_run_id == effective_run_id)
        ).scalar_one_or_none()

        if existing is not None:
            logger.warning(
                "MCPChecker run %s already exists (id=%s) — skipping",
                effective_run_id,
                existing.id,
            )
            return True

        mcp_run = map_mcpchecker_result(result, effective_run_id)
        mcp_tasks = map_mcpchecker_tasks(result, mcp_run)

        session.add(mcp_run)
        session.add_all(mcp_tasks)
        try:
            session.commit()
        except IntegrityError as e:
            session.rollback()
            logger.warning(
                "Concurrent insert for MCPChecker run %s — treating as idempotent: %s",
                effective_run_id,
                str(e)[:100],
            )
            return True

        logger.info(
            "Stored MCPChecker: submission=%s run_id=%s tasks=%d score=%.2f recommendation=%s",
            result.submission_name,
            effective_run_id,
            len(mcp_tasks),
            result.overall_score,
            result.recommendation,
        )

    return True


def store(
    report_dir: Path,
    database_url: str | None = None,
    run_id: str | None = None,
) -> bool:
    """Load, validate, and persist a report. Returns True on success.

    Supports both A/B reports (report.json) and MCPChecker reports
    (mcpchecker-report.json). Checks for both and stores whichever exists.
    """
    report_path = report_dir / "report.json"
    mcpchecker_report_path = report_dir / "mcpchecker-report.json"

    if mcpchecker_report_path.exists():
        logger.info("Found MCPChecker report, storing to mcpchecker_results table")
        return store_mcpchecker(report_dir, database_url, run_id)

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
            select(EvaluationRun).where(EvaluationRun.pipeline_run_id == effective_run_id)
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
        security_scans = map_security_scans(result, effective_run_id)

        # Check if security scans already exist (may have been inserted by
        # security-scan task's immediate persistence step)
        existing_scan = session.execute(
            select(SecurityScan).where(SecurityScan.pipeline_run_id == effective_run_id)
        ).scalar_one_or_none()

        session.add(ev_run)
        session.add_all(trials)
        if existing_scan is None:
            session.add_all(security_scans)
        else:
            logger.info(
                "Security scans already exist for run %s, skipping",
                effective_run_id,
            )
        try:
            session.commit()
        except IntegrityError as e:
            session.rollback()
            logger.warning(
                "Concurrent insert for run %s — treating as idempotent: %s",
                effective_run_id,
                str(e)[:100],
            )
            return True

        logger.info(
            "Stored: submission=%s run_id=%s trials=%d security_scans=%d recommendation=%s",
            result.submission_name,
            effective_run_id,
            len(trials),
            len(security_scans),
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
