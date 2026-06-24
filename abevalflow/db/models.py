"""SQLAlchemy models for evaluation results persistence.

Three tables:
- ``evaluation_runs``: one row per pipeline run (flattened summary for fast queries)
- ``trials``: one row per trial (drill-down into individual outcomes)
- ``security_scans``: one row per security scan (independent of evaluation runs)
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
)
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


# Portable JSON that upgrades to JSONB on PostgreSQL
_JsonVariant = JSON().with_variant(postgresql.JSONB(), "postgresql")


class EvaluationRun(Base):
    """One row per pipeline run with flattened summary statistics."""

    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    submission_name: Mapped[str] = mapped_column(String(255), nullable=False)
    pipeline_run_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)

    # Provenance
    commit_sha: Mapped[str | None] = mapped_column(String(64))
    treatment_image_ref: Mapped[str | None] = mapped_column(Text)
    control_image_ref: Mapped[str | None] = mapped_column(Text)
    harbor_fork_revision: Mapped[str | None] = mapped_column(String(64))
    eval_engine: Mapped[str] = mapped_column(String(10), nullable=False, default="harbor")

    # Summary
    recommendation: Mapped[str] = mapped_column(String(10), nullable=False)
    uplift: Mapped[float] = mapped_column(Float, nullable=False)
    mean_reward_gap: Mapped[float | None] = mapped_column(Float)
    ttest_p_value: Mapped[float | None] = mapped_column(Float)
    fisher_p_value: Mapped[float | None] = mapped_column(Float)

    # Treatment variant stats
    treatment_n_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    treatment_n_passed: Mapped[int] = mapped_column(Integer, nullable=False)
    treatment_n_failed: Mapped[int] = mapped_column(Integer, nullable=False)
    treatment_n_errors: Mapped[int] = mapped_column(Integer, nullable=False)
    treatment_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    treatment_mean_reward: Mapped[float | None] = mapped_column(Float)
    treatment_median_reward: Mapped[float | None] = mapped_column(Float)
    treatment_std_reward: Mapped[float | None] = mapped_column(Float)

    # Control variant stats
    control_n_trials: Mapped[int] = mapped_column(Integer, nullable=False)
    control_n_passed: Mapped[int] = mapped_column(Integer, nullable=False)
    control_n_failed: Mapped[int] = mapped_column(Integer, nullable=False)
    control_n_errors: Mapped[int] = mapped_column(Integer, nullable=False)
    control_pass_rate: Mapped[float] = mapped_column(Float, nullable=False)
    control_mean_reward: Mapped[float | None] = mapped_column(Float)
    control_median_reward: Mapped[float | None] = mapped_column(Float)
    control_std_reward: Mapped[float | None] = mapped_column(Float)

    # Full report for flexibility / future queries
    report_json: Mapped[dict] = mapped_column(_JsonVariant, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    trials: Mapped[list[Trial]] = relationship(back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_evaluation_runs_submission_name", "submission_name"),
        Index(
            "ix_evaluation_runs_submission_created",
            "submission_name",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        return f"<EvaluationRun {self.submission_name!r} recommendation={self.recommendation!r}>"


class Trial(Base):
    """One row per trial for drill-down queries."""

    __tablename__ = "trials"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    variant: Mapped[str] = mapped_column(String(20), nullable=False)
    trial_name: Mapped[str] = mapped_column(String(255), nullable=False)
    reward: Mapped[float | None] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    run: Mapped[EvaluationRun] = relationship(back_populates="trials")

    __table_args__ = (
        Index("ix_trials_run_id", "run_id"),
        Index("ix_trials_run_variant", "run_id", "variant"),
    )

    def __repr__(self) -> str:
        return f"<Trial {self.trial_name!r} variant={self.variant!r} reward={self.reward!r}>"


class SecurityScan(Base):
    """One row per security scan, independent of evaluation runs.

    Allows querying security findings across all submissions without
    coupling to the A/B evaluation workflow.
    """

    __tablename__ = "security_scans"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    submission_name: Mapped[str] = mapped_column(String(255), nullable=False)

    scanner: Mapped[str] = mapped_column(String(50), nullable=False)
    scan_mode: Mapped[str] = mapped_column(String(10), nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)

    findings_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Only blocking severities (critical, high) are indexed for fast queries.
    # Medium/low/info counts can be derived from findings_json if needed.
    critical_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    high_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    findings_json: Mapped[list] = mapped_column(_JsonVariant, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_security_scans_submission_name", "submission_name"),
        Index(
            "ix_security_scans_submission_created",
            "submission_name",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<SecurityScan {self.submission_name!r} scanner={self.scanner!r} "
            f"passed={self.passed!r} findings={self.findings_count}>"
        )


class MCPCheckerRun(Base):
    """One row per MCPChecker evaluation run.

    MCPChecker evaluates MCP servers/agents directly (single-agent mode),
    unlike Harbor/ASE which use A/B comparison.
    """

    __tablename__ = "mcpchecker_results"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    pipeline_run_id: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    submission_name: Mapped[str] = mapped_column(String(255), nullable=False)
    eval_name: Mapped[str] = mapped_column(String(255), nullable=False)

    overall_score: Mapped[float] = mapped_column(Float, nullable=False)
    passed_tasks: Mapped[int] = mapped_column(Integer, nullable=False)
    failed_tasks: Mapped[int] = mapped_column(Integer, nullable=False)
    error_tasks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    skipped_tasks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_tasks: Mapped[int] = mapped_column(Integer, nullable=False)

    total_duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    raw_output_json: Mapped[dict] = mapped_column(_JsonVariant, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    tasks: Mapped[list[MCPCheckerTask]] = relationship(back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_mcpchecker_results_submission_name", "submission_name"),
        Index(
            "ix_mcpchecker_results_submission_created",
            "submission_name",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<MCPCheckerRun {self.submission_name!r} "
            f"score={self.overall_score:.2f} ({self.passed_tasks}/{self.total_tasks})>"
        )


class MCPCheckerTask(Base):
    """One row per task within an MCPChecker evaluation run."""

    __tablename__ = "mcpchecker_tasks"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("mcpchecker_results.id", ondelete="CASCADE"), nullable=False
    )

    task_id: Mapped[str] = mapped_column(String(255), nullable=False)
    task_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    tool_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    llm_judge_passed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_judge_total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    details_json: Mapped[dict | None] = mapped_column(_JsonVariant, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, default=_utcnow)

    run: Mapped[MCPCheckerRun] = relationship(back_populates="tasks")

    __table_args__ = (
        Index("ix_mcpchecker_tasks_run_id", "run_id"),
        Index("ix_mcpchecker_tasks_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<MCPCheckerTask {self.task_id!r} status={self.status!r} "
            f"judge={self.llm_judge_passed}/{self.llm_judge_total}>"
        )
