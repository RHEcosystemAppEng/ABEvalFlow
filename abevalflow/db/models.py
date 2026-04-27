"""SQLAlchemy models for evaluation results persistence.

Two tables:
- ``evaluation_runs``: one row per pipeline run (flattened summary for fast queries)
- ``trials``: one row per trial (drill-down into individual outcomes)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

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
    return datetime.now(timezone.utc)


# Portable JSON that upgrades to JSONB on PostgreSQL
_JsonVariant = JSON().with_variant(postgresql.JSONB(), "postgresql")


class EvaluationRun(Base):
    """One row per pipeline run with flattened summary statistics."""

    __tablename__ = "evaluation_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    submission_name: Mapped[str] = mapped_column(String(255), nullable=False)
    pipeline_run_id: Mapped[str] = mapped_column(
        String(255), unique=True, nullable=False
    )

    # Provenance
    commit_sha: Mapped[str | None] = mapped_column(String(64))
    treatment_image_ref: Mapped[str | None] = mapped_column(Text)
    control_image_ref: Mapped[str | None] = mapped_column(Text)
    harbor_fork_revision: Mapped[str | None] = mapped_column(String(64))

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

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    trials: Mapped[list[Trial]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_evaluation_runs_submission_name", "submission_name"),
        Index(
            "ix_evaluation_runs_submission_created",
            "submission_name",
            "created_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<EvaluationRun {self.submission_name!r} "
            f"recommendation={self.recommendation!r}>"
        )


class Trial(Base):
    """One row per trial for drill-down queries."""

    __tablename__ = "trials"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid, primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("evaluation_runs.id", ondelete="CASCADE"), nullable=False
    )
    variant: Mapped[str] = mapped_column(String(20), nullable=False)
    trial_name: Mapped[str] = mapped_column(String(255), nullable=False)
    reward: Mapped[float | None] = mapped_column(Float)
    passed: Mapped[bool] = mapped_column(Boolean, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    run: Mapped[EvaluationRun] = relationship(back_populates="trials")

    __table_args__ = (
        Index("ix_trials_run_id", "run_id"),
        Index("ix_trials_run_variant", "run_id", "variant"),
    )

    def __repr__(self) -> str:
        return (
            f"<Trial {self.trial_name!r} variant={self.variant!r} "
            f"reward={self.reward!r}>"
        )
