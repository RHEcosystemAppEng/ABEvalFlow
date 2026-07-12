"""Add observability_metrics table.

Revision ID: 004
Revises: 003
Create Date: 2026-06-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "observability_metrics",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_run_id", sa.String(255), nullable=False),
        sa.Column("submission_name", sa.String(255), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=True),
        sa.Column("pipeline_duration_ms", sa.Integer(), nullable=True),
        sa.Column("prepare_duration_ms", sa.Integer(), nullable=True),
        sa.Column("test_duration_ms", sa.Integer(), nullable=True),
        sa.Column("evaluate_duration_ms", sa.Integer(), nullable=True),
        sa.Column("analyze_duration_ms", sa.Integer(), nullable=True),
        sa.Column("store_duration_ms", sa.Integer(), nullable=True),
        sa.Column("total_prompt_tokens", sa.BigInteger(), nullable=True),
        sa.Column("total_completion_tokens", sa.BigInteger(), nullable=True),
        sa.Column("total_tokens", sa.BigInteger(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("llm_calls_count", sa.Integer(), nullable=True),
        sa.Column("trials_count", sa.Integer(), nullable=True),
        sa.Column("attempt_number", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pipeline_run_id", "attempt_number", name="uq_obs_metrics_run_attempt"),
    )
    op.create_index("ix_observability_metrics_submission_name", "observability_metrics", ["submission_name"])
    op.create_index("ix_observability_metrics_created_at", "observability_metrics", ["created_at"])
    op.create_index("ix_observability_metrics_model_name", "observability_metrics", ["model_name"])


def downgrade() -> None:
    op.drop_index("ix_observability_metrics_model_name", table_name="observability_metrics")
    op.drop_index("ix_observability_metrics_created_at", table_name="observability_metrics")
    op.drop_index("ix_observability_metrics_submission_name", table_name="observability_metrics")
    op.drop_table("observability_metrics")
