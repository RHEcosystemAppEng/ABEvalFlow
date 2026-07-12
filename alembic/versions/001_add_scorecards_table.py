"""Add scorecards table.

Revision ID: 001
Revises: None
Create Date: 2026-06-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "scorecards",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("pipeline_run_id", sa.String(255), nullable=False),
        sa.Column("submission_name", sa.String(255), nullable=False),
        sa.Column("eval_engine", sa.String(50), nullable=False),
        sa.Column("recommendation", sa.String(20), nullable=False),
        sa.Column("recommendation_reason", sa.Text(), nullable=False),
        sa.Column("combination_mode", sa.String(20), nullable=False),
        sa.Column("gates_passed", sa.Integer(), nullable=False),
        sa.Column("gates_failed", sa.Integer(), nullable=False),
        sa.Column("blocking_gates_passed", sa.Integer(), nullable=False),
        sa.Column("blocking_gates_failed", sa.Integer(), nullable=False),
        sa.Column("highest_certification", sa.String(20), nullable=False),
        sa.Column(
            "scorecard_json",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("pipeline_run_id"),
    )
    op.create_index("ix_scorecards_submission_name", "scorecards", ["submission_name"])
    op.create_index("ix_scorecards_submission_created", "scorecards", ["submission_name", "created_at"])
    op.create_index("ix_scorecards_highest_certification", "scorecards", ["highest_certification"])


def downgrade() -> None:
    op.drop_index("ix_scorecards_highest_certification", table_name="scorecards")
    op.drop_index("ix_scorecards_submission_created", table_name="scorecards")
    op.drop_index("ix_scorecards_submission_name", table_name="scorecards")
    op.drop_table("scorecards")
