"""Add gate_results table.

Revision ID: 002
Revises: 001
Create Date: 2026-06-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "gate_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("scorecard_id", sa.Uuid(), nullable=False),
        sa.Column("gate_name", sa.String(100), nullable=False),
        sa.Column("gate_type", sa.String(20), nullable=False),
        sa.Column("policy_key", sa.String(100), nullable=True),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("mode", sa.String(20), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=True),
        sa.Column("findings_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("message", sa.Text(), nullable=True),
        sa.Column(
            "details_json",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        sa.Column("evaluated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["scorecard_id"], ["scorecards.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_gate_results_scorecard_id", "gate_results", ["scorecard_id"])
    op.create_index("ix_gate_results_scorecard_policy", "gate_results", ["scorecard_id", "policy_key"])
    op.create_index("ix_gate_results_gate_passed", "gate_results", ["gate_name", "passed"])
    op.create_index("ix_gate_results_policy_key", "gate_results", ["policy_key"])


def downgrade() -> None:
    op.drop_index("ix_gate_results_policy_key", table_name="gate_results")
    op.drop_index("ix_gate_results_gate_passed", table_name="gate_results")
    op.drop_index("ix_gate_results_scorecard_policy", table_name="gate_results")
    op.drop_index("ix_gate_results_scorecard_id", table_name="gate_results")
    op.drop_table("gate_results")
