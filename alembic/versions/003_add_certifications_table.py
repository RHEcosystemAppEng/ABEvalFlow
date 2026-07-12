"""Add certifications table.

Revision ID: 003
Revises: 002
Create Date: 2026-06-29
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "certifications",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("scorecard_id", sa.Uuid(), nullable=False),
        sa.Column("level", sa.String(20), nullable=False),
        sa.Column("passed", sa.Boolean(), nullable=False),
        sa.Column("checks_total", sa.Integer(), nullable=False),
        sa.Column("checks_passed", sa.Integer(), nullable=False),
        sa.Column("checks_failed", sa.Integer(), nullable=False),
        sa.Column(
            "failed_checks",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.Column(
            "details_json",
            sa.JSON().with_variant(postgresql.JSONB(), "postgresql"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["scorecard_id"], ["scorecards.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_certifications_scorecard_id", "certifications", ["scorecard_id"])
    op.create_index("ix_certifications_level_passed", "certifications", ["level", "passed"])
    op.create_unique_constraint("uq_certifications_scorecard_level", "certifications", ["scorecard_id", "level"])


def downgrade() -> None:
    op.drop_constraint("uq_certifications_scorecard_level", "certifications", type_="unique")
    op.drop_index("ix_certifications_level_passed", table_name="certifications")
    op.drop_index("ix_certifications_scorecard_id", table_name="certifications")
    op.drop_table("certifications")
