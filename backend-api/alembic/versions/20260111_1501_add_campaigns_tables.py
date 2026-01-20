"""Add campaigns and campaign sharing tables.

This migration adds support for campaign ownership and sharing:
- campaigns table with owner_id foreign key to users
- campaign_shares table for explicit user sharing with permission levels

Revision ID: c2d3e4f5a6b7
Revises: b1c2d3e4f5a6
Create Date: 2026-01-11 15:01:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c2d3e4f5a6b7"
down_revision: str | Sequence[str] | None = "b1c2d3e4f5a6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create campaigns and campaign_shares tables."""
    # Step 1: Create campaigns table
    op.create_table(
        "campaigns",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("owner_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "visibility",
            sa.String(length=20),
            nullable=False,
            server_default="private",
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("is_archived", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("config", sa.JSON(), nullable=False, server_default="{}"),
        sa.Column("target_provider", sa.String(length=50), nullable=True),
        sa.Column("target_model", sa.String(length=100), nullable=True),
        sa.Column("total_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("successful_attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        ),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("last_activity_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name="fk_campaigns_owner_id",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for campaigns
    op.create_index(op.f("ix_campaigns_id"), "campaigns", ["id"], unique=False)
    op.create_index("ix_campaigns_owner_id", "campaigns", ["owner_id"], unique=False)
    op.create_index(
        "ix_campaigns_owner_active",
        "campaigns",
        ["owner_id", "is_active"],
        unique=False,
    )
    op.create_index("ix_campaigns_visibility", "campaigns", ["visibility"], unique=False)
    op.create_index(
        "ix_campaigns_owner_visibility",
        "campaigns",
        ["owner_id", "visibility"],
        unique=False,
    )
    op.create_index("ix_campaigns_created_at", "campaigns", ["created_at"], unique=False)

    # Step 2: Create campaign_shares table
    op.create_table(
        "campaign_shares",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("campaign_id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "permission",
            sa.String(length=10),
            nullable=False,
            server_default="view",
        ),
        sa.Column("shared_by_id", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
        ),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["campaign_id"],
            ["campaigns.id"],
            name="fk_campaign_shares_campaign_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name="fk_campaign_shares_user_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["shared_by_id"],
            ["users.id"],
            name="fk_campaign_shares_shared_by_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for campaign_shares
    op.create_index(op.f("ix_campaign_shares_id"), "campaign_shares", ["id"], unique=False)
    op.create_index(
        "ix_campaign_shares_campaign_id",
        "campaign_shares",
        ["campaign_id"],
        unique=False,
    )
    op.create_index("ix_campaign_shares_user_id", "campaign_shares", ["user_id"], unique=False)
    op.create_index(
        "ix_campaign_shares_unique",
        "campaign_shares",
        ["campaign_id", "user_id"],
        unique=True,
    )
    op.create_index(
        "ix_campaign_shares_user_permission",
        "campaign_shares",
        ["user_id", "permission"],
        unique=False,
    )


def downgrade() -> None:
    """Drop campaigns and campaign_shares tables."""
    # Drop campaign_shares indexes and table first (due to FK dependency)
    op.drop_index("ix_campaign_shares_user_permission", table_name="campaign_shares")
    op.drop_index("ix_campaign_shares_unique", table_name="campaign_shares")
    op.drop_index("ix_campaign_shares_user_id", table_name="campaign_shares")
    op.drop_index("ix_campaign_shares_campaign_id", table_name="campaign_shares")
    op.drop_index(op.f("ix_campaign_shares_id"), table_name="campaign_shares")
    op.drop_table("campaign_shares")

    # Drop campaigns indexes and table
    op.drop_index("ix_campaigns_created_at", table_name="campaigns")
    op.drop_index("ix_campaigns_owner_visibility", table_name="campaigns")
    op.drop_index("ix_campaigns_visibility", table_name="campaigns")
    op.drop_index("ix_campaigns_owner_active", table_name="campaigns")
    op.drop_index("ix_campaigns_owner_id", table_name="campaigns")
    op.drop_index(op.f("ix_campaigns_id"), table_name="campaigns")
    op.drop_table("campaigns")
