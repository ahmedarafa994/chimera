"""Add version and metadata columns to selections table

Adds optimistic concurrency control support via version column
and metadata JSONB column for flexible data storage.

Revision ID: a1b2c3d4e5f6
Revises: 9aa6cbd74e9b
Create Date: 2026-01-07 13:20:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: str | Sequence[str] | None = "9aa6cbd74e9b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add version and metadata columns to selections table."""
    # Add version column for optimistic concurrency control
    # Default to 1 for existing rows
    op.add_column(
        "selections", sa.Column("version", sa.Integer(), nullable=False, server_default="1")
    )

    # Add metadata JSONB column for flexible data storage
    # Use JSON for SQLite compatibility, JSONB for PostgreSQL
    op.add_column(
        "selections", sa.Column("metadata", sa.JSON(), nullable=False, server_default="{}")
    )

    # Add source column to track where selection came from
    op.add_column(
        "selections",
        sa.Column("source", sa.String(length=50), nullable=False, server_default="unknown"),
    )

    # Add expires_at column for session expiration handling
    op.add_column("selections", sa.Column("expires_at", sa.DateTime(), nullable=True))

    # Create index on version for concurrency queries
    op.create_index("idx_selections_version", "selections", ["version"])

    # Create index on expires_at for cleanup queries
    op.create_index("idx_selections_expires_at", "selections", ["expires_at"])

    # Create index on updated_at for sorting/filtering
    op.create_index("idx_selections_updated_at", "selections", ["updated_at"])


def downgrade() -> None:
    """Remove version and metadata columns from selections table."""
    # Drop indexes first
    op.drop_index("idx_selections_updated_at", table_name="selections")
    op.drop_index("idx_selections_expires_at", table_name="selections")
    op.drop_index("idx_selections_version", table_name="selections")

    # Drop columns
    op.drop_column("selections", "expires_at")
    op.drop_column("selections", "source")
    op.drop_column("selections", "metadata")
    op.drop_column("selections", "version")
