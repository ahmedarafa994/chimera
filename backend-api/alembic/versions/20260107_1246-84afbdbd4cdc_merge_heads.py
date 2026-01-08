"""merge heads

Revision ID: 84afbdbd4cdc
Revises: d1d2c3b4e5f6
Create Date: 2026-01-07 12:46:51.672549

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '84afbdbd4cdc'
down_revision: Union[str, Sequence[str], None] = 'd1d2c3b4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
