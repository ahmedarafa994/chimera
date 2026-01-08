"""Add selections table for provider model selection system

Revision ID: 9aa6cbd74e9b
Revises: 84afbdbd4cdc
Create Date: 2026-01-07 13:06:54.583263

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9aa6cbd74e9b'
down_revision: Union[str, Sequence[str], None] = '84afbdbd4cdc'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create selections table for provider/model selection system."""
    # Create selections table with composite primary key
    op.create_table(
        'selections',
        sa.Column('session_id', sa.String(length=255), nullable=True),
        sa.Column('user_id', sa.String(length=255), nullable=True),
        sa.Column('provider_id', sa.String(length=100), nullable=False),
        sa.Column('model_id', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.PrimaryKeyConstraint('session_id', 'user_id')
    )

    # Create indexes for efficient lookups
    op.create_index('idx_selections_session_id', 'selections', ['session_id'])
    op.create_index('idx_selections_user_id', 'selections', ['user_id'])
    op.create_index('idx_selections_provider_model', 'selections', ['provider_id', 'model_id'])
    op.create_index('idx_selections_user_session', 'selections', ['user_id', 'session_id'])


def downgrade() -> None:
    """Drop selections table and indexes."""
    # Drop indexes first
    op.drop_index('idx_selections_user_session', table_name='selections')
    op.drop_index('idx_selections_provider_model', table_name='selections')
    op.drop_index('idx_selections_user_id', table_name='selections')
    op.drop_index('idx_selections_session_id', table_name='selections')

    # Drop table
    op.drop_table('selections')
