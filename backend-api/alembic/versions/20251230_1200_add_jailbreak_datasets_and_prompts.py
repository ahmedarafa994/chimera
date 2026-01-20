"""Add jailbreak datasets and prompts.

Revision ID: d1d2c3b4e5f6
Revises: 7f25595d4099
Create Date: 2025-12-30 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d1d2c3b4e5f6"
down_revision: str | Sequence[str] | None = "7f25595d4099"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jailbreak_datasets",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("description", sa.String(), nullable=True),
        sa.Column("source_url", sa.String(), nullable=True),
        sa.Column("license", sa.String(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )
    op.create_index(op.f("ix_jailbreak_datasets_id"), "jailbreak_datasets", ["id"], unique=False)
    op.create_index(
        op.f("ix_jailbreak_datasets_name"),
        "jailbreak_datasets",
        ["name"],
        unique=False,
    )

    op.create_table(
        "jailbreak_prompts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("dataset_id", sa.Integer(), nullable=False),
        sa.Column("prompt_text", sa.Text(), nullable=False),
        sa.Column("jailbreak_type", sa.String(), nullable=True),
        sa.Column("is_jailbreak", sa.Boolean(), nullable=True),
        sa.Column("platform", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=True),
        sa.Column("community", sa.String(), nullable=True),
        sa.Column("source_metadata", sa.JSON(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.text("(CURRENT_TIMESTAMP)"),
            nullable=True,
        ),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_id"], ["jailbreak_datasets.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_jailbreak_prompts_dataset_id"),
        "jailbreak_prompts",
        ["dataset_id"],
        unique=False,
    )

    # add column to evasion_tasks (SQLite-safe batch alter)
    with op.batch_alter_table("evasion_tasks", schema=None) as batch_op:
        batch_op.add_column(sa.Column("jailbreak_prompt_id", sa.Integer(), nullable=True))
        batch_op.create_foreign_key(
            "fk_evasion_task_prompt",
            "jailbreak_prompts",
            ["jailbreak_prompt_id"],
            ["id"],
            ondelete="SET NULL",
        )


def downgrade() -> None:
    op.drop_constraint("fk_evasion_task_prompt", "evasion_tasks", type_="foreignkey")
    op.drop_column("evasion_tasks", "jailbreak_prompt_id")
    op.drop_index(op.f("ix_jailbreak_prompts_dataset_id"), table_name="jailbreak_prompts")
    op.drop_table("jailbreak_prompts")
    op.drop_index(op.f("ix_jailbreak_datasets_name"), table_name="jailbreak_datasets")
    op.drop_index(op.f("ix_jailbreak_datasets_id"), table_name="jailbreak_datasets")
    op.drop_table("jailbreak_datasets")
