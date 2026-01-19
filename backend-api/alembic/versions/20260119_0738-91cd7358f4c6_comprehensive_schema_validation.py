"""comprehensive_schema_validation

Revision ID: 91cd7358f4c6
Revises: c2d3e4f5a6b7
Create Date: 2026-01-19 07:38:15.598943

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "91cd7358f4c6"
down_revision: Union[str, Sequence[str], None] = "c2d3e4f5a6b7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Comprehensive schema validation and missing table creation.

    This migration ensures all required tables exist with proper indexes,
    constraints, and relationships for the Chimera platform.
    """

    # Get connection to check existing tables
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    existing_tables = inspector.get_table_names()

    # =========================================================================
    # Provider and Model Management Tables
    # =========================================================================

    if "llm_providers" not in existing_tables:
        op.create_table(
            "llm_providers",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(100), nullable=False, unique=True),
            sa.Column("display_name", sa.String(200), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("provider_type", sa.String(50), nullable=False),
            sa.Column("base_url", sa.String(500), nullable=True),
            sa.Column("api_version", sa.String(20), nullable=True),
            sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
            sa.Column("health_status", sa.String(20), server_default="unknown"),
            sa.Column("last_health_check", sa.DateTime(), nullable=True),
            sa.Column("default_config", sa.JSON(), server_default="{}"),
            sa.Column("required_config_keys", sa.JSON(), server_default="[]"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("(CURRENT_TIMESTAMP)")),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
        )
        op.create_index("ix_llm_providers_name", "llm_providers", ["name"], unique=True)
        op.create_index("ix_llm_providers_active", "llm_providers", ["is_active"])

    if "llm_models" not in existing_tables:
        op.create_table(
            "llm_models",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("provider_id", sa.Integer(), nullable=False),
            sa.Column("model_name", sa.String(100), nullable=False),
            sa.Column("display_name", sa.String(200), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("max_tokens", sa.Integer(), nullable=True),
            sa.Column("supports_function_calling", sa.Boolean(), server_default="0"),
            sa.Column("supports_streaming", sa.Boolean(), server_default="0"),
            sa.Column("supports_json_mode", sa.Boolean(), server_default="0"),
            sa.Column("cost_per_input_token", sa.Numeric(12, 8), nullable=True),
            sa.Column("cost_per_output_token", sa.Numeric(12, 8), nullable=True),
            sa.Column("model_version", sa.String(50), nullable=True),
            sa.Column("release_date", sa.Date(), nullable=True),
            sa.Column("deprecation_date", sa.Date(), nullable=True),
            sa.Column("is_available", sa.Boolean(), nullable=False, server_default="1"),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("(CURRENT_TIMESTAMP)")),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["provider_id"], ["llm_providers.id"], ondelete="CASCADE"),
            sa.UniqueConstraint("provider_id", "model_name", name="uq_provider_model"),
        )
        op.create_index("ix_llm_models_provider", "llm_models", ["provider_id"])
        op.create_index("ix_llm_models_available", "llm_models", ["is_available"])

    # =========================================================================
    # Campaign Execution and Telemetry Tables
    # =========================================================================

    if "campaign_executions" not in existing_tables:
        op.create_table(
            "campaign_executions",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("campaign_id", sa.Integer(), nullable=False),
            sa.Column("iteration_number", sa.Integer(), nullable=False),
            sa.Column("execution_status", sa.String(20), nullable=False, server_default="pending"),
            sa.Column("original_prompt", sa.Text(), nullable=False),
            sa.Column("transformed_prompt", sa.Text(), nullable=True),
            sa.Column("technique_applied", sa.String(100), nullable=True),
            sa.Column("technique_config", sa.JSON(), server_default="{}"),
            sa.Column("llm_response", sa.Text(), nullable=True),
            sa.Column("success_score", sa.Numeric(5, 4), nullable=True),
            sa.Column("is_successful", sa.Boolean(), nullable=True),
            sa.Column("latency_ms", sa.Integer(), nullable=True),
            sa.Column("token_count_input", sa.Integer(), nullable=True),
            sa.Column("token_count_output", sa.Integer(), nullable=True),
            sa.Column("cost_usd", sa.Numeric(10, 6), nullable=True),
            sa.Column("provider_used", sa.String(50), nullable=True),
            sa.Column("model_used", sa.String(100), nullable=True),
            sa.Column("error_message", sa.Text(), nullable=True),
            sa.Column("error_code", sa.String(50), nullable=True),
            sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.text("(CURRENT_TIMESTAMP)")),
            sa.Column("completed_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["campaign_id"], ["campaigns.id"], ondelete="CASCADE"),
            sa.UniqueConstraint("campaign_id", "iteration_number", name="uq_campaign_iteration"),
        )
        op.create_index("ix_campaign_executions_campaign", "campaign_executions", ["campaign_id"])
        op.create_index("ix_campaign_executions_status", "campaign_executions", ["execution_status"])
        op.create_index("ix_campaign_executions_success", "campaign_executions", ["is_successful"])
        op.create_index("ix_campaign_executions_started", "campaign_executions", ["started_at"])

    # =========================================================================
    # Prompt Template Library Tables
    # =========================================================================

    if "prompt_templates" not in existing_tables:
        op.create_table(
            "prompt_templates",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("title", sa.String(255), nullable=False),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("prompt_text", sa.Text(), nullable=False),
            sa.Column("technique_type", sa.String(50), nullable=False),
            sa.Column("vulnerability_type", sa.String(50), nullable=True),
            sa.Column("tags", sa.JSON(), server_default="[]"),
            sa.Column("template_status", sa.String(20), nullable=False, server_default="draft"),
            sa.Column("sharing_level", sa.String(20), nullable=False, server_default="private"),
            sa.Column("version_number", sa.Integer(), nullable=False, server_default="1"),
            sa.Column("parent_template_id", sa.Integer(), nullable=True),
            sa.Column("is_latest_version", sa.Boolean(), nullable=False, server_default="1"),
            sa.Column("success_rate", sa.Numeric(5, 4), nullable=True),
            sa.Column("usage_count", sa.Integer(), server_default="0"),
            sa.Column("average_rating", sa.Numeric(3, 2), nullable=True),
            sa.Column("total_ratings", sa.Integer(), server_default="0"),
            sa.Column("created_by", sa.Integer(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.text("(CURRENT_TIMESTAMP)")),
            sa.Column("updated_at", sa.DateTime(), nullable=True),
            sa.Column("deleted_at", sa.DateTime(), nullable=True),
            sa.ForeignKeyConstraint(["created_by"], ["users.id"], ondelete="CASCADE"),
            sa.ForeignKeyConstraint(["parent_template_id"], ["prompt_templates.id"], ondelete="SET NULL"),
        )
        op.create_index("ix_prompt_templates_status", "prompt_templates", ["template_status"])
        op.create_index("ix_prompt_templates_sharing", "prompt_templates", ["sharing_level"])
        op.create_index("ix_prompt_templates_technique", "prompt_templates", ["technique_type"])
        op.create_index("ix_prompt_templates_created_by", "prompt_templates", ["created_by"])
        op.create_index("ix_prompt_templates_created_at", "prompt_templates", ["created_at"])

    # =========================================================================
    # Audit and Compliance Tables
    # =========================================================================

    if "audit_log" not in existing_tables:
        op.create_table(
            "audit_log",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("event_id", sa.String(36), nullable=False, unique=True),
            sa.Column("action", sa.String(50), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=True),
            sa.Column("session_id", sa.String(255), nullable=True),
            sa.Column("resource_type", sa.String(100), nullable=True),
            sa.Column("resource_id", sa.String(255), nullable=True),
            sa.Column("details", sa.JSON(), nullable=True),
            sa.Column("severity", sa.String(20), nullable=False, server_default="info"),
            sa.Column("ip_address", sa.String(45), nullable=True),
            sa.Column("user_agent", sa.Text(), nullable=True),
            sa.Column("request_id", sa.String(255), nullable=True),
            sa.Column("event_timestamp", sa.DateTime(), nullable=False, server_default=sa.text("(CURRENT_TIMESTAMP)")),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="SET NULL"),
        )
        op.create_index("ix_audit_log_user", "audit_log", ["user_id"])
        op.create_index("ix_audit_log_action", "audit_log", ["action"])
        op.create_index("ix_audit_log_timestamp", "audit_log", ["event_timestamp"])
        op.create_index("ix_audit_log_severity", "audit_log", ["severity"])
        op.create_index("ix_audit_log_resource", "audit_log", ["resource_type", "resource_id"])


def downgrade() -> None:
    """
    Downgrade schema by removing tables created in upgrade.

    Note: This only removes tables created by this migration.
    Core tables (users, campaigns) are preserved.
    """

    # Drop tables in reverse order of creation (respecting foreign keys)
    op.drop_table("audit_log")
    op.drop_table("prompt_templates")
    op.drop_table("campaign_executions")
    op.drop_table("llm_models")
    op.drop_table("llm_providers")
