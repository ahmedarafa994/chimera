"""
Database Infrastructure Module

Story 1.3: Configuration Persistence System
Provides async SQLite database layer with connection pooling.
"""

from app.infrastructure.database.connection import (
    DatabaseConnection,
    get_database,
    initialize_database,
    shutdown_database,
)
from app.infrastructure.database.schema import (
    DatabaseSchema,
    get_schema_manager,
    initialize_schema,
)
from app.infrastructure.database.unit_of_work import (
    UnitOfWork,
    create_unit_of_work,
)
from app.infrastructure.database.campaign_models import (
    Campaign,
    CampaignResult,
    CampaignStatus,
    CampaignTelemetryEvent,
    ExecutionStatus,
    CAMPAIGN_SCHEMA_SQL,
)

__all__ = [
    # Core database infrastructure
    "DatabaseConnection",
    "DatabaseSchema",
    "UnitOfWork",
    "create_unit_of_work",
    "get_database",
    "get_schema_manager",
    "initialize_database",
    "initialize_schema",
    "shutdown_database",
    # Campaign models
    "Campaign",
    "CampaignResult",
    "CampaignStatus",
    "CampaignTelemetryEvent",
    "ExecutionStatus",
    "CAMPAIGN_SCHEMA_SQL",
]
