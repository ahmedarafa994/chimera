"""Database Infrastructure Module.

Story 1.3: Configuration Persistence System
Provides async SQLite database layer with connection pooling.
"""

from app.infrastructure.database.campaign_models import (
    CAMPAIGN_SCHEMA_SQL,
    Campaign,
    CampaignResult,
    CampaignStatus,
    CampaignTelemetryEvent,
    ExecutionStatus,
)
from app.infrastructure.database.connection import (
    DatabaseConnection,
    get_database,
    initialize_database,
    shutdown_database,
)
from app.infrastructure.database.schema import DatabaseSchema, get_schema_manager, initialize_schema
from app.infrastructure.database.unit_of_work import UnitOfWork, create_unit_of_work

__all__ = [
    "CAMPAIGN_SCHEMA_SQL",
    # Campaign models
    "Campaign",
    "CampaignResult",
    "CampaignStatus",
    "CampaignTelemetryEvent",
    # Core database infrastructure
    "DatabaseConnection",
    "DatabaseSchema",
    "ExecutionStatus",
    "UnitOfWork",
    "create_unit_of_work",
    "get_database",
    "get_schema_manager",
    "initialize_database",
    "initialize_schema",
    "shutdown_database",
]
