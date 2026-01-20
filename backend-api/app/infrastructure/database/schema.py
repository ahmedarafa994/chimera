"""Database Schema Management.

Story 1.3: Configuration Persistence System
Defines tables and handles schema migrations for configuration persistence.
"""

import logging
from datetime import UTC, datetime

from app.infrastructure.database.connection import DatabaseConnection

logger = logging.getLogger(__name__)

# Schema version for migrations
CURRENT_SCHEMA_VERSION = 1


class SchemaError(Exception):
    """Raised when schema operations fail."""


class DatabaseSchema:
    """Manages database schema creation and migrations.

    Provides idempotent schema setup with version tracking.
    """

    def __init__(self, db: DatabaseConnection) -> None:
        """Initialize schema manager.

        Args:
            db: Database connection instance.

        """
        self._db = db

    async def initialize(self) -> None:
        """Initialize database schema.

        Creates all tables if they don't exist and applies migrations.
        """
        try:
            await self._create_version_table()
            current_version = await self._get_schema_version()

            if current_version < CURRENT_SCHEMA_VERSION:
                await self._apply_migrations(current_version)

            logger.info(f"Schema initialized at version {CURRENT_SCHEMA_VERSION}")
        except Exception as e:
            msg = f"Schema initialization failed: {e}"
            raise SchemaError(msg) from e

    async def _create_version_table(self) -> None:
        """Create schema version tracking table."""
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                version INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            )
        """,
        )

        # Insert initial version if not exists
        result = await self._db.fetchone("SELECT version FROM schema_version WHERE id = 1")
        if result is None:
            await self._db.execute(
                "INSERT INTO schema_version (id, version, updated_at) VALUES (1, 0, ?)",
                (datetime.now(UTC).isoformat(),),
            )

    async def _get_schema_version(self) -> int:
        """Get current schema version."""
        result = await self._db.fetchone("SELECT version FROM schema_version WHERE id = 1")
        return result["version"] if result else 0

    async def _set_schema_version(self, version: int) -> None:
        """Update schema version."""
        await self._db.execute(
            "UPDATE schema_version SET version = ?, updated_at = ? WHERE id = 1",
            (version, datetime.now(UTC).isoformat()),
        )

    async def _apply_migrations(self, from_version: int) -> None:
        """Apply schema migrations.

        Args:
            from_version: Starting schema version.

        """
        migrations = [
            self._migration_v1,
        ]

        for version, migration in enumerate(migrations, start=1):
            if version > from_version:
                logger.info(f"Applying migration to version {version}")
                await migration()
                await self._set_schema_version(version)

    async def _migration_v1(self) -> None:
        """Version 1: Initial schema with configs and API keys tables."""
        # Provider configurations table
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS provider_configs (
                id TEXT PRIMARY KEY,
                provider_type TEXT NOT NULL,
                name TEXT NOT NULL,
                is_default INTEGER NOT NULL DEFAULT 0,
                settings TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """,
        )

        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_provider_configs_type
            ON provider_configs(provider_type)
        """,
        )

        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_provider_configs_default
            ON provider_configs(is_default)
        """,
        )

        # API keys table with encryption flag
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS api_keys (
                id TEXT PRIMARY KEY,
                provider_type TEXT NOT NULL,
                key_name TEXT NOT NULL,
                encrypted_key TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_used_at TEXT,
                UNIQUE(provider_type, key_name)
            )
        """,
        )

        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_api_keys_provider
            ON api_keys(provider_type)
        """,
        )

        await self._db.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_api_keys_active
            ON api_keys(is_active)
        """,
        )

        # Configuration cache metadata table
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS config_cache_metadata (
                cache_key TEXT PRIMARY KEY,
                config_type TEXT NOT NULL,
                expires_at TEXT,
                created_at TEXT NOT NULL
            )
        """,
        )

        logger.info("Migration v1 applied: created core tables")

    async def drop_all_tables(self) -> None:
        """Drop all tables (for testing/reset).

        Warning: This is destructive and should only be used in testing.
        """
        tables = [
            "config_cache_metadata",
            "api_keys",
            "provider_configs",
            "schema_version",
        ]

        for table in tables:
            await self._db.execute(f"DROP TABLE IF EXISTS {table}")

        logger.warning("All database tables dropped")


# Module-level schema manager singleton
_schema_manager: DatabaseSchema | None = None


async def get_schema_manager(db: DatabaseConnection) -> DatabaseSchema:
    """Get or create schema manager instance.

    Args:
        db: Database connection instance.

    Returns:
        DatabaseSchema singleton instance.

    """
    global _schema_manager
    if _schema_manager is None:
        _schema_manager = DatabaseSchema(db)
    return _schema_manager


async def initialize_schema(db: DatabaseConnection) -> None:
    """Initialize database schema.

    Args:
        db: Database connection instance.

    """
    schema = await get_schema_manager(db)
    await schema.initialize()
