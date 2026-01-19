"""
Async SQLite Database Connection Manager

Story 1.3: Configuration Persistence System
Provides connection pooling and lifecycle management for aiosqlite.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    """Base exception for database operations."""


class ConnectionError(DatabaseError):
    """Raised when database connection fails."""


class DatabaseConnection:
    """
    Async SQLite connection manager with pooling and lifecycle support.

    Uses aiosqlite for non-blocking database operations.
    Implements singleton pattern for application-wide connection sharing.
    """

    _instance: Optional["DatabaseConnection"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self, db_path: str | None = None):
        """
        Initialize database connection manager.

        Args:
            db_path: Path to SQLite database file. Defaults to config setting.
        """
        self._db_path = db_path or self._get_default_db_path()
        self._connection: aiosqlite.Connection | None = None
        self._initialized = False

    @staticmethod
    def _get_default_db_path() -> str:
        """Get default database path from environment or use default."""
        default_path = os.getenv(
            "CHIMERA_DB_PATH",
            str(Path(__file__).parent.parent.parent.parent / "data" / "chimera.db"),
        )
        return default_path

    @classmethod
    async def get_instance(cls, db_path: str | None = None) -> "DatabaseConnection":
        """
        Get singleton instance of database connection manager.

        Args:
            db_path: Optional path to override default database location.

        Returns:
            Singleton DatabaseConnection instance.
        """
        async with cls._lock:
            if cls._instance is None:
                cls._instance = cls(db_path)
            return cls._instance

    @classmethod
    async def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        async with cls._lock:
            if cls._instance is not None:
                await cls._instance.close()
                cls._instance = None

    async def initialize(self) -> None:
        """
        Initialize database connection and ensure schema exists.

        Creates database file and parent directories if needed.
        """
        if self._initialized:
            return

        try:
            db_dir = Path(self._db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            self._connection = await aiosqlite.connect(
                self._db_path, isolation_level=None  # Autocommit mode, we manage transactions
            )

            # Enable WAL mode for better concurrency
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA foreign_keys=ON")
            await self._connection.execute("PRAGMA busy_timeout=5000")

            # Return rows as dict-like objects
            self._connection.row_factory = aiosqlite.Row

            self._initialized = True
            logger.info(f"Database initialized at {self._db_path}")

        except Exception as e:
            raise ConnectionError(f"Failed to initialize database: {e}") from e

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            try:
                await self._connection.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.warning(f"Error closing database: {e}")
            finally:
                self._connection = None
                self._initialized = False

    @asynccontextmanager
    async def get_connection(self):
        """
        Get database connection context manager.

        Yields:
            aiosqlite.Connection for database operations.

        Raises:
            ConnectionError: If database not initialized.
        """
        if not self._initialized or self._connection is None:
            await self.initialize()

        try:
            yield self._connection
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            raise

    @asynccontextmanager
    async def transaction(self):
        """
        Execute operations within a transaction.

        Yields:
            aiosqlite.Connection for transactional operations.

        Raises:
            DatabaseError: If transaction fails.
        """
        async with self.get_connection() as conn:
            try:
                await conn.execute("BEGIN")
                yield conn
                await conn.execute("COMMIT")
            except Exception as e:
                await conn.execute("ROLLBACK")
                logger.error(f"Transaction rolled back: {e}")
                raise DatabaseError(f"Transaction failed: {e}") from e

    async def execute(self, sql: str, parameters: tuple = ()) -> aiosqlite.Cursor:
        """
        Execute SQL statement.

        Args:
            sql: SQL statement to execute.
            parameters: Query parameters.

        Returns:
            Cursor with execution results.
        """
        async with self.get_connection() as conn:
            return await conn.execute(sql, parameters)

    async def executemany(self, sql: str, parameters: list[tuple]) -> aiosqlite.Cursor:
        """
        Execute SQL statement with multiple parameter sets.

        Args:
            sql: SQL statement to execute.
            parameters: List of parameter tuples.

        Returns:
            Cursor with execution results.
        """
        async with self.get_connection() as conn:
            return await conn.executemany(sql, parameters)

    async def fetchone(self, sql: str, parameters: tuple = ()) -> dict | None:
        """
        Execute query and fetch single row.

        Args:
            sql: SQL query to execute.
            parameters: Query parameters.

        Returns:
            Single row as dict or None.
        """
        async with self.get_connection() as conn:
            cursor = await conn.execute(sql, parameters)
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def fetchall(self, sql: str, parameters: tuple = ()) -> list[dict]:
        """
        Execute query and fetch all rows.

        Args:
            sql: SQL query to execute.
            parameters: Query parameters.

        Returns:
            List of rows as dicts.
        """
        async with self.get_connection() as conn:
            cursor = await conn.execute(sql, parameters)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    @property
    def db_path(self) -> str:
        """Get database file path."""
        return self._db_path

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._initialized and self._connection is not None


# Module-level functions for FastAPI lifecycle management

_database: DatabaseConnection | None = None


async def get_database() -> DatabaseConnection:
    """
    Get database connection instance.

    Returns:
        DatabaseConnection singleton instance.
    """
    global _database
    if _database is None:
        _database = await DatabaseConnection.get_instance()
    if not _database.is_connected:
        await _database.initialize()
    return _database


async def initialize_database(db_path: str | None = None) -> DatabaseConnection:
    """
    Initialize database for application startup.

    Args:
        db_path: Optional custom database path.

    Returns:
        Initialized DatabaseConnection instance.
    """
    global _database
    _database = await DatabaseConnection.get_instance(db_path)
    await _database.initialize()
    logger.info("Database system initialized")
    return _database


async def shutdown_database() -> None:
    """Shutdown database connection on application exit."""
    global _database
    if _database:
        await _database.close()
        await DatabaseConnection.reset_instance()
        _database = None
        logger.info("Database system shutdown complete")
