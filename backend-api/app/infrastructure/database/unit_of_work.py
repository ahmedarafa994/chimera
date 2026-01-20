"""Unit of Work Pattern Implementation.

Story 1.3: Configuration Persistence System
Provides transaction management across multiple repositories.
"""

import logging
from contextlib import asynccontextmanager

from app.infrastructure.database.connection import DatabaseConnection
from app.infrastructure.repositories.api_key_repository import ApiKeyRepository
from app.infrastructure.repositories.config_repository import ConfigRepository

logger = logging.getLogger(__name__)


class UnitOfWorkError(Exception):
    """Raised when unit of work operations fail."""


class UnitOfWork:
    """Unit of Work pattern for managing database transactions.

    Provides a single transaction boundary for multiple repository operations.
    Ensures atomicity of related database changes.
    """

    def __init__(self, db: DatabaseConnection) -> None:
        """Initialize unit of work with database connection.

        Args:
            db: Database connection instance.

        """
        self._db = db
        self._config_repo: ConfigRepository | None = None
        self._api_key_repo: ApiKeyRepository | None = None
        self._in_transaction = False

    @property
    def configs(self) -> ConfigRepository:
        """Get configuration repository."""
        if self._config_repo is None:
            self._config_repo = ConfigRepository(self._db)
        return self._config_repo

    @property
    def api_keys(self) -> ApiKeyRepository:
        """Get API key repository."""
        if self._api_key_repo is None:
            self._api_key_repo = ApiKeyRepository(self._db)
        return self._api_key_repo

    async def __aenter__(self) -> "UnitOfWork":
        """Enter transaction context."""
        await self.begin()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit transaction context with commit or rollback."""
        if exc_type is not None:
            await self.rollback()
            return False
        await self.commit()
        return True

    async def begin(self) -> None:
        """Begin a new transaction."""
        if self._in_transaction:
            msg = "Transaction already in progress"
            raise UnitOfWorkError(msg)

        async with self._db.get_connection() as conn:
            await conn.execute("BEGIN")
        self._in_transaction = True
        logger.debug("Transaction started")

    async def commit(self) -> None:
        """Commit the current transaction."""
        if not self._in_transaction:
            return

        async with self._db.get_connection() as conn:
            await conn.execute("COMMIT")
        self._in_transaction = False
        logger.debug("Transaction committed")

    async def rollback(self) -> None:
        """Rollback the current transaction."""
        if not self._in_transaction:
            return

        try:
            async with self._db.get_connection() as conn:
                await conn.execute("ROLLBACK")
            logger.debug("Transaction rolled back")
        except Exception as e:
            logger.warning(f"Rollback failed: {e}")
        finally:
            self._in_transaction = False


@asynccontextmanager
async def create_unit_of_work(db: DatabaseConnection):
    """Create a unit of work context manager.

    Args:
        db: Database connection instance.

    Yields:
        UnitOfWork instance for transactional operations.

    Example:
        async with create_unit_of_work(db) as uow:
            config = await uow.configs.create(config_entity)
            api_key = await uow.api_keys.create(key_entity)
            # Both operations commit together or rollback on error

    """
    uow = UnitOfWork(db)
    async with uow:
        yield uow
