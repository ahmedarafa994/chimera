"""
Base Repository Interface

Story 1.3: Configuration Persistence System
Provides abstract base class for all repositories.
"""

import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from app.infrastructure.database.connection import DatabaseConnection

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RepositoryError(Exception):
    """Base exception for repository operations."""


class EntityNotFoundError(RepositoryError):
    """Raised when entity is not found."""


class DuplicateEntityError(RepositoryError):
    """Raised when entity already exists."""


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository with common CRUD operations.

    Type Parameters:
        T: Entity type managed by this repository.
    """

    def __init__(self, db: DatabaseConnection):
        """
        Initialize repository with database connection.

        Args:
            db: Database connection instance.
        """
        self._db = db

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Get table name for this repository."""

    @abstractmethod
    async def get_by_id(self, entity_id: str) -> T | None:
        """
        Get entity by ID.

        Args:
            entity_id: Unique entity identifier.

        Returns:
            Entity if found, None otherwise.
        """

    @abstractmethod
    async def get_all(self) -> list[T]:
        """
        Get all entities.

        Returns:
            List of all entities.
        """

    @abstractmethod
    async def create(self, entity: T) -> T:
        """
        Create new entity.

        Args:
            entity: Entity to create.

        Returns:
            Created entity with generated ID.

        Raises:
            DuplicateEntityError: If entity already exists.
        """

    @abstractmethod
    async def update(self, entity: T) -> T:
        """
        Update existing entity.

        Args:
            entity: Entity with updated values.

        Returns:
            Updated entity.

        Raises:
            EntityNotFoundError: If entity doesn't exist.
        """

    @abstractmethod
    async def delete(self, entity_id: str) -> bool:
        """
        Delete entity by ID.

        Args:
            entity_id: Unique entity identifier.

        Returns:
            True if deleted, False if not found.
        """

    async def exists(self, entity_id: str) -> bool:
        """
        Check if entity exists.

        Args:
            entity_id: Unique entity identifier.

        Returns:
            True if entity exists.
        """
        result = await self._db.fetchone(
            f"SELECT 1 FROM {self.table_name} WHERE id = ?",
            (entity_id,)
        )
        return result is not None

    async def count(self) -> int:
        """
        Count total entities.

        Returns:
            Total count of entities.
        """
        result = await self._db.fetchone(
            f"SELECT COUNT(*) as count FROM {self.table_name}"
        )
        return result["count"] if result else 0
