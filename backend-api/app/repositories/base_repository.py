"""
Base Repository Pattern

This module provides a generic repository pattern for data access operations.
It abstracts database operations and provides a consistent interface for
CRUD operations across all entities.
"""

from typing import Any, Generic, TypeVar

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase

T = TypeVar("T", bound=DeclarativeBase)


class BaseRepository(Generic[T]):
    """Base repository for data access operations."""

    def __init__(self, model: type[T], session: AsyncSession):
        self.model = model
        self.session = session

    async def create(self, **kwargs) -> T:
        """Create a new entity."""
        instance = self.model(**kwargs)
        self.session.add(instance)
        await self.session.flush()
        return instance

    async def get_by_id(self, id: Any) -> T | None:
        """Get entity by ID."""
        result = await self.session.execute(select(self.model).where(self.model.id == id))
        return result.scalar_one_or_none()

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[T]:
        """Get all entities with pagination."""
        result = await self.session.execute(select(self.model).limit(limit).offset(offset))
        return list(result.scalars().all())

    async def update(self, id: Any, **kwargs) -> T | None:
        """Update entity by ID."""
        await self.session.execute(update(self.model).where(self.model.id == id).values(**kwargs))
        return await self.get_by_id(id)

    async def delete(self, id: Any) -> bool:
        """Delete entity by ID."""
        result = await self.session.execute(delete(self.model).where(self.model.id == id))
        return result.rowcount > 0

    async def exists(self, id: Any) -> bool:
        """Check if entity exists."""
        result = await self.session.execute(select(self.model.id).where(self.model.id == id))
        return result.scalar_one_or_none() is not None

    async def count(self) -> int:
        """Count total entities."""
        from sqlalchemy import func

        result = await self.session.execute(select(func.count()).select_from(self.model))
        return result.scalar_one()
