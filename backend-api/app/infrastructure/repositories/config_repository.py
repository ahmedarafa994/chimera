"""
Provider Configuration Repository

Story 1.3: Configuration Persistence System
Handles CRUD operations for provider configurations.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.infrastructure.repositories.base import (
    BaseRepository,
    DuplicateEntityError,
    EntityNotFoundError,
)

logger = logging.getLogger(__name__)


@dataclass
class ProviderConfigEntity:
    """Provider configuration entity for persistence."""

    provider_type: str
    name: str
    settings: dict[str, Any] = field(default_factory=dict)
    is_default: bool = False
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        now = datetime.now(UTC)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    def to_dict(self) -> dict[str, Any]:
        """Convert entity to dictionary for storage."""
        return {
            "id": self.id,
            "provider_type": self.provider_type,
            "name": self.name,
            "is_default": 1 if self.is_default else 0,
            "settings": json.dumps(self.settings),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProviderConfigEntity":
        """Create entity from database row."""
        settings = data.get("settings", "{}")
        if isinstance(settings, str):
            settings = json.loads(settings)

        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        updated_at = data.get("updated_at")
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)

        return cls(
            id=data["id"],
            provider_type=data["provider_type"],
            name=data["name"],
            is_default=bool(data.get("is_default", 0)),
            settings=settings,
            created_at=created_at,
            updated_at=updated_at,
        )


class ConfigRepository(BaseRepository[ProviderConfigEntity]):
    """
    Repository for provider configuration persistence.

    Provides CRUD operations for provider configurations with
    support for default provider management.
    """

    @property
    def table_name(self) -> str:
        return "provider_configs"

    async def get_by_id(self, entity_id: str) -> ProviderConfigEntity | None:
        """Get configuration by ID."""
        row = await self._db.fetchone(
            f"SELECT * FROM {self.table_name} WHERE id = ?",
            (entity_id,)
        )
        return ProviderConfigEntity.from_dict(row) if row else None

    async def get_all(self) -> list[ProviderConfigEntity]:
        """Get all configurations."""
        rows = await self._db.fetchall(f"SELECT * FROM {self.table_name}")
        return [ProviderConfigEntity.from_dict(row) for row in rows]

    async def create(self, entity: ProviderConfigEntity) -> ProviderConfigEntity:
        """Create new configuration."""
        if await self.exists(entity.id):
            raise DuplicateEntityError(f"Config {entity.id} already exists")

        # Handle default provider logic
        if entity.is_default:
            await self._clear_default_for_type(entity.provider_type)

        data = entity.to_dict()
        await self._db.execute(
            f"""
            INSERT INTO {self.table_name}
            (id, provider_type, name, is_default, settings, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["provider_type"],
                data["name"],
                data["is_default"],
                data["settings"],
                data["created_at"],
                data["updated_at"],
            )
        )

        logger.info(f"Created provider config: {entity.name} ({entity.provider_type})")
        return entity

    async def update(self, entity: ProviderConfigEntity) -> ProviderConfigEntity:
        """Update existing configuration."""
        if not await self.exists(entity.id):
            raise EntityNotFoundError(f"Config {entity.id} not found")

        # Handle default provider logic
        if entity.is_default:
            await self._clear_default_for_type(entity.provider_type)

        entity.updated_at = datetime.now(UTC)
        data = entity.to_dict()

        await self._db.execute(
            f"""
            UPDATE {self.table_name}
            SET provider_type = ?, name = ?, is_default = ?,
                settings = ?, updated_at = ?
            WHERE id = ?
            """,
            (
                data["provider_type"],
                data["name"],
                data["is_default"],
                data["settings"],
                data["updated_at"],
                data["id"],
            )
        )

        logger.info(f"Updated provider config: {entity.name}")
        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete configuration by ID."""
        if not await self.exists(entity_id):
            return False

        await self._db.execute(
            f"DELETE FROM {self.table_name} WHERE id = ?",
            (entity_id,)
        )
        logger.info(f"Deleted provider config: {entity_id}")
        return True

    async def get_by_provider_type(
        self, provider_type: str
    ) -> list[ProviderConfigEntity]:
        """
        Get all configurations for a provider type.

        Args:
            provider_type: Provider type to filter by.

        Returns:
            List of configurations for the provider type.
        """
        rows = await self._db.fetchall(
            f"SELECT * FROM {self.table_name} WHERE provider_type = ?",
            (provider_type,)
        )
        return [ProviderConfigEntity.from_dict(row) for row in rows]

    async def get_default(
        self, provider_type: str | None = None
    ) -> ProviderConfigEntity | None:
        """
        Get default configuration.

        Args:
            provider_type: Optional provider type filter.

        Returns:
            Default configuration or None.
        """
        if provider_type:
            row = await self._db.fetchone(
                f"""
                SELECT * FROM {self.table_name}
                WHERE provider_type = ? AND is_default = 1
                """,
                (provider_type,)
            )
        else:
            row = await self._db.fetchone(
                f"SELECT * FROM {self.table_name} WHERE is_default = 1"
            )

        return ProviderConfigEntity.from_dict(row) if row else None

    async def set_default(self, entity_id: str) -> ProviderConfigEntity:
        """
        Set configuration as default for its provider type.

        Args:
            entity_id: Configuration ID to set as default.

        Returns:
            Updated configuration.

        Raises:
            EntityNotFoundError: If configuration not found.
        """
        entity = await self.get_by_id(entity_id)
        if not entity:
            raise EntityNotFoundError(f"Config {entity_id} not found")

        await self._clear_default_for_type(entity.provider_type)

        entity.is_default = True
        return await self.update(entity)

    async def get_by_name(
        self, name: str, provider_type: str | None = None
    ) -> ProviderConfigEntity | None:
        """
        Get configuration by name.

        Args:
            name: Configuration name.
            provider_type: Optional provider type filter.

        Returns:
            Configuration or None.
        """
        if provider_type:
            row = await self._db.fetchone(
                f"""
                SELECT * FROM {self.table_name}
                WHERE name = ? AND provider_type = ?
                """,
                (name, provider_type)
            )
        else:
            row = await self._db.fetchone(
                f"SELECT * FROM {self.table_name} WHERE name = ?",
                (name,)
            )

        return ProviderConfigEntity.from_dict(row) if row else None

    async def _clear_default_for_type(self, provider_type: str) -> None:
        """Clear default flag for all configs of a provider type."""
        await self._db.execute(
            f"""
            UPDATE {self.table_name}
            SET is_default = 0, updated_at = ?
            WHERE provider_type = ? AND is_default = 1
            """,
            (datetime.now(UTC).isoformat(), provider_type)
        )
