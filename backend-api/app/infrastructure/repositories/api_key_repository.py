"""API Key Repository with Encryption.

Story 1.3: Configuration Persistence System
Handles encrypted API key storage and retrieval.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.core.encryption import decrypt_api_key, encrypt_api_key, is_encrypted
from app.infrastructure.repositories.base import (
    BaseRepository,
    DuplicateEntityError,
    EntityNotFoundError,
)

logger = logging.getLogger(__name__)

TABLE_NAME = "api_keys"


@dataclass
class ApiKeyEntity:
    """API key entity for persistence with encryption support."""

    provider_type: str
    key_name: str
    api_key: str  # Plaintext key (encrypted during storage)
    is_active: bool = True
    id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_used_at: datetime | None = None

    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        now = datetime.now(UTC)
        if self.created_at is None:
            self.created_at = now
        if self.updated_at is None:
            self.updated_at = now

    def to_dict(self, encrypt: bool = True) -> dict[str, Any]:
        """Convert entity to dictionary for storage.

        Args:
            encrypt: Whether to encrypt the API key.

        Returns:
            Dictionary representation with encrypted key.

        """
        encrypted_key = self.api_key
        if encrypt and not is_encrypted(self.api_key):
            encrypted_key = encrypt_api_key(self.api_key)

        return {
            "id": self.id,
            "provider_type": self.provider_type,
            "key_name": self.key_name,
            "encrypted_key": encrypted_key,
            "is_active": 1 if self.is_active else 0,
            "created_at": (self.created_at.isoformat() if self.created_at else None),
            "updated_at": (self.updated_at.isoformat() if self.updated_at else None),
            "last_used_at": (self.last_used_at.isoformat() if self.last_used_at else None),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], decrypt: bool = True) -> "ApiKeyEntity":
        """Create entity from database row.

        Args:
            data: Database row dictionary.
            decrypt: Whether to decrypt the API key.

        Returns:
            ApiKeyEntity instance.

        """
        encrypted_key = data.get("encrypted_key", "")
        api_key = encrypted_key
        if decrypt and is_encrypted(encrypted_key):
            api_key = decrypt_api_key(encrypted_key)

        def parse_datetime(val):
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return val

        return cls(
            id=data["id"],
            provider_type=data["provider_type"],
            key_name=data["key_name"],
            api_key=api_key,
            is_active=bool(data.get("is_active", 1)),
            created_at=parse_datetime(data.get("created_at")),
            updated_at=parse_datetime(data.get("updated_at")),
            last_used_at=parse_datetime(data.get("last_used_at")),
        )


class ApiKeyRepository(BaseRepository[ApiKeyEntity]):
    """Repository for encrypted API key persistence.

    API keys are encrypted at rest using Fernet encryption.
    Keys are decrypted only when retrieved for runtime use.
    """

    @property
    def table_name(self) -> str:
        return TABLE_NAME

    async def get_by_id(self, entity_id: str) -> ApiKeyEntity | None:
        """Get API key by ID (decrypted)."""
        row = await self._db.fetchone("SELECT * FROM api_keys WHERE id = ?", (entity_id,))
        return ApiKeyEntity.from_dict(row) if row else None

    async def get_all(self) -> list[ApiKeyEntity]:
        """Get all API keys (decrypted)."""
        rows = await self._db.fetchall("SELECT * FROM api_keys")
        return [ApiKeyEntity.from_dict(row) for row in rows]

    async def create(self, entity: ApiKeyEntity) -> ApiKeyEntity:
        """Create new API key (encrypts before storage)."""
        if await self.exists(entity.id):
            msg = f"API key {entity.id} already exists"
            raise DuplicateEntityError(msg)

        # Check for duplicate provider/name combination
        existing = await self.get_by_provider_and_name(entity.provider_type, entity.key_name)
        if existing:
            msg = f"API key '{entity.key_name}' already exists for {entity.provider_type}"
            raise DuplicateEntityError(
                msg,
            )

        data = entity.to_dict(encrypt=True)
        await self._db.execute(
            """
            INSERT INTO api_keys
            (id, provider_type, key_name, encrypted_key, is_active,
             created_at, updated_at, last_used_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                data["id"],
                data["provider_type"],
                data["key_name"],
                data["encrypted_key"],
                data["is_active"],
                data["created_at"],
                data["updated_at"],
                data["last_used_at"],
            ),
        )

        logger.info(f"Created API key: {entity.key_name} for {entity.provider_type}")
        return entity

    async def update(self, entity: ApiKeyEntity) -> ApiKeyEntity:
        """Update existing API key (re-encrypts if needed)."""
        if not await self.exists(entity.id):
            msg = f"API key {entity.id} not found"
            raise EntityNotFoundError(msg)

        entity.updated_at = datetime.now(UTC)
        data = entity.to_dict(encrypt=True)

        await self._db.execute(
            """
            UPDATE api_keys
            SET provider_type = ?, key_name = ?, encrypted_key = ?,
                is_active = ?, updated_at = ?, last_used_at = ?
            WHERE id = ?
            """,
            (
                data["provider_type"],
                data["key_name"],
                data["encrypted_key"],
                data["is_active"],
                data["updated_at"],
                data["last_used_at"],
                data["id"],
            ),
        )

        logger.info(f"Updated API key: {entity.key_name}")
        return entity

    async def delete(self, entity_id: str) -> bool:
        """Delete API key by ID."""
        if not await self.exists(entity_id):
            return False

        await self._db.execute("DELETE FROM api_keys WHERE id = ?", (entity_id,))
        logger.info(f"Deleted API key: {entity_id}")
        return True

    async def get_by_provider_type(
        self,
        provider_type: str,
        active_only: bool = True,
    ) -> list[ApiKeyEntity]:
        """Get all API keys for a provider type.

        Args:
            provider_type: Provider type to filter by.
            active_only: Whether to return only active keys.

        Returns:
            List of API keys for the provider.

        """
        if active_only:
            rows = await self._db.fetchall(
                """
                SELECT * FROM api_keys
                WHERE provider_type = ? AND is_active = 1
                """,
                (provider_type,),
            )
        else:
            rows = await self._db.fetchall(
                "SELECT * FROM api_keys WHERE provider_type = ?",
                (provider_type,),
            )
        return [ApiKeyEntity.from_dict(row) for row in rows]

    async def get_by_provider_and_name(
        self,
        provider_type: str,
        key_name: str,
    ) -> ApiKeyEntity | None:
        """Get API key by provider type and name.

        Args:
            provider_type: Provider type.
            key_name: Key name.

        Returns:
            API key entity or None.

        """
        row = await self._db.fetchone(
            """
            SELECT * FROM api_keys
            WHERE provider_type = ? AND key_name = ?
            """,
            (provider_type, key_name),
        )
        return ApiKeyEntity.from_dict(row) if row else None

    async def get_active_key(self, provider_type: str) -> ApiKeyEntity | None:
        """Get the first active API key for a provider.

        Args:
            provider_type: Provider type.

        Returns:
            Active API key or None.

        """
        row = await self._db.fetchone(
            """
            SELECT * FROM api_keys
            WHERE provider_type = ? AND is_active = 1
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (provider_type,),
        )
        return ApiKeyEntity.from_dict(row) if row else None

    async def deactivate(self, entity_id: str) -> bool:
        """Deactivate an API key.

        Args:
            entity_id: Key ID to deactivate.

        Returns:
            True if deactivated successfully.

        """
        entity = await self.get_by_id(entity_id)
        if not entity:
            return False

        entity.is_active = False
        await self.update(entity)
        logger.info(f"Deactivated API key: {entity_id}")
        return True

    async def record_usage(self, entity_id: str) -> bool:
        """Record API key usage timestamp.

        Args:
            entity_id: Key ID to update.

        Returns:
            True if updated successfully.

        """
        if not await self.exists(entity_id):
            return False

        now = datetime.now(UTC).isoformat()
        await self._db.execute(
            f"UPDATE {self.table_name} SET last_used_at = ? WHERE id = ?",
            (now, entity_id),
        )
        return True
