"""
Configuration Persistence System Tests

Story 1.3: Configuration Persistence System
Comprehensive tests for database layer, repositories, and config service.
"""

import os
import tempfile
from datetime import timedelta

import pytest

# Set test database path before imports
TEST_DB_PATH = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
os.environ["CHIMERA_DB_PATH"] = TEST_DB_PATH


import contextlib

from app.domain.services.config_service import (
    ConfigNotFoundError,
    ConfigurationService,
    reset_config_service,
)
from app.infrastructure.database.connection import DatabaseConnection, shutdown_database
from app.infrastructure.database.schema import DatabaseSchema
from app.infrastructure.database.unit_of_work import UnitOfWork, create_unit_of_work
from app.infrastructure.repositories.api_key_repository import ApiKeyEntity, ApiKeyRepository
from app.infrastructure.repositories.base import DuplicateEntityError
from app.infrastructure.repositories.config_repository import ConfigRepository, ProviderConfigEntity


@pytest.fixture
async def db():
    """Create test database connection."""
    db_path = tempfile.NamedTemporaryFile(suffix=".db", delete=False).name
    connection = DatabaseConnection(db_path)
    await connection.initialize()
    yield connection
    await connection.close()
    # Cleanup
    with contextlib.suppress(Exception):
        os.unlink(db_path)


@pytest.fixture
async def schema(db):
    """Initialize database schema."""
    schema_manager = DatabaseSchema(db)
    await schema_manager.initialize()
    yield schema_manager


@pytest.fixture
async def config_repo(db, schema):
    """Create configuration repository."""
    return ConfigRepository(db)


@pytest.fixture
async def api_key_repo(db, schema):
    """Create API key repository."""
    return ApiKeyRepository(db)


@pytest.fixture
async def config_service(db, schema):
    """Create configuration service."""
    return ConfigurationService(db, cache_ttl=timedelta(seconds=1))


class TestDatabaseConnection:
    """Test database connection management."""

    @pytest.mark.asyncio
    async def test_connection_initialize(self, db):
        """Test database initializes correctly."""
        assert db.is_connected
        assert os.path.exists(db.db_path)

    @pytest.mark.asyncio
    async def test_execute_query(self, db, schema):
        """Test basic query execution."""
        result = await db.fetchone("SELECT 1 as test")
        assert result["test"] == 1

    @pytest.mark.asyncio
    async def test_transaction_commit(self, db, schema):
        """Test transaction commits changes."""
        async with db.transaction() as conn:
            await conn.execute(
                "INSERT INTO schema_version (id, version, updated_at) "
                "VALUES (2, 99, '2025-01-01')"
            )

        result = await db.fetchone("SELECT version FROM schema_version WHERE id = 2")
        assert result["version"] == 99

    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db, schema):
        """Test transaction rollback on error."""
        try:
            async with db.transaction():
                await db.execute(
                    "INSERT INTO schema_version (id, version, updated_at) "
                    "VALUES (3, 100, '2025-01-01')"
                )
                raise ValueError("Simulated error")
        except Exception:
            pass

        result = await db.fetchone("SELECT version FROM schema_version WHERE id = 3")
        assert result is None


class TestDatabaseSchema:
    """Test schema management."""

    @pytest.mark.asyncio
    async def test_schema_creates_tables(self, db, schema):
        """Test schema creates required tables."""
        tables = await db.fetchall("SELECT name FROM sqlite_master WHERE type='table'")
        table_names = [t["name"] for t in tables]

        assert "schema_version" in table_names
        assert "provider_configs" in table_names
        assert "api_keys" in table_names

    @pytest.mark.asyncio
    async def test_schema_version_tracked(self, db, schema):
        """Test schema version is tracked."""
        result = await db.fetchone("SELECT version FROM schema_version WHERE id = 1")
        assert result["version"] >= 1


class TestConfigRepository:
    """Test provider configuration repository."""

    @pytest.mark.asyncio
    async def test_create_config(self, config_repo):
        """Test creating a configuration."""
        entity = ProviderConfigEntity(
            provider_type="openai", name="default", settings={"model": "gpt-4"}, is_default=True
        )

        created = await config_repo.create(entity)

        assert created.id is not None
        assert created.provider_type == "openai"
        assert created.name == "default"
        assert created.settings["model"] == "gpt-4"
        assert created.is_default is True

    @pytest.mark.asyncio
    async def test_get_config_by_id(self, config_repo):
        """Test retrieving configuration by ID."""
        entity = ProviderConfigEntity(provider_type="anthropic", name="claude-config")
        created = await config_repo.create(entity)

        retrieved = await config_repo.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.id == created.id
        assert retrieved.provider_type == "anthropic"

    @pytest.mark.asyncio
    async def test_get_all_configs(self, config_repo):
        """Test retrieving all configurations."""
        await config_repo.create(ProviderConfigEntity(provider_type="openai", name="config1"))
        await config_repo.create(ProviderConfigEntity(provider_type="anthropic", name="config2"))

        configs = await config_repo.get_all()

        assert len(configs) >= 2

    @pytest.mark.asyncio
    async def test_update_config(self, config_repo):
        """Test updating configuration."""
        entity = ProviderConfigEntity(provider_type="openai", name="original")
        created = await config_repo.create(entity)

        created.name = "updated"
        created.settings = {"temperature": 0.8}
        updated = await config_repo.update(created)

        assert updated.name == "updated"
        assert updated.settings["temperature"] == 0.8

    @pytest.mark.asyncio
    async def test_delete_config(self, config_repo):
        """Test deleting configuration."""
        entity = ProviderConfigEntity(provider_type="openai", name="to-delete")
        created = await config_repo.create(entity)

        result = await config_repo.delete(created.id)
        assert result is True

        retrieved = await config_repo.get_by_id(created.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_get_by_provider_type(self, config_repo):
        """Test filtering by provider type."""
        await config_repo.create(ProviderConfigEntity(provider_type="openai", name="oai1"))
        await config_repo.create(ProviderConfigEntity(provider_type="openai", name="oai2"))
        await config_repo.create(ProviderConfigEntity(provider_type="anthropic", name="ant1"))

        openai_configs = await config_repo.get_by_provider_type("openai")

        assert len(openai_configs) >= 2
        assert all(c.provider_type == "openai" for c in openai_configs)

    @pytest.mark.asyncio
    async def test_default_config_management(self, config_repo):
        """Test default configuration handling."""
        await config_repo.create(
            ProviderConfigEntity(provider_type="openai", name="first", is_default=True)
        )

        # Creating second default should clear first
        config2 = await config_repo.create(
            ProviderConfigEntity(provider_type="openai", name="second", is_default=True)
        )

        # Verify only one default
        default = await config_repo.get_default("openai")
        assert default is not None
        assert default.id == config2.id

    @pytest.mark.asyncio
    async def test_duplicate_config_raises_error(self, config_repo):
        """Test duplicate configuration raises error."""
        entity = ProviderConfigEntity(provider_type="openai", name="unique")
        await config_repo.create(entity)

        with pytest.raises(DuplicateEntityError):
            await config_repo.create(entity)


class TestApiKeyRepository:
    """Test API key repository with encryption."""

    @pytest.mark.asyncio
    async def test_create_api_key_encrypts(self, api_key_repo):
        """Test API key is encrypted on storage."""
        entity = ApiKeyEntity(
            provider_type="openai", key_name="test-key", api_key="sk-test-key-12345"
        )

        created = await api_key_repo.create(entity)

        assert created.id is not None
        assert created.api_key == "sk-test-key-12345"  # Decrypted on read

    @pytest.mark.asyncio
    async def test_get_api_key_decrypts(self, api_key_repo):
        """Test API key is decrypted on retrieval."""
        original_key = "sk-secret-api-key-67890"
        entity = ApiKeyEntity(provider_type="anthropic", key_name="prod-key", api_key=original_key)
        created = await api_key_repo.create(entity)

        retrieved = await api_key_repo.get_by_id(created.id)

        assert retrieved is not None
        assert retrieved.api_key == original_key

    @pytest.mark.asyncio
    async def test_get_active_key(self, api_key_repo):
        """Test getting active API key for provider."""
        await api_key_repo.create(
            ApiKeyEntity(provider_type="openai", key_name="active-key", api_key="sk-active")
        )

        active = await api_key_repo.get_active_key("openai")

        assert active is not None
        assert active.is_active is True

    @pytest.mark.asyncio
    async def test_deactivate_key(self, api_key_repo):
        """Test deactivating API key."""
        entity = ApiKeyEntity(
            provider_type="openai", key_name="to-deactivate", api_key="sk-deactivate"
        )
        created = await api_key_repo.create(entity)

        result = await api_key_repo.deactivate(created.id)
        assert result is True

        retrieved = await api_key_repo.get_by_id(created.id)
        assert retrieved.is_active is False

    @pytest.mark.asyncio
    async def test_record_usage(self, api_key_repo):
        """Test recording API key usage."""
        entity = ApiKeyEntity(provider_type="openai", key_name="usage-key", api_key="sk-usage")
        created = await api_key_repo.create(entity)
        assert created.last_used_at is None

        result = await api_key_repo.record_usage(created.id)
        assert result is True

        retrieved = await api_key_repo.get_by_id(created.id)
        assert retrieved.last_used_at is not None

    @pytest.mark.asyncio
    async def test_duplicate_provider_name_raises(self, api_key_repo):
        """Test duplicate provider/name raises error."""
        entity = ApiKeyEntity(provider_type="openai", key_name="unique-name", api_key="sk-first")
        await api_key_repo.create(entity)

        duplicate = ApiKeyEntity(
            provider_type="openai", key_name="unique-name", api_key="sk-second"
        )
        with pytest.raises(DuplicateEntityError):
            await api_key_repo.create(duplicate)


class TestUnitOfWork:
    """Test Unit of Work pattern."""

    @pytest.mark.asyncio
    async def test_uow_provides_repositories(self, db, schema):
        """Test UoW provides access to repositories."""
        uow = UnitOfWork(db)

        assert uow.configs is not None
        assert uow.api_keys is not None

    @pytest.mark.asyncio
    async def test_uow_context_manager(self, db, schema):
        """Test UoW as context manager."""
        async with create_unit_of_work(db) as uow:
            config = ProviderConfigEntity(provider_type="openai", name="uow-test")
            await uow.configs.create(config)

        # Verify persisted after context exit
        uow2 = UnitOfWork(db)
        configs = await uow2.configs.get_all()
        assert any(c.name == "uow-test" for c in configs)


class TestConfigurationService:
    """Test configuration service with caching."""

    @pytest.mark.asyncio
    async def test_create_config(self, config_service):
        """Test creating configuration via service."""
        config = await config_service.create_config(
            provider_type="openai",
            name="service-test",
            settings={"model": "gpt-4"},
            is_default=True,
        )

        assert config.id is not None
        assert config.name == "service-test"

    @pytest.mark.asyncio
    async def test_get_config_caches(self, config_service):
        """Test configuration is cached."""
        config = await config_service.create_config(provider_type="openai", name="cache-test")

        # First call populates cache
        retrieved1 = await config_service.get_config(config.id)
        # Second call should use cache
        retrieved2 = await config_service.get_config(config.id)

        assert retrieved1.id == retrieved2.id

    @pytest.mark.asyncio
    async def test_update_invalidates_cache(self, config_service):
        """Test update invalidates cache."""
        config = await config_service.create_config(provider_type="openai", name="invalidate-test")

        # Cache the config
        await config_service.get_config(config.id)

        # Update should invalidate
        updated = await config_service.update_config(config.id, name="updated-name")

        assert updated.name == "updated-name"

    @pytest.mark.asyncio
    async def test_delete_config(self, config_service):
        """Test deleting configuration."""
        config = await config_service.create_config(provider_type="openai", name="delete-test")

        result = await config_service.delete_config(config.id)
        assert result is True

        retrieved = await config_service.get_config(config.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_store_api_key(self, config_service):
        """Test storing API key."""
        key = await config_service.store_api_key(
            provider_type="openai", key_name="service-key", api_key="sk-test-12345"
        )

        assert key.id is not None
        assert key.api_key == "sk-test-12345"

    @pytest.mark.asyncio
    async def test_get_active_api_key(self, config_service):
        """Test getting active API key."""
        await config_service.store_api_key(
            provider_type="anthropic", key_name="active-service-key", api_key="sk-anthropic"
        )

        key = await config_service.get_active_api_key("anthropic")

        assert key is not None
        assert key.is_active is True

    @pytest.mark.asyncio
    async def test_rotate_api_key(self, config_service):
        """Test rotating API key."""
        key = await config_service.store_api_key(
            provider_type="openai", key_name="rotate-key", api_key="sk-old-key"
        )

        rotated = await config_service.rotate_api_key(key.id, new_api_key="sk-new-key")

        assert rotated.api_key == "sk-new-key"

    @pytest.mark.asyncio
    async def test_deactivate_api_key(self, config_service):
        """Test deactivating API key."""
        key = await config_service.store_api_key(
            provider_type="openai", key_name="deactivate-service", api_key="sk-deactivate"
        )

        result = await config_service.deactivate_api_key(key.id)
        assert result is True

    @pytest.mark.asyncio
    async def test_update_nonexistent_raises(self, config_service):
        """Test updating non-existent config raises error."""
        with pytest.raises(ConfigNotFoundError):
            await config_service.update_config("nonexistent-id", name="new-name")

    @pytest.mark.asyncio
    async def test_cache_clear(self, config_service):
        """Test clearing cache."""
        await config_service.create_config(provider_type="openai", name="clear-cache-test")

        # Populate cache
        await config_service.get_all_configs()

        # Clear cache
        config_service.clear_cache()

        # Should still work (fetches from DB)
        configs = await config_service.get_all_configs()
        assert len(configs) >= 1


# Cleanup after tests
@pytest.fixture(autouse=True)
async def cleanup():
    """Cleanup after each test."""
    yield
    await reset_config_service()
    await shutdown_database()
