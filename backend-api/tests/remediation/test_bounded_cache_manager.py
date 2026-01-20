"""Unit tests for BoundedCacheManager."""

import pytest

from app.infrastructure.bounded_cache_manager import BoundedCacheManager


class TestBoundedCacheManager:
    """Test suite for BoundedCacheManager."""

    def test_initialization(self) -> None:
        """Test manager initializes correctly."""
        manager = BoundedCacheManager()
        assert manager is not None
        assert manager._max_value_size == 1_000_000

    def test_initialization_with_custom_values(self) -> None:
        """Test manager initializes with custom max_value_size."""
        manager = BoundedCacheManager(max_value_size=500)
        assert manager._max_value_size == 500

    def test_value_size_validation_valid(self) -> None:
        """Test size validation for valid values."""
        manager = BoundedCacheManager(max_value_size=1000)

        # Small string should pass
        small_value = {"key": "value"}
        assert manager._validate_value_size(small_value) is True

    def test_value_size_validation_too_large(self) -> None:
        """Test size validation rejects oversized values."""
        manager = BoundedCacheManager(max_value_size=10)

        # Large dict should fail
        large_dict = {"key": "x" * 200}
        assert manager._validate_value_size(large_dict) is False

    @pytest.mark.asyncio
    async def test_get_returns_none_for_missing_key(self) -> None:
        """Test get returns None for non-existent keys."""
        manager = BoundedCacheManager()
        await manager.initialize()
        result = await manager.get("nonexistent_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get_roundtrip(self) -> None:
        """Test setting and getting a value."""
        manager = BoundedCacheManager()
        await manager.initialize()

        await manager.set("test_key", {"data": "test_value"})
        result = await manager.get("test_key")

        assert result == {"data": "test_value"}

    @pytest.mark.asyncio
    async def test_set_rejects_oversized_values(self) -> None:
        """Test that oversized values are rejected."""
        manager = BoundedCacheManager(max_value_size=10)
        await manager.initialize()

        large_value = "x" * 100
        result = await manager.set("test_key", large_value)

        # Should return False indicating rejection
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_removes_value(self) -> None:
        """Test delete removes cached value."""
        manager = BoundedCacheManager()
        await manager.initialize()

        await manager.set("test_key", "test_value")
        await manager.delete("test_key")

        result = await manager.get("test_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_execution_result_cache_helpers(self) -> None:
        """Test execution result caching helpers."""
        manager = BoundedCacheManager()
        await manager.initialize()

        result = {"success": True, "output": "test"}
        await manager.cache_execution_result("exec_123", result)

        cached = await manager.get_execution_result("exec_123")
        assert cached == result

    @pytest.mark.asyncio
    async def test_technique_cache_helpers(self) -> None:
        """Test technique caching helpers."""
        manager = BoundedCacheManager()
        await manager.initialize()

        technique = {"id": "tech_1", "name": "Test Technique"}
        await manager.cache_technique("tech_1", technique)

        cached = await manager.get_technique("tech_1")
        assert cached == technique
