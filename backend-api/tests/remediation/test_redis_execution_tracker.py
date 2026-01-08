"""Unit tests for RedisExecutionTracker."""

from collections import OrderedDict

import pytest

from app.infrastructure.redis_execution_tracker import RedisExecutionTracker


class TestRedisExecutionTracker:
    """Test suite for RedisExecutionTracker - using local fallback mode."""

    def test_initialization(self):
        """Test tracker initializes correctly."""
        tracker = RedisExecutionTracker()
        assert tracker.active_ttl == 3600
        assert tracker.completed_ttl == 86400
        assert tracker.local_cache_max_size == 1000

    def test_initialization_with_custom_values(self):
        """Test tracker initializes with custom TTLs."""
        tracker = RedisExecutionTracker(active_ttl=1800, completed_ttl=43200, local_cache_size=5000)
        assert tracker.active_ttl == 1800
        assert tracker.completed_ttl == 43200
        assert tracker.local_cache_max_size == 5000

    def test_local_cache_is_bounded_ordereddict(self):
        """Test that local caches use OrderedDict for LRU."""
        tracker = RedisExecutionTracker()
        assert isinstance(tracker._local_active_cache, OrderedDict)
        assert isinstance(tracker._local_completed_cache, OrderedDict)

    @pytest.mark.asyncio
    async def test_start_execution(self):
        """Test starting execution tracking."""
        tracker = RedisExecutionTracker()

        result = await tracker.start_execution(
            execution_id="exec_123", technique_id="tech_1", user_id="user_1"
        )

        assert result is True
        assert "exec_123" in tracker._local_active_cache
        data = tracker._local_active_cache["exec_123"]
        assert data["technique_id"] == "tech_1"
        assert data["status"] == "running"

    @pytest.mark.asyncio
    async def test_start_execution_prevents_duplicates(self):
        """Test that duplicate execution IDs are rejected."""
        tracker = RedisExecutionTracker()

        # First start should succeed
        result1 = await tracker.start_execution("exec_123", "tech_1")
        assert result1 is True

        # Second start with same ID should fail
        result2 = await tracker.start_execution("exec_123", "tech_1")
        assert result2 is False

    @pytest.mark.asyncio
    async def test_complete_execution(self):
        """Test completing execution tracking."""
        tracker = RedisExecutionTracker()

        # Start execution first
        await tracker.start_execution("exec_123", "tech_1")

        # Complete it
        await tracker.complete_execution(
            execution_id="exec_123", success=True, execution_time_ms=1500.0
        )

        # Should move from active to completed
        assert "exec_123" not in tracker._local_active_cache
        assert "exec_123" in tracker._local_completed_cache

    @pytest.mark.asyncio
    async def test_get_active_count_total(self):
        """Test getting total active execution count."""
        tracker = RedisExecutionTracker()

        # Start multiple executions
        await tracker.start_execution("exec_1", "tech_1")
        await tracker.start_execution("exec_2", "tech_1")
        await tracker.start_execution("exec_3", "tech_2")

        # Total active
        total = await tracker.get_active_count()
        assert total == 3

    @pytest.mark.asyncio
    async def test_get_active_count_by_technique(self):
        """Test getting active execution count for specific technique."""
        tracker = RedisExecutionTracker()

        # Start multiple executions
        await tracker.start_execution("exec_1", "tech_1")
        await tracker.start_execution("exec_2", "tech_1")
        await tracker.start_execution("exec_3", "tech_2")

        # Active for specific technique
        tech1_count = await tracker.get_active_count("tech_1")
        assert tech1_count == 2

        tech2_count = await tracker.get_active_count("tech_2")
        assert tech2_count == 1

    @pytest.mark.asyncio
    async def test_local_cache_lru_eviction(self):
        """Test LRU eviction when local cache exceeds limit."""
        tracker = RedisExecutionTracker(local_cache_size=3)

        # Add executions up to limit
        await tracker.start_execution("exec_1", "tech_1")
        await tracker.start_execution("exec_2", "tech_1")
        await tracker.start_execution("exec_3", "tech_1")

        # Add one more - should trigger eviction
        await tracker.start_execution("exec_4", "tech_1")

        # Should have evicted oldest
        assert len(tracker._local_active_cache) == 3
        assert "exec_1" not in tracker._local_active_cache
        assert "exec_4" in tracker._local_active_cache

    def test_key_generation(self):
        """Test Redis key generation helpers."""
        tracker = RedisExecutionTracker()

        active_key = tracker._active_key("exec_123")
        assert "exec_123" in active_key
        assert "active" in active_key

        completed_key = tracker._completed_key("exec_123")
        assert "exec_123" in completed_key
        assert "completed" in completed_key
