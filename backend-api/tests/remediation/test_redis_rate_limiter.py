"""Unit tests for RedisRateLimiter."""

import time

import pytest

from app.infrastructure.redis_rate_limiter import RedisRateLimiter


@pytest.mark.skipif(
    not pytest.importorskip("redis", reason="Redis not available"),
    reason="Redis connection required",
)
class TestRedisRateLimiter:
    """Test suite for RedisRateLimiter - using local fallback mode."""

    def test_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RedisRateLimiter()
        assert limiter is not None

    def test_initialization_with_custom_values(self):
        """Test rate limiter initializes with custom settings."""
        limiter = RedisRateLimiter(default_limit=100, default_window=120)
        assert limiter is not None

    @pytest.mark.asyncio
    async def test_is_allowed_under_limit(self):
        """Test request is allowed when under limit."""
        limiter = RedisRateLimiter(default_limit=10, default_window=60)

        allowed = await limiter.is_allowed("user_123")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_is_allowed_at_limit(self):
        """Test request is rejected at limit."""
        limiter = RedisRateLimiter(default_limit=3, default_window=60)

        # Make requests up to limit
        for _ in range(3):
            allowed = await limiter.is_allowed("user_123")
            assert allowed is True

        # Next request should be rejected
        allowed = await limiter.is_allowed("user_123")
        assert allowed is False

    @pytest.mark.asyncio
    async def test_get_remaining_requests(self):
        """Test getting remaining requests count."""
        limiter = RedisRateLimiter(default_limit=5, default_window=60)

        # Initially should have all requests available
        remaining = await limiter.get_remaining_requests("user_123")
        assert remaining == 5

        # Use some requests
        await limiter.is_allowed("user_123")
        await limiter.is_allowed("user_123")

        remaining = await limiter.get_remaining_requests("user_123")
        assert remaining == 3

    @pytest.mark.asyncio
    async def test_get_reset_time(self):
        """Test getting reset time."""
        limiter = RedisRateLimiter(default_limit=5, default_window=60)

        # Make a request to create window
        await limiter.is_allowed("user_123")

        reset_time = await limiter.get_reset_time("user_123")
        # Should be approximately 60 seconds from now
        assert reset_time > time.time()
        assert reset_time <= time.time() + 60

    @pytest.mark.asyncio
    async def test_get_usage(self):
        """Test getting comprehensive usage info."""
        limiter = RedisRateLimiter(default_limit=10, default_window=60)

        # Make some requests
        for _ in range(3):
            await limiter.is_allowed("user_123")

        usage = await limiter.get_usage("user_123")

        assert usage["used"] == 3
        assert usage["limit"] == 10
        assert usage["remaining"] == 7
        assert "reset_at" in usage

    @pytest.mark.asyncio
    async def test_different_identifiers_isolated(self):
        """Test that different identifiers are tracked separately."""
        limiter = RedisRateLimiter(default_limit=2, default_window=60)

        # User 1 uses up limit
        await limiter.is_allowed("user_1")
        await limiter.is_allowed("user_1")
        allowed = await limiter.is_allowed("user_1")
        assert allowed is False

        # User 2 should still have requests available
        allowed = await limiter.is_allowed("user_2")
        assert allowed is True

    @pytest.mark.asyncio
    async def test_custom_limit_override(self):
        """Test custom limit overrides default."""
        limiter = RedisRateLimiter(default_limit=10, default_window=60)

        # Use custom limit of 2
        await limiter.is_allowed("user_123", limit=2)
        await limiter.is_allowed("user_123", limit=2)
        allowed = await limiter.is_allowed("user_123", limit=2)

        assert allowed is False

    @pytest.mark.asyncio
    async def test_sliding_window_cleanup(self):
        """Test that old requests are cleaned up in sliding window."""
        limiter = RedisRateLimiter(default_limit=5, default_window=1)  # 1 second window

        # Use up limit
        for _ in range(5):
            await limiter.is_allowed("user_123")

        # Should be limited
        allowed = await limiter.is_allowed("user_123")
        assert allowed is False

        # Wait for window to pass
        import asyncio

        await asyncio.sleep(1.1)

        # Should be allowed again
        allowed = await limiter.is_allowed("user_123")
        assert allowed is True
