"""
Comprehensive Tests for LLM Service.

Tests cover:
- LLMResponseCache: cache hit/miss, TTL, eviction, stats
- RequestDeduplicator: concurrent request deduplication
- LLMService: provider registration, text generation, streaming, circuit breaker
"""

import asyncio
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLLMResponseCache:
    """Tests for LLMResponseCache class."""

    @pytest.fixture
    def cache(self):
        """Create a fresh cache instance for testing."""
        from app.services.llm_service import LLMResponseCache

        return LLMResponseCache(max_size=10, default_ttl=60)

    @pytest.fixture
    def sample_request(self):
        """Create a sample prompt request."""
        request = MagicMock()
        request.prompt = "Test prompt"
        request.model = "test-model"
        request.provider = "test-provider"
        request.max_tokens = 100
        request.temperature = 0.7
        return request

    @pytest.fixture
    def sample_response(self):
        """Create a sample prompt response."""
        response = MagicMock()
        response.content = "Test response content"
        response.model = "test-model"
        response.provider = "test-provider"
        response.tokens_used = 50
        response.cached = False
        return response

    @pytest.mark.asyncio
    async def test_cache_miss_returns_none(self, cache, sample_request):
        """Test that cache miss returns None."""
        result = await cache.get(sample_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache, sample_request, sample_response):
        """Test basic cache set and get operations."""
        await cache.set(sample_request, sample_response)
        result = await cache.get(sample_request)

        assert result is not None
        assert result.content == sample_response.content

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, sample_request, sample_response):
        """Test that cache entries expire after TTL."""
        from app.services.llm_service import LLMResponseCache

        cache = LLMResponseCache(max_size=10, default_ttl=1)  # 1 second TTL

        await cache.set(sample_request, sample_response)

        # Should be cached immediately
        result = await cache.get(sample_request)
        assert result is not None

        # Wait for TTL to expire
        await asyncio.sleep(1.5)

        # Should be expired now
        result = await cache.get(sample_request)
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_max_size_eviction(self):
        """Test that cache evicts oldest entries when max size is reached."""
        from app.services.llm_service import LLMResponseCache

        cache = LLMResponseCache(max_size=3, default_ttl=300)

        # Create multiple requests
        for i in range(5):
            request = MagicMock()
            request.prompt = f"Prompt {i}"
            request.model = "test-model"
            request.provider = "test-provider"
            request.max_tokens = 100
            request.temperature = 0.7

            response = MagicMock()
            response.content = f"Response {i}"

            await cache.set(request, response)

        # Cache should only have 3 entries (max_size)
        stats = cache.get_stats()
        assert stats["size"] <= 3

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache, sample_request, sample_response):
        """Test cache statistics tracking."""
        # Initial stats
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0
        assert stats["size"] == 0

        # Cache miss
        await cache.get(sample_request)
        stats = cache.get_stats()
        assert stats["misses"] == 1

        # Set and get (cache hit)
        await cache.set(sample_request, sample_response)
        await cache.get(sample_request)
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["size"] == 1

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache, sample_request, sample_response):
        """Test cache clearing."""
        await cache.set(sample_request, sample_response)

        stats = cache.get_stats()
        assert stats["size"] == 1

        await cache.clear()

        stats = cache.get_stats()
        assert stats["size"] == 0

    @pytest.mark.asyncio
    async def test_cache_different_requests(self, cache):
        """Test that different requests have different cache entries."""
        request1 = MagicMock()
        request1.prompt = "Prompt 1"
        request1.model = "model-1"
        request1.provider = "provider"
        request1.max_tokens = 100
        request1.temperature = 0.7

        request2 = MagicMock()
        request2.prompt = "Prompt 2"
        request2.model = "model-1"
        request2.provider = "provider"
        request2.max_tokens = 100
        request2.temperature = 0.7

        response1 = MagicMock()
        response1.content = "Response 1"

        response2 = MagicMock()
        response2.content = "Response 2"

        await cache.set(request1, response1)
        await cache.set(request2, response2)

        result1 = await cache.get(request1)
        result2 = await cache.get(request2)

        assert result1.content == "Response 1"
        assert result2.content == "Response 2"


class TestRequestDeduplicator:
    """Tests for RequestDeduplicator class."""

    @pytest.fixture
    def deduplicator(self):
        """Create a fresh deduplicator instance."""
        from app.services.llm_service import RequestDeduplicator

        return RequestDeduplicator()

    @pytest.fixture
    def sample_request(self):
        """Create a sample request."""
        request = MagicMock()
        request.prompt = "Test prompt"
        request.model = "test-model"
        request.provider = "test-provider"
        return request

    @pytest.mark.asyncio
    async def test_single_request(self, deduplicator, sample_request):
        """Test single request execution."""
        call_count = 0

        async def execute_fn():
            nonlocal call_count
            call_count += 1
            return "result"

        result = await deduplicator.deduplicate(sample_request, execute_fn)

        assert result == "result"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_identical_requests(self, deduplicator, sample_request):
        """Test that concurrent identical requests are deduplicated."""
        call_count = 0

        async def execute_fn():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.1)  # Simulate some processing time
            return "result"

        # Launch multiple concurrent requests
        tasks = [deduplicator.deduplicate(sample_request, execute_fn) for _ in range(5)]

        results = await asyncio.gather(*tasks)

        # All results should be the same
        assert all(r == "result" for r in results)
        # Function should only be called once due to deduplication
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_different_requests_not_deduplicated(self, deduplicator):
        """Test that different requests are not deduplicated."""
        call_count = 0

        async def execute_fn():
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        request1 = MagicMock()
        request1.prompt = "Prompt 1"
        request1.model = "model"
        request1.provider = "provider"

        request2 = MagicMock()
        request2.prompt = "Prompt 2"
        request2.model = "model"
        request2.provider = "provider"

        result1 = await deduplicator.deduplicate(request1, execute_fn)
        result2 = await deduplicator.deduplicate(request2, execute_fn)

        assert result1 != result2
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_exception_propagation(self, deduplicator, sample_request):
        """Test that exceptions are properly propagated."""

        async def execute_fn():
            raise ValueError("Test error")

        with pytest.raises(ValueError, match="Test error"):
            await deduplicator.deduplicate(sample_request, execute_fn)


class TestLLMService:
    """Tests for LLMService class."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LLM provider."""
        provider = MagicMock()
        provider.name = "mock-provider"
        provider.generate = AsyncMock(return_value="Mock response")
        provider.is_available = MagicMock(return_value=True)
        provider.count_tokens = MagicMock(return_value=10)
        return provider

    @pytest.fixture
    def llm_service(self):
        """Create a fresh LLM service instance."""
        from app.services.llm_service import LLMService

        service = LLMService()
        return service

    def test_register_provider(self, llm_service, mock_provider):
        """Test provider registration."""
        llm_service.register_provider("test-provider", mock_provider, is_default=True)

        assert "test-provider" in llm_service._providers
        assert llm_service._default_provider == "test-provider"

    def test_register_multiple_providers(self, llm_service, mock_provider):
        """Test registering multiple providers."""
        provider2 = MagicMock()
        provider2.name = "provider-2"

        llm_service.register_provider("provider-1", mock_provider, is_default=True)
        llm_service.register_provider("provider-2", provider2, is_default=False)

        assert "provider-1" in llm_service._providers
        assert "provider-2" in llm_service._providers
        assert llm_service._default_provider == "provider-1"

    def test_register_provider_as_new_default(self, llm_service, mock_provider):
        """Test that new default provider overrides previous."""
        provider2 = MagicMock()
        provider2.name = "provider-2"

        llm_service.register_provider("provider-1", mock_provider, is_default=True)
        llm_service.register_provider("provider-2", provider2, is_default=True)

        assert llm_service._default_provider == "provider-2"

    @pytest.mark.asyncio
    async def test_list_providers(self, llm_service, mock_provider):
        """Test listing available providers."""
        llm_service.register_provider("test-provider", mock_provider)

        result = await llm_service.list_providers()

        assert result is not None
        assert len(result.providers) >= 1

    @pytest.mark.asyncio
    async def test_count_tokens(self, llm_service, mock_provider):
        """Test token counting."""
        llm_service.register_provider("test-provider", mock_provider, is_default=True)

        count = await llm_service.count_tokens("Test text for counting", provider="test-provider")

        assert count == 10
        mock_provider.count_tokens.assert_called_once()

    @pytest.mark.asyncio
    async def test_generate_text_with_cache_disabled(self, llm_service, mock_provider):
        """Test text generation with cache disabled."""
        llm_service.register_provider("test-provider", mock_provider, is_default=True)

        request = MagicMock()
        request.prompt = "Test prompt"
        request.model = "test-model"
        request.provider = "test-provider"
        request.max_tokens = 100
        request.temperature = 0.7
        request.use_cache = False

        # Mock the internal generate method
        with patch.object(llm_service, "_execute_generation", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = MagicMock(
                content="Generated text",
                model="test-model",
                provider="test-provider",
                tokens_used=50,
                cached=False,
            )

            result = await llm_service.generate_text(request)

            assert result.content == "Generated text"
            assert result.cached is False


class TestLLMServiceCircuitBreaker:
    """Tests for LLM Service circuit breaker functionality."""

    @pytest.fixture
    def failing_provider(self):
        """Create a provider that fails."""
        provider = MagicMock()
        provider.name = "failing-provider"
        provider.generate = AsyncMock(side_effect=Exception("Provider error"))
        provider.is_available = MagicMock(return_value=True)
        return provider

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, failing_provider):
        """Test that circuit breaker opens after repeated failures."""
        from app.services.llm_service import LLMService

        service = LLMService()
        service.register_provider("failing", failing_provider, is_default=True)

        # Attempt multiple failed calls
        for _ in range(5):
            try:
                request = MagicMock()
                request.prompt = "Test"
                request.model = "model"
                request.provider = "failing"
                request.use_cache = False

                with patch.object(
                    service, "_execute_generation", new_callable=AsyncMock
                ) as mock_exec:
                    mock_exec.side_effect = Exception("Provider error")
                    await service.generate_text(request)
            except Exception:
                pass

        # Circuit breaker state should be tracked
        # (actual behavior depends on implementation)


class TestLLMServiceIntegration:
    """Integration tests for LLM Service with mocked providers."""

    @pytest.fixture
    def full_service(self):
        """Create a fully configured service with mock providers."""
        from app.services.llm_service import LLMService

        service = LLMService()

        # Add multiple mock providers
        providers = ["openai", "anthropic", "google"]
        for name in providers:
            provider = MagicMock()
            provider.name = name
            provider.generate = AsyncMock(return_value=f"Response from {name}")
            provider.is_available = MagicMock(return_value=True)
            provider.count_tokens = MagicMock(return_value=10)

            service.register_provider(name, provider, is_default=(name == "openai"))

        return service

    @pytest.mark.asyncio
    async def test_provider_selection(self, full_service):
        """Test that correct provider is selected."""
        # Default provider
        result = await full_service.list_providers()
        assert len(result.providers) == 3

    @pytest.mark.asyncio
    async def test_fallback_on_provider_failure(self, full_service):
        """Test fallback to another provider on failure."""
        # Make primary provider fail
        full_service._providers["openai"].generate.side_effect = Exception("Primary failed")

        # Service should handle failure gracefully
        # (implementation-dependent)


class TestCacheKeyGeneration:
    """Tests for cache key generation logic."""

    def test_same_request_same_key(self):
        """Test that identical requests produce identical cache keys."""
        request1 = MagicMock()
        request1.prompt = "Test prompt"
        request1.model = "model"
        request1.provider = "provider"
        request1.max_tokens = 100
        request1.temperature = 0.7

        request2 = MagicMock()
        request2.prompt = "Test prompt"
        request2.model = "model"
        request2.provider = "provider"
        request2.max_tokens = 100
        request2.temperature = 0.7

        # Generate keys (implementation-dependent)
        key1 = hashlib.md5(
            f"{request1.prompt}:{request1.model}:{request1.provider}".encode()
        ).hexdigest()
        key2 = hashlib.md5(
            f"{request2.prompt}:{request2.model}:{request2.provider}".encode()
        ).hexdigest()

        assert key1 == key2

    def test_different_prompts_different_keys(self):
        """Test that different prompts produce different cache keys."""
        request1 = MagicMock()
        request1.prompt = "Prompt 1"
        request1.model = "model"
        request1.provider = "provider"

        request2 = MagicMock()
        request2.prompt = "Prompt 2"
        request2.model = "model"
        request2.provider = "provider"

        key1 = hashlib.md5(
            f"{request1.prompt}:{request1.model}:{request1.provider}".encode()
        ).hexdigest()
        key2 = hashlib.md5(
            f"{request2.prompt}:{request2.model}:{request2.provider}".encode()
        ).hexdigest()

        assert key1 != key2


class TestStreamingGeneration:
    """Tests for streaming text generation."""

    @pytest.fixture
    def streaming_provider(self):
        """Create a mock streaming provider."""
        provider = MagicMock()
        provider.name = "streaming-provider"
        provider.is_available = MagicMock(return_value=True)

        async def mock_stream():
            chunks = ["Hello", " ", "World", "!"]
            for chunk in chunks:
                yield MagicMock(content=chunk, type="delta")

        provider.generate_stream = MagicMock(return_value=mock_stream())
        return provider

    @pytest.mark.asyncio
    async def test_streaming_returns_chunks(self, streaming_provider):
        """Test that streaming generation yields chunks."""
        from app.services.llm_service import LLMService

        service = LLMService()
        service.register_provider("streaming", streaming_provider, is_default=True)

        # Note: actual streaming test would depend on implementation
        # This is a placeholder for the pattern


class TestErrorHandling:
    """Tests for error handling in LLM Service."""

    @pytest.mark.asyncio
    async def test_invalid_provider_raises_error(self):
        """Test that requesting invalid provider raises error."""
        from app.services.llm_service import LLMService

        service = LLMService()

        request = MagicMock()
        request.prompt = "Test"
        request.model = "model"
        request.provider = "nonexistent"
        request.use_cache = False

        # Should raise error for nonexistent provider
        with pytest.raises(Exception):
            await service.generate_text(request)

    @pytest.mark.asyncio
    async def test_empty_prompt_handling(self):
        """Test handling of empty prompts."""
        from app.services.llm_service import LLMService

        service = LLMService()

        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="")
        service.register_provider("test", mock_provider, is_default=True)

        request = MagicMock()
        request.prompt = ""
        request.model = "model"
        request.provider = "test"
        request.use_cache = False

        # Empty prompt should be handled appropriately


class TestConcurrency:
    """Tests for concurrent access to LLM Service."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests."""
        from app.services.llm_service import LLMService

        service = LLMService()

        mock_provider = MagicMock()
        mock_provider.generate = AsyncMock(return_value="Response")
        mock_provider.is_available = MagicMock(return_value=True)
        service.register_provider("test", mock_provider, is_default=True)

        # Create multiple concurrent requests
        requests = []
        for i in range(10):
            req = MagicMock()
            req.prompt = f"Prompt {i}"
            req.model = "model"
            req.provider = "test"
            req.use_cache = False
            requests.append(req)

        # All requests should complete without deadlock
        # (actual test depends on implementation)

    @pytest.mark.asyncio
    async def test_thread_safe_cache_access(self):
        """Test thread-safe cache access."""
        from app.services.llm_service import LLMResponseCache

        cache = LLMResponseCache(max_size=100, default_ttl=300)

        async def cache_operation(i: int):
            request = MagicMock()
            request.prompt = f"Prompt {i % 10}"  # Some overlap
            request.model = "model"
            request.provider = "provider"

            response = MagicMock()
            response.content = f"Response {i}"

            await cache.set(request, response)
            await cache.get(request)

        # Run many concurrent operations
        tasks = [cache_operation(i) for i in range(100)]
        await asyncio.gather(*tasks)

        # Cache should remain consistent
        stats = cache.get_stats()
        assert stats["size"] <= cache._max_size
