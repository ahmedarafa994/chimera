"""
Optimized LLM Service with advanced performance features.

This module provides comprehensive optimizations for LLM operations:
- Request batching and parallel execution
- Connection pooling per provider
- Advanced caching with Redis support
- Token usage optimization
- Circuit breaker resilience
- Performance monitoring and metrics
"""

import asyncio
import hashlib
import json
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from redis import asyncio as aioredis

from app.core.config import settings
from app.core.logging import logger
from app.domain.interfaces import LLMProvider
from app.domain.models import (
    GenerationConfig,
    PromptRequest,
    PromptResponse,
)
from app.services.llm_service import LLMService as BaseLLMService


@dataclass
class BatchRequest:
    """Represents a batched LLM request."""

    request_id: str
    request: PromptRequest
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # Higher = more priority


@dataclass
class ConnectionPoolConfig:
    """Configuration for connection pools."""

    max_connections: int = 20
    min_connections: int = 5
    max_idle_time: float = 300.0  # 5 minutes
    connection_timeout: float = 30.0
    request_timeout: float = 120.0


@dataclass
class CacheConfig:
    """Configuration for caching systems."""

    # In-memory cache
    memory_cache_size: int = 1000
    memory_cache_ttl: int = 900  # 15 minutes

    # Redis cache
    use_redis: bool = True
    redis_ttl: int = 3600  # 1 hour
    redis_key_prefix: str = "chimera:llm:"

    # Cache strategies
    cache_deterministic_only: bool = True  # Only cache temperature=0
    cache_compression: bool = True


@dataclass
class BatchConfig:
    """Configuration for request batching."""

    max_batch_size: int = 5
    batch_timeout_ms: int = 100  # Wait up to 100ms to form batch
    max_concurrent_batches: int = 10
    enable_priority_batching: bool = True


class AdvancedCache:
    """
    Multi-tier caching system with Redis and in-memory tiers.
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self._memory_cache: dict[str, tuple[Any, float]] = {}
        self._redis_client: aioredis.Redis | None = None
        self._stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "sets": 0,
            "evictions": 0,
        }
        self._lock = asyncio.Lock()

    async def initialize(self):
        """Initialize Redis connection if enabled."""
        if self.config.use_redis:
            try:
                redis_url = settings.REDIS_URL or "redis://localhost:6379/0"
                self._redis_client = aioredis.from_url(
                    redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                    max_connections=10,
                )
                # Test connection
                await self._redis_client.ping()
                logger.info("Redis cache initialized successfully")
            except Exception as e:
                logger.warning(f"Redis initialization failed: {e}. Using memory cache only.")
                self._redis_client = None

    def _generate_key(self, request: PromptRequest) -> str:
        """Generate cache key for request."""
        key_data = {
            "prompt": request.prompt[:500],  # Limit key size
            "provider": request.provider.value if request.provider else "default",
            "model": request.model or "default",
            "temperature": request.config.temperature if request.config else 0.7,
            "max_tokens": request.config.max_output_tokens if request.config else 2048,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        return f"{self.config.redis_key_prefix}{key_hash}"

    async def get(self, request: PromptRequest) -> PromptResponse | None:
        """Get cached response, checking memory first then Redis."""
        key = self._generate_key(request)

        # Check memory cache first
        async with self._lock:
            if key in self._memory_cache:
                response, timestamp = self._memory_cache[key]
                if time.time() - timestamp < self.config.memory_cache_ttl:
                    self._stats["memory_hits"] += 1
                    return response
                else:
                    del self._memory_cache[key]
                    self._stats["evictions"] += 1

        self._stats["memory_misses"] += 1

        # Check Redis cache
        if self._redis_client:
            try:
                cached_data = await self._redis_client.get(key)
                if cached_data:
                    response = PromptResponse.model_validate_json(cached_data)
                    self._stats["redis_hits"] += 1

                    # Populate memory cache for faster access
                    async with self._lock:
                        if len(self._memory_cache) >= self.config.memory_cache_size:
                            # Remove oldest entry
                            oldest_key = min(
                                self._memory_cache.keys(),
                                key=lambda k: self._memory_cache[k][1]
                            )
                            del self._memory_cache[oldest_key]
                            self._stats["evictions"] += 1

                        self._memory_cache[key] = (response, time.time())

                    return response
            except Exception as e:
                logger.warning(f"Redis cache read failed: {e}")

        self._stats["redis_misses"] += 1
        return None

    async def set(self, request: PromptRequest, response: PromptResponse) -> None:
        """Cache response in both memory and Redis."""
        # Only cache deterministic responses if configured
        if (
            self.config.cache_deterministic_only
            and request.config
            and request.config.temperature > 0
        ):
            return

        key = self._generate_key(request)
        current_time = time.time()

        # Cache in memory
        async with self._lock:
            if len(self._memory_cache) >= self.config.memory_cache_size:
                # LRU eviction
                oldest_key = min(
                    self._memory_cache.keys(),
                    key=lambda k: self._memory_cache[k][1]
                )
                del self._memory_cache[oldest_key]
                self._stats["evictions"] += 1

            self._memory_cache[key] = (response, current_time)

        # Cache in Redis
        if self._redis_client:
            try:
                response_json = response.model_dump_json()
                await self._redis_client.setex(
                    key,
                    self.config.redis_ttl,
                    response_json
                )
            except Exception as e:
                logger.warning(f"Redis cache write failed: {e}")

        self._stats["sets"] += 1

    async def clear(self) -> None:
        """Clear all caches."""
        async with self._lock:
            self._memory_cache.clear()

        if self._redis_client:
            try:
                # Clear only our keys
                pattern = f"{self.config.redis_key_prefix}*"
                keys = await self._redis_client.keys(pattern)
                if keys:
                    await self._redis_client.delete(*keys)
            except Exception as e:
                logger.warning(f"Redis cache clear failed: {e}")

        logger.info("All caches cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = sum([
            self._stats["memory_hits"],
            self._stats["memory_misses"]
        ])

        return {
            **self._stats,
            "memory_size": len(self._memory_cache),
            "memory_hit_rate": (
                self._stats["memory_hits"] / total_requests
                if total_requests > 0 else 0
            ),
            "redis_enabled": self._redis_client is not None,
        }


class ConnectionPool:
    """Connection pool for LLM providers."""

    def __init__(self, provider: LLMProvider, config: ConnectionPoolConfig):
        self.provider = provider
        self.config = config
        self._connections: deque = deque()
        self._active_connections = 0
        self._lock = asyncio.Lock()
        self._stats = {
            "created": 0,
            "reused": 0,
            "expired": 0,
            "errors": 0,
        }

    async def get_connection(self) -> LLMProvider:
        """Get a connection from the pool."""
        async with self._lock:
            # Try to reuse an existing connection
            while self._connections:
                connection, timestamp = self._connections.popleft()
                if time.time() - timestamp < self.config.max_idle_time:
                    self._stats["reused"] += 1
                    return connection
                else:
                    self._stats["expired"] += 1

            # Create new connection if under limit
            if self._active_connections < self.config.max_connections:
                self._active_connections += 1
                self._stats["created"] += 1
                return self.provider

        # Wait for connection to become available
        while True:
            await asyncio.sleep(0.01)
            async with self._lock:
                if self._connections:
                    connection, timestamp = self._connections.popleft()
                    if time.time() - timestamp < self.config.max_idle_time:
                        self._stats["reused"] += 1
                        return connection

    async def return_connection(self, connection: LLMProvider) -> None:
        """Return connection to pool."""
        async with self._lock:
            if len(self._connections) < self.config.max_connections:
                self._connections.append((connection, time.time()))
            else:
                self._active_connections -= 1

    def get_stats(self) -> dict[str, Any]:
        """Get pool statistics."""
        return {
            **self._stats,
            "active_connections": self._active_connections,
            "pooled_connections": len(self._connections),
            "max_connections": self.config.max_connections,
        }


class BatchProcessor:
    """Processes batched LLM requests for improved throughput."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self._pending_batches: dict[str, list[BatchRequest]] = defaultdict(list)
        self._batch_timers: dict[str, asyncio.Handle] = {}
        self._active_batches = 0
        self._lock = asyncio.Lock()
        self._stats = {
            "total_requests": 0,
            "batched_requests": 0,
            "batch_count": 0,
            "avg_batch_size": 0.0,
        }

    async def submit_request(
        self,
        request: PromptRequest,
        provider_func: Callable,
        priority: int = 0
    ) -> PromptResponse:
        """Submit a request for batched processing."""
        self._stats["total_requests"] += 1

        batch_key = self._get_batch_key(request)
        batch_request = BatchRequest(
            request_id=str(time.time()),
            request=request,
            future=asyncio.Future(),
            priority=priority
        )

        async with self._lock:
            self._pending_batches[batch_key].append(batch_request)

            # Sort by priority if enabled
            if self.config.enable_priority_batching:
                self._pending_batches[batch_key].sort(
                    key=lambda x: x.priority,
                    reverse=True
                )

            # Check if we should process this batch
            should_process = (
                len(self._pending_batches[batch_key]) >= self.config.max_batch_size
                or self._active_batches < self.config.max_concurrent_batches
            )

            if should_process and batch_key not in self._batch_timers:
                # Schedule batch processing
                self._batch_timers[batch_key] = asyncio.get_event_loop().call_later(
                    self.config.batch_timeout_ms / 1000,
                    lambda: asyncio.create_task(
                        self._process_batch(batch_key, provider_func)
                    )
                )

        return await batch_request.future

    def _get_batch_key(self, request: PromptRequest) -> str:
        """Generate batching key for request."""
        return f"{request.provider}:{request.model}:{request.config.temperature if request.config else 0.7}"

    async def _process_batch(self, batch_key: str, provider_func: Callable) -> None:
        """Process a batch of requests."""
        async with self._lock:
            batch_requests = self._pending_batches.pop(batch_key, [])
            if batch_key in self._batch_timers:
                self._batch_timers[batch_key].cancel()
                del self._batch_timers[batch_key]

            if not batch_requests:
                return

            self._active_batches += 1
            self._stats["batch_count"] += 1
            self._stats["batched_requests"] += len(batch_requests)
            self._stats["avg_batch_size"] = (
                self._stats["batched_requests"] / self._stats["batch_count"]
            )

        try:
            # Process requests in parallel
            tasks = []
            for batch_req in batch_requests:
                task = asyncio.create_task(
                    self._process_single_request(batch_req, provider_func)
                )
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

        finally:
            async with self._lock:
                self._active_batches -= 1

    async def _process_single_request(
        self,
        batch_req: BatchRequest,
        provider_func: Callable
    ) -> None:
        """Process a single request within a batch."""
        try:
            response = await provider_func(batch_req.request)
            batch_req.future.set_result(response)
        except Exception as e:
            batch_req.future.set_exception(e)

    def get_stats(self) -> dict[str, Any]:
        """Get batch processing statistics."""
        return {
            **self._stats,
            "pending_batches": len(self._pending_batches),
            "active_batches": self._active_batches,
            "batch_efficiency": (
                self._stats["batched_requests"] / self._stats["total_requests"]
                if self._stats["total_requests"] > 0 else 0
            ),
        }


class OptimizedLLMService(BaseLLMService):
    """
    Optimized LLM Service with advanced performance features.

    Features:
    - Advanced multi-tier caching (Redis + in-memory)
    - Request batching for improved throughput
    - Connection pooling per provider
    - Enhanced circuit breaker patterns
    - Performance monitoring and metrics
    - Token usage optimization
    """

    def __init__(
        self,
        cache_config: CacheConfig | None = None,
        batch_config: BatchConfig | None = None,
        pool_config: ConnectionPoolConfig | None = None,
    ):
        super().__init__()

        # Configuration
        self.cache_config = cache_config or CacheConfig()
        self.batch_config = batch_config or BatchConfig()
        self.pool_config = pool_config or ConnectionPoolConfig()

        # Components
        self._advanced_cache = AdvancedCache(self.cache_config)
        self._batch_processor = BatchProcessor(self.batch_config)
        self._connection_pools: dict[str, ConnectionPool] = {}

        # Performance tracking
        self._request_times: deque = deque(maxlen=1000)
        self._token_usage: dict[str, int] = defaultdict(int)
        self._performance_stats = {
            "requests_processed": 0,
            "total_tokens_used": 0,
            "avg_response_time_ms": 0.0,
            "cache_hit_rate": 0.0,
            "error_rate": 0.0,
        }

        # Background tasks
        self._monitoring_task: asyncio.Task | None = None
        self._cleanup_task: asyncio.Task | None = None

    async def initialize(self) -> None:
        """Initialize the optimized service."""
        await self._advanced_cache.initialize()

        # Start background monitoring
        self._monitoring_task = asyncio.create_task(self._performance_monitor())
        self._cleanup_task = asyncio.create_task(self._cleanup_task_runner())

        logger.info("OptimizedLLMService initialized successfully")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()

        # Wait for pending batches to complete
        while self._batch_processor._active_batches > 0:
            await asyncio.sleep(0.1)

        logger.info("OptimizedLLMService shutdown complete")

    def register_provider(self, name: str, provider: LLMProvider, is_default: bool = False):
        """Register provider with connection pooling."""
        super().register_provider(name, provider, is_default)

        # Create connection pool for this provider
        normalized_name = self._normalize_provider_name(name) or name
        self._connection_pools[normalized_name] = ConnectionPool(
            provider, self.pool_config
        )

        logger.info(f"Provider {normalized_name} registered with connection pool")

    async def generate_text(self, request: PromptRequest) -> PromptResponse:
        """
        Optimized text generation with caching, batching, and pooling.
        """
        start_time = time.time()

        try:
            # Check cache first
            cached_response = await self._advanced_cache.get(request)
            if cached_response:
                logger.debug("Returning cached LLM response")
                self._update_performance_stats(start_time, cached=True)
                return cached_response

            # Determine provider
            provider_name = None
            if request.provider:
                provider_name = request.provider.value

            provider = self.get_provider(provider_name)
            circuit_name = provider_name or self._default_provider

            # Use batch processing for improved throughput
            response = await self._batch_processor.submit_request(
                request,
                lambda req: self._execute_with_pooling(circuit_name, provider, req),
                priority=1 if request.config and request.config.temperature == 0 else 0
            )

            # Cache the response
            await self._advanced_cache.set(request, response)

            # Update performance tracking
            self._update_performance_stats(start_time, cached=False)
            self._track_token_usage(response)

            return response

        except Exception as e:
            self._performance_stats["error_rate"] += 1
            raise e

    async def _execute_with_pooling(
        self,
        circuit_name: str,
        provider: LLMProvider,
        request: PromptRequest
    ) -> PromptResponse:
        """Execute request using connection pooling and circuit breaker."""
        # Get connection from pool
        pool = self._connection_pools.get(circuit_name)
        if pool:
            connection = await pool.get_connection()
            try:
                response = await self._call_with_circuit_breaker(
                    circuit_name, connection.generate, request
                )
                return response
            finally:
                await pool.return_connection(connection)
        else:
            # Fallback to direct execution
            return await self._call_with_circuit_breaker(
                circuit_name, provider.generate, request
            )

    def _update_performance_stats(self, start_time: float, cached: bool = False) -> None:
        """Update performance statistics."""
        response_time_ms = (time.time() - start_time) * 1000
        self._request_times.append(response_time_ms)

        self._performance_stats["requests_processed"] += 1

        if not cached:
            # Update average response time (excluding cached responses)
            non_cached_times = [t for t in self._request_times if t > 1]  # Cached are usually <1ms
            if non_cached_times:
                self._performance_stats["avg_response_time_ms"] = sum(non_cached_times) / len(non_cached_times)

    def _track_token_usage(self, response: PromptResponse) -> None:
        """Track token usage for cost analysis."""
        if response.usage_metadata:
            total_tokens = response.usage_metadata.get("total_token_count", 0)
            self._token_usage[response.provider] += total_tokens
            self._performance_stats["total_tokens_used"] += total_tokens

    async def _performance_monitor(self) -> None:
        """Background task for performance monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute

                # Update cache hit rate
                cache_stats = self._advanced_cache.get_stats()
                total_cache_requests = cache_stats["memory_hits"] + cache_stats["memory_misses"]
                if total_cache_requests > 0:
                    self._performance_stats["cache_hit_rate"] = (
                        (cache_stats["memory_hits"] + cache_stats["redis_hits"])
                        / total_cache_requests
                    )

                # Log performance metrics
                logger.info(
                    f"Performance metrics - "
                    f"Requests: {self._performance_stats['requests_processed']}, "
                    f"Avg response: {self._performance_stats['avg_response_time_ms']:.2f}ms, "
                    f"Cache hit rate: {self._performance_stats['cache_hit_rate']:.2%}, "
                    f"Total tokens: {self._performance_stats['total_tokens_used']}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")

    async def _cleanup_task_runner(self) -> None:
        """Background cleanup tasks."""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes

                # Clean expired cache entries
                # This is handled by the cache itself during access

                # Log pool statistics
                for name, pool in self._connection_pools.items():
                    pool_stats = pool.get_stats()
                    logger.debug(f"Pool {name}: {pool_stats}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_stats = self._advanced_cache.get_stats()
        batch_stats = self._batch_processor.get_stats()

        pool_stats = {}
        for name, pool in self._connection_pools.items():
            pool_stats[name] = pool.get_stats()

        return {
            "service": self._performance_stats,
            "cache": cache_stats,
            "batching": batch_stats,
            "connection_pools": pool_stats,
            "token_usage_by_provider": dict(self._token_usage),
            "recent_response_times_ms": list(self._request_times)[-10:],  # Last 10 requests
        }

    async def optimize_for_high_load(self) -> None:
        """Optimize settings for high-load scenarios."""
        # Increase batch sizes
        self.batch_config.max_batch_size = 10
        self.batch_config.batch_timeout_ms = 50

        # Increase connection pool sizes
        self.pool_config.max_connections = 50

        # Optimize cache settings
        self.cache_config.memory_cache_size = 2000

        logger.info("Service optimized for high-load scenarios")

    async def warmup_cache(self, common_prompts: list[str]) -> None:
        """Warm up cache with common prompts."""
        logger.info(f"Warming up cache with {len(common_prompts)} common prompts")

        tasks = []
        for prompt in common_prompts:
            request = PromptRequest(
                prompt=prompt,
                config=GenerationConfig(temperature=0),  # Deterministic for caching
            )
            task = asyncio.create_task(self.generate_text(request))
            tasks.append(task)

        # Process in smaller batches to avoid overwhelming the service
        batch_size = 5
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            try:
                await asyncio.gather(*batch, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Cache warmup batch failed: {e}")

            # Small delay between batches
            await asyncio.sleep(0.1)

        logger.info("Cache warmup completed")


# Global optimized service instance
optimized_llm_service = OptimizedLLMService()
