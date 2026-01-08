"""
High-performance HTTP client with connection pooling, retries, and caching.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from functools import wraps
from typing import Any

import httpx
from httpx import AsyncClient, Limits, Response, Timeout
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.core.cache import cache

logger = logging.getLogger(__name__)


@dataclass
class HttpClientConfig:
    """HTTP client configuration for optimal performance."""

    # Connection pool settings
    pool_limits: Limits = Limits(
        max_keepalive_connections=50,  # Increased keepalive connections
        max_connections=100,  # Total connection limit
        keepalive_expiry=30.0,  # Keepalive expiry
    )

    # Timeout settings disabled - no timeouts
    timeout: Timeout = Timeout(
        connect=None,  # No connection timeout
        read=None,  # No read timeout
        write=None,  # No write timeout
        pool=None,  # No pool acquisition timeout
    )

    # Retry settings
    max_retries: int = 3
    retry_backoff_factor: float = 1.0
    retry_jitter: bool = True

    # Performance settings
    enable_http2: bool = True
    enable_compression: bool = True
    follow_redirects: bool = True

    # Caching settings
    enable_response_cache: bool = True
    cache_ttl: int = 300  # 5 minutes
    cacheable_methods: list[str] = field(default_factory=lambda: ["GET"])
    cacheable_status_codes: list[int] = field(default_factory=lambda: [200, 301, 302])


class PerformanceMetrics:
    """Track HTTP client performance metrics."""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.cache_hits = 0
        self.total_response_time = 0.0
        self.average_response_time = 0.0
        self.start_time = time.time()

    def record_request(self, response_time: float, error: bool = False, cache_hit: bool = False):
        """Record a request metric."""
        self.request_count += 1
        if error:
            self.error_count += 1
        if cache_hit:
            self.cache_hits += 1

        self.total_response_time += response_time
        self.average_response_time = self.total_response_time / self.request_count

    def get_stats(self) -> dict[str, Any]:
        """Get current performance statistics."""
        uptime = time.time() - self.start_time
        error_rate = (self.error_count / self.request_count * 100) if self.request_count > 0 else 0
        cache_hit_rate = (
            (self.cache_hits / self.request_count * 100) if self.request_count > 0 else 0
        )

        return {
            "request_count": self.request_count,
            "error_count": self.error_count,
            "error_rate_percent": error_rate,
            "cache_hits": self.cache_hits,
            "cache_hit_rate_percent": cache_hit_rate,
            "average_response_time_ms": self.average_response_time * 1000,
            "uptime_seconds": uptime,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
        }


class CachedResponse:
    """Cached response wrapper."""

    def __init__(self, response: Response):
        self.content = response.content
        self.status_code = response.status_code
        self.headers = dict(response.headers)
        self.cached_at = time.time()

    def is_expired(self, ttl: int) -> bool:
        """Check if cached response is expired."""
        return time.time() - self.cached_at > ttl


class OptimizedHttpClient:
    """High-performance HTTP client with connection pooling and optimizations."""

    def __init__(self, config: HttpClientConfig = None):
        self.config = config or HttpClientConfig()
        self._client: AsyncClient | None = None
        self.metrics = PerformanceMetrics()
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the HTTP client with optimized settings."""
        if self._initialized:
            return

        try:
            self._client = AsyncClient(
                limits=self.config.pool_limits,
                timeout=self.config.timeout,
                http2=self.config.enable_http2,
                follow_redirects=self.config.follow_redirects,
                event_hooks={"response": [self._log_response_time]},
            )
            self._initialized = True
            logger.info("Optimized HTTP client initialized")

        except Exception as e:
            logger.error(f"HTTP client initialization failed: {e}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._initialized = False
            logger.info("HTTP client shutdown completed")

    def _generate_cache_key(self, method: str, url: str, **kwargs) -> str:
        """Generate cache key for request."""
        cache_data = {
            "method": method,
            "url": url,
            "headers": {
                k: v
                for k, v in kwargs.get("headers", {}).items()
                if k.lower() not in ["authorization", "cookie"]
            },
            "params": kwargs.get("params", {}),
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    async def _log_response_time(self, response: Response) -> None:
        """Log response time for metrics."""
        response_time = response.elapsed.total_seconds() if response.elapsed else 0
        self.metrics.record_request(response_time)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
    )
    async def request(self, method: str, url: str, **kwargs) -> Response:
        """Make HTTP request with caching and retries."""
        if not self._client:
            await self.initialize()

        start_time = time.time()
        cache_hit = False

        # Check cache for GET requests
        if self.config.enable_response_cache and method.upper() in self.config.cacheable_methods:
            cache_key = f"http_response:{self._generate_cache_key(method, url, **kwargs)}"
            cached_data = await cache.get(cache_key)

            if cached_data:
                cached_response = CachedResponse.__new__(CachedResponse)
                cached_response.__dict__.update(cached_data)

                if not cached_response.is_expired(self.config.cache_ttl):
                    cache_hit = True
                    self.metrics.record_request(0, cache_hit=True)

                    # Create Response object from cached data
                    response = httpx.Response(
                        status_code=cached_response.status_code,
                        content=cached_response.content,
                        headers=cached_response.headers,
                        request=httpx.Request(method, url),
                    )

                    logger.debug(f"Cache hit for {method} {url}")
                    return response

        # Make actual request
        try:
            response = await self._client.request(method, url, **kwargs)
            response_time = time.time() - start_time

            # Cache successful responses
            if (
                self.config.enable_response_cache
                and method.upper() in self.config.cacheable_methods
                and response.status_code in self.config.cacheable_status_codes
            ):
                cache_key = f"http_response:{self._generate_cache_key(method, url, **kwargs)}"
                cached_response = CachedResponse(response)
                await cache.set(cache_key, cached_response.__dict__, ttl=self.config.cache_ttl)

            self.metrics.record_request(response_time, cache_hit=cache_hit)
            return response

        except Exception as e:
            response_time = time.time() - start_time
            self.metrics.record_request(response_time, error=True)
            logger.error(f"HTTP request failed: {method} {url} - {e}")
            raise

    async def get(self, url: str, params: dict | None = None, **kwargs) -> Response:
        """GET request."""
        return await self.request("GET", url, params=params, **kwargs)

    async def post(self, url: str, data: dict | None = None, json: dict | None = None, **kwargs) -> Response:
        """POST request."""
        if json and not data:
            data = json
            kwargs["headers"] = kwargs.get("headers", {})
            kwargs["headers"]["Content-Type"] = "application/json"

        return await self.request("POST", url, data=data, **kwargs)

    async def put(self, url: str, data: dict | None = None, json: dict | None = None, **kwargs) -> Response:
        """PUT request."""
        if json and not data:
            data = json
            kwargs["headers"] = kwargs.get("headers", {})
            kwargs["headers"]["Content-Type"] = "application/json"

        return await self.request("PUT", url, data=data, **kwargs)

    async def delete(self, url: str, **kwargs) -> Response:
        """DELETE request."""
        return await self.request("DELETE", url, **kwargs)

    async def get_json(self, url: str, params: dict | None = None, **kwargs) -> dict[str, Any]:
        """GET request and return JSON."""
        response = await self.get(url, params=params, **kwargs)
        response.raise_for_status()
        return response.json()

    async def post_json(self, url: str, data: dict | None = None, **kwargs) -> dict[str, Any]:
        """POST request with JSON and return JSON."""
        response = await self.post(url, json=data, **kwargs)
        response.raise_for_status()
        return response.json()

    def get_metrics(self) -> dict[str, Any]:
        """Get HTTP client performance metrics."""
        return self.metrics.get_stats()

    async def health_check(self) -> dict[str, Any]:
        """Health check for HTTP client."""
        try:
            if not self._client:
                await self.initialize()

            # Test with a simple request
            start_time = time.time()
            await self.get("https://httpbin.org/get", timeout=5.0)
            response_time = time.time() - start_time

            return {
                "status": "healthy",
                "response_time_ms": response_time * 1000,
                "metrics": self.get_metrics(),
            }

        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "metrics": self.get_metrics()}


class BatchHttpClient:
    """Batch HTTP client for concurrent requests."""

    def __init__(self, client: OptimizedHttpClient):
        self.client = client

    async def batch_request(self, requests: list[dict[str, Any]]) -> list[Response]:
        """Execute multiple requests concurrently."""
        tasks = []
        for req in requests:
            method = req.pop("method", "GET")
            url = req.pop("url")
            tasks.append(self.client.request(method, url, **req))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        responses = []
        for result in results:
            if isinstance(result, Exception):
                responses.append(httpx.Response(500, content=str(result).encode()))
            else:
                responses.append(result)

        return responses

    async def batch_get(self, urls: list[str], **kwargs) -> list[Response]:
        """Batch GET requests."""
        requests = [{"url": url, **kwargs} for url in urls]
        return await self.batch_request(requests)


# Global optimized HTTP client
http_client = OptimizedHttpClient()
batch_client = BatchHttpClient(http_client)


# Decorator for caching HTTP requests
def http_cached(ttl: int = 300, key_params: list[str] | None = None):
    """Decorator for caching HTTP-based function calls."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key_data = {
                "function": func.__name__,
                "args": args,
                "filtered_kwargs": {
                    k: v for k, v in kwargs.items() if not key_params or k in key_params
                },
            }
            cache_key = f"http_func:{hashlib.sha256(json.dumps(cache_key_data, sort_keys=True).encode()).hexdigest()}"

            # Try cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function
            try:
                result = await func(*args, **kwargs)
                await cache.set(cache_key, result, ttl=ttl)
                return result
            except Exception as e:
                logger.error(f"HTTP function error for {func.__name__}: {e}")
                raise

        return wrapper

    return decorator


async def initialize_http_client():
    """Initialize the HTTP client system."""
    await http_client.initialize()
    logger.info("HTTP client system initialized")


async def shutdown_http_client():
    """Shutdown the HTTP client system."""
    await http_client.shutdown()
    logger.info("HTTP client system shutdown")
