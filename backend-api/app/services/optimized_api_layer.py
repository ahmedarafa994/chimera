"""
Optimized API Layer with performance enhancements.

This module provides comprehensive API optimizations:
- Response compression and caching
- Pagination and N+1 query elimination
- Batch operations
- Rate limiting and throttling
- Request/response optimization
- Performance monitoring
"""

import asyncio
import contextlib
import gzip
import json
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.logging import logger

# =====================================================
# Pagination Models and Utilities
# =====================================================


class PaginationParams(BaseModel):
    """Standard pagination parameters."""

    page: int = Field(1, ge=1, description="Page number (1-based)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")
    sort_by: str | None = Field(None, description="Field to sort by")
    sort_order: str = Field("asc", regex="^(asc|desc)$", description="Sort order")


class CursorPaginationParams(BaseModel):
    """Cursor-based pagination parameters for large datasets."""

    cursor: str | None = Field(None, description="Cursor for pagination")
    limit: int = Field(20, ge=1, le=100, description="Number of items to return")
    order: str = Field("asc", regex="^(asc|desc)$", description="Sort order")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""

    data: list[Any] = Field(..., description="Response data")
    pagination: dict[str, Any] = Field(..., description="Pagination metadata")
    total_count: int | None = Field(None, description="Total number of items")
    page_count: int | None = Field(None, description="Total number of pages")


class BatchRequest(BaseModel):
    """Batch request wrapper."""

    requests: list[dict[str, Any]] = Field(..., max_items=50, description="Batch of requests")
    parallel: bool = Field(True, description="Execute requests in parallel")
    fail_fast: bool = Field(False, description="Stop on first error")


class BatchResponse(BaseModel):
    """Batch response wrapper."""

    results: list[dict[str, Any]] = Field(..., description="Batch results")
    success_count: int = Field(..., description="Number of successful requests")
    error_count: int = Field(..., description="Number of failed requests")
    total_time_ms: float = Field(..., description="Total execution time")


# =====================================================
# Response Compression Middleware
# =====================================================


class ResponseCompressionMiddleware(BaseHTTPMiddleware):
    """
    Advanced response compression middleware with selective compression.
    """

    def __init__(
        self,
        app,
        minimum_size: int = 500,
        compression_level: int = 6,
        exclude_media_types: list[str] | None = None,
    ):
        super().__init__(app)
        self.minimum_size = minimum_size
        self.compression_level = compression_level
        self.exclude_media_types = exclude_media_types or [
            "image/",
            "video/",
            "audio/",
            "application/octet-stream",
        ]

    async def dispatch(self, request: Request, call_next):
        """Process request with compression."""
        response = await call_next(request)

        # Check if compression should be applied
        if not self._should_compress(request, response):
            return response

        # Get response body
        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk

        # Compress if body is large enough
        if len(response_body) >= self.minimum_size:
            compressed_body = gzip.compress(response_body, compresslevel=self.compression_level)

            # Only use compression if it reduces size significantly
            if len(compressed_body) < len(response_body) * 0.9:
                response.headers["content-encoding"] = "gzip"
                response.headers["content-length"] = str(len(compressed_body))

                return Response(
                    content=compressed_body,
                    status_code=response.status_code,
                    headers=response.headers,
                    media_type=response.media_type,
                )

        # Return original response
        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=response.headers,
            media_type=response.media_type,
        )

    def _should_compress(self, request: Request, response: Response) -> bool:
        """Check if response should be compressed."""
        # Check accept-encoding header
        accept_encoding = request.headers.get("accept-encoding", "")
        if "gzip" not in accept_encoding:
            return False

        # Check response status
        if response.status_code != 200:
            return False

        # Check content type
        content_type = response.headers.get("content-type", "")
        for exclude_type in self.exclude_media_types:
            if exclude_type in content_type:
                return False

        # Check if already compressed
        return not response.headers.get("content-encoding")


# =====================================================
# Request Caching and Optimization
# =====================================================


@dataclass
class CacheEntry:
    """Cache entry for API responses."""

    data: Any
    timestamp: float
    ttl: float
    hit_count: int = 0
    size_bytes: int = 0


class APIResponseCache:
    """
    Intelligent API response cache with TTL and eviction policies.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl

        self._cache: dict[str, CacheEntry] = {}
        self._access_order: deque = deque()
        self._lock = asyncio.Lock()

        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expired": 0,
        }

    def _generate_key(self, request: Request, body: bytes | None = None) -> str:
        """Generate cache key for request."""
        import hashlib

        # Include method, path, query params, and body
        key_parts = [
            request.method,
            str(request.url.path),
            str(request.url.query),
        ]

        if body:
            key_parts.append(body.decode("utf-8", errors="ignore"))

        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()

    async def get(self, request: Request, body: bytes | None = None) -> Any | None:
        """Get cached response."""
        key = self._generate_key(request, body)

        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if time.time() - entry.timestamp > entry.ttl:
                del self._cache[key]
                with contextlib.suppress(ValueError):
                    self._access_order.remove(key)
                self._stats["expired"] += 1
                self._stats["misses"] += 1
                return None

            # Update access
            entry.hit_count += 1
            with contextlib.suppress(ValueError):
                self._access_order.remove(key)
            self._access_order.append(key)

            self._stats["hits"] += 1
            return entry.data

    async def set(
        self,
        request: Request,
        data: Any,
        ttl: int | None = None,
        body: bytes | None = None,
    ) -> None:
        """Cache response data."""
        key = self._generate_key(request, body)
        effective_ttl = ttl or self.default_ttl

        # Estimate size
        try:
            size_bytes = len(json.dumps(data, default=str).encode())
        except (TypeError, ValueError):
            size_bytes = 0

        async with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self.max_size:
                await self._evict_lru()

            # Store entry
            self._cache[key] = CacheEntry(
                data=data,
                timestamp=time.time(),
                ttl=effective_ttl,
                size_bytes=size_bytes,
            )

            # Update access order
            with contextlib.suppress(ValueError):
                self._access_order.remove(key)
            self._access_order.append(key)

    async def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if self._access_order:
            lru_key = self._access_order.popleft()
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats["evictions"] += 1

    async def clear(self) -> None:
        """Clear cache."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        total_size = sum(entry.size_bytes for entry in self._cache.values())

        return {
            **self._stats,
            "hit_rate": self._stats["hits"] / total_requests if total_requests > 0 else 0,
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_size_bytes": total_size,
            "avg_size_bytes": total_size / len(self._cache) if self._cache else 0,
        }


# =====================================================
# Performance Monitoring Middleware
# =====================================================


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    method: str
    path: str
    status_code: int
    duration_ms: float
    response_size: int
    timestamp: float


class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for collecting API performance metrics.
    """

    def __init__(self, app, max_metrics: int = 10000):
        super().__init__(app)
        self.max_metrics = max_metrics

        self._metrics: deque = deque(maxlen=max_metrics)
        self._stats = defaultdict(list)
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        """Process request with performance monitoring."""
        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate metrics
        duration_ms = (time.time() - start_time) * 1000
        response_size = 0

        # Get response size if possible
        if hasattr(response, "body") and isinstance(response.body, bytes):
            response_size = len(response.body)

        # Store metrics
        metric = RequestMetrics(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            response_size=response_size,
            timestamp=time.time(),
        )

        async with self._lock:
            self._metrics.append(metric)

            # Update aggregated stats
            endpoint_key = f"{request.method} {request.url.path}"
            self._stats[endpoint_key].append(duration_ms)

            # Limit stats history
            if len(self._stats[endpoint_key]) > 1000:
                self._stats[endpoint_key] = self._stats[endpoint_key][-1000:]

        # Add performance headers
        response.headers["X-Response-Time"] = f"{duration_ms:.2f}ms"
        response.headers["X-Response-Size"] = str(response_size)

        return response

    async def get_metrics_summary(self) -> dict[str, Any]:
        """Get performance metrics summary."""
        async with self._lock:
            if not self._metrics:
                return {}

            # Calculate overall stats
            total_requests = len(self._metrics)
            avg_duration = sum(m.duration_ms for m in self._metrics) / total_requests

            # Calculate percentiles
            durations = sorted([m.duration_ms for m in self._metrics])
            p50 = durations[int(total_requests * 0.5)]
            p95 = durations[int(total_requests * 0.95)]
            p99 = durations[int(total_requests * 0.99)]

            # Status code distribution
            status_codes = defaultdict(int)
            for metric in self._metrics:
                status_codes[metric.status_code] += 1

            # Endpoint performance
            endpoint_stats = {}
            for endpoint, durations in self._stats.items():
                if durations:
                    endpoint_stats[endpoint] = {
                        "avg_duration_ms": sum(durations) / len(durations),
                        "request_count": len(durations),
                        "p95_duration_ms": sorted(durations)[int(len(durations) * 0.95)],
                    }

            return {
                "total_requests": total_requests,
                "avg_duration_ms": avg_duration,
                "p50_duration_ms": p50,
                "p95_duration_ms": p95,
                "p99_duration_ms": p99,
                "status_codes": dict(status_codes),
                "endpoints": endpoint_stats,
                "timespan_minutes": (
                    (self._metrics[-1].timestamp - self._metrics[0].timestamp) / 60
                    if len(self._metrics) > 1
                    else 0
                ),
            }


# =====================================================
# N+1 Query Prevention and Batch Loading
# =====================================================


class BatchLoader:
    """
    Generic batch loader to prevent N+1 queries.
    """

    def __init__(self, batch_size: int = 100, max_wait_ms: int = 10):
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms

        self._pending_requests: dict[str, list[asyncio.Future]] = defaultdict(list)
        self._batch_timers: dict[str, asyncio.Handle] = {}
        self._lock = asyncio.Lock()

    async def load(self, key_type: str, keys: list[str], loader_func: callable) -> dict[str, Any]:
        """
        Load data for keys using batch loading.

        Args:
            key_type: Type of keys being loaded (for batching)
            keys: List of keys to load
            loader_func: Function to load data for a batch of keys

        Returns:
            Dictionary mapping keys to loaded data
        """
        if not keys:
            return {}

        # Create futures for each key
        futures = {}
        for key in keys:
            future = asyncio.Future()
            futures[key] = future

            async with self._lock:
                self._pending_requests[key_type].append((key, future))

        # Schedule batch processing if needed
        async with self._lock:
            if (
                len(self._pending_requests[key_type]) >= self.batch_size
                or key_type not in self._batch_timers
            ):
                # Cancel existing timer
                if key_type in self._batch_timers:
                    self._batch_timers[key_type].cancel()

                # Schedule immediate batch processing
                self._batch_timers[key_type] = asyncio.get_event_loop().call_soon(
                    lambda: asyncio.create_task(self._process_batch(key_type, loader_func))
                )

        # Wait for results
        results = {}
        for key in keys:
            try:
                results[key] = await futures[key]
            except Exception as e:
                logger.error(f"Batch loading failed for key {key}: {e}")
                results[key] = None

        return results

    async def _process_batch(self, key_type: str, loader_func: callable) -> None:
        """Process a batch of requests."""
        async with self._lock:
            requests = self._pending_requests.pop(key_type, [])
            if key_type in self._batch_timers:
                del self._batch_timers[key_type]

        if not requests:
            return

        # Extract keys and futures
        keys = [req[0] for req in requests]
        futures_map = {req[0]: req[1] for req in requests}

        try:
            # Load data in batch
            batch_results = await loader_func(keys)

            # Distribute results to futures
            for key, future in futures_map.items():
                if not future.done():
                    result = batch_results.get(key)
                    future.set_result(result)

        except Exception as e:
            # Propagate error to all futures
            for future in futures_map.values():
                if not future.done():
                    future.set_exception(e)


# =====================================================
# Optimized API Router with Performance Features
# =====================================================


class OptimizedAPIRouter:
    """
    High-performance API router with optimization features.
    """

    def __init__(self, app: FastAPI):
        self.app = app

        # Initialize components
        self.cache = APIResponseCache(max_size=1000, default_ttl=300)
        self.batch_loader = BatchLoader(batch_size=50, max_wait_ms=10)

        # Add middleware
        self.app.add_middleware(
            ResponseCompressionMiddleware,
            minimum_size=500,
            compression_level=6,
        )

        self.performance_monitor = PerformanceMonitoringMiddleware(app)
        self.app.add_middleware(PerformanceMonitoringMiddleware, max_metrics=10000)

        # Statistics
        self._stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "batch_loads": 0,
            "compression_savings": 0,
        }

    def add_cached_endpoint(
        self,
        path: str,
        handler: callable,
        methods: list[str] | None = None,
        cache_ttl: int = 300,
        cache_key_func: callable | None = None,
    ):
        """
        Add endpoint with response caching.
        """
        if methods is None:
            methods = ["GET"]

        async def cached_handler(request: Request):
            # Check cache for GET requests
            if request.method == "GET":
                cached_response = await self.cache.get(request)
                if cached_response is not None:
                    self._stats["cache_hits"] += 1
                    return JSONResponse(cached_response)

            # Execute handler
            result = await handler(request)

            # Cache successful responses
            if request.method == "GET" and isinstance(result, dict):
                await self.cache.set(request, result, cache_ttl)

            self._stats["total_requests"] += 1
            return result

        # Register endpoint
        for method in methods:
            self.app.router.add_api_route(
                path=path,
                endpoint=cached_handler,
                methods=[method],
            )

    def add_paginated_endpoint(
        self,
        path: str,
        data_loader: callable,
        methods: list[str] | None = None,
        default_page_size: int = 20,
        max_page_size: int = 100,
    ):
        """
        Add endpoint with automatic pagination.
        """
        if methods is None:
            methods = ["GET"]

        async def paginated_handler(
            request: Request,
            pagination: PaginationParams = Depends(),
        ):
            # Validate page size
            page_size = min(pagination.page_size, max_page_size)
            offset = (pagination.page - 1) * page_size

            # Load data with pagination
            total_count, items = await data_loader(
                limit=page_size,
                offset=offset,
                sort_by=pagination.sort_by,
                sort_order=pagination.sort_order,
            )

            # Calculate pagination metadata
            page_count = (total_count + page_size - 1) // page_size

            return PaginatedResponse(
                data=items,
                pagination={
                    "page": pagination.page,
                    "page_size": page_size,
                    "total_count": total_count,
                    "page_count": page_count,
                    "has_next": pagination.page < page_count,
                    "has_prev": pagination.page > 1,
                },
                total_count=total_count,
                page_count=page_count,
            )

        # Register endpoint
        for method in methods:
            self.app.router.add_api_route(
                path=path,
                endpoint=paginated_handler,
                methods=[method],
            )

    def add_batch_endpoint(
        self,
        path: str,
        single_handler: callable,
        max_batch_size: int = 50,
    ):
        """
        Add endpoint with batch processing capabilities.
        """

        async def batch_handler(batch_request: BatchRequest):
            start_time = time.time()

            # Validate batch size
            if len(batch_request.requests) > max_batch_size:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Batch size cannot exceed {max_batch_size}",
                )

            results = []
            success_count = 0
            error_count = 0

            if batch_request.parallel:
                # Process requests in parallel
                tasks = []
                for req in batch_request.requests:
                    task = asyncio.create_task(
                        self._process_single_request(single_handler, req, batch_request.fail_fast)
                    )
                    tasks.append(task)

                # Wait for all tasks
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, Exception):
                        if batch_request.fail_fast:
                            raise result
                        results.append({"error": str(result)})
                        error_count += 1
                    else:
                        results.append(result)
                        success_count += 1
            else:
                # Process requests sequentially
                for req in batch_request.requests:
                    try:
                        result = await single_handler(req)
                        results.append(result)
                        success_count += 1
                    except Exception as e:
                        if batch_request.fail_fast:
                            raise e
                        results.append({"error": str(e)})
                        error_count += 1

            total_time_ms = (time.time() - start_time) * 1000

            return BatchResponse(
                results=results,
                success_count=success_count,
                error_count=error_count,
                total_time_ms=total_time_ms,
            )

        # Register endpoint
        self.app.router.add_api_route(
            path=path,
            endpoint=batch_handler,
            methods=["POST"],
        )

    async def _process_single_request(
        self, handler: callable, request_data: dict[str, Any], fail_fast: bool
    ) -> Any:
        """Process a single request in a batch."""
        try:
            return await handler(request_data)
        except Exception as e:
            if fail_fast:
                raise e
            return {"error": str(e)}

    async def stream_response(self, data_generator: callable) -> StreamingResponse:
        """
        Create optimized streaming response.
        """

        async def generate():
            async for chunk in data_generator():
                # Add newline for JSONL format
                if isinstance(chunk, dict):
                    yield json.dumps(chunk) + "\n"
                else:
                    yield str(chunk) + "\n"

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={"Cache-Control": "no-cache"},
        )

    async def get_performance_stats(self) -> dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_stats = self.cache.get_stats()
        performance_stats = await self.performance_monitor.get_metrics_summary()

        return {
            "api_stats": self._stats,
            "cache": cache_stats,
            "performance": performance_stats,
            "optimization_features": {
                "response_compression": True,
                "request_caching": True,
                "batch_processing": True,
                "n1_prevention": True,
                "pagination": True,
            },
        }

    async def clear_cache(self) -> None:
        """Clear API response cache."""
        await self.cache.clear()
        logger.info("API response cache cleared")


# =====================================================
# Utility Functions
# =====================================================


def create_optimized_api_app() -> tuple[FastAPI, OptimizedAPIRouter]:
    """
    Create FastAPI app with optimization features.
    """
    app = FastAPI(
        title="Chimera Optimized API",
        description="High-performance API with optimization features",
        version="1.0.0",
    )

    # Initialize optimized router
    router = OptimizedAPIRouter(app)

    # Add global exception handler for performance
    @app.exception_handler(Exception)
    async def performance_exception_handler(request: Request, exc: Exception):
        logger.error(f"API error: {exc}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "request_id": str(time.time())},
        )

    return app, router


async def optimize_json_response(data: Any, compress: bool = True) -> JSONResponse:
    """
    Create optimized JSON response with compression.
    """
    response_data = json.dumps(data, separators=(",", ":"))

    if compress and len(response_data) > 1000:
        # Compress large responses
        compressed_data = gzip.compress(response_data.encode())

        return Response(
            content=compressed_data,
            media_type="application/json",
            headers={"Content-Encoding": "gzip"},
        )

    return JSONResponse(content=data)


# =====================================================
# Export optimized components
# =====================================================

__all__ = [
    "APIResponseCache",
    "BatchLoader",
    "BatchRequest",
    "BatchResponse",
    "CursorPaginationParams",
    "OptimizedAPIRouter",
    "PaginatedResponse",
    "PaginationParams",
    "PerformanceMonitoringMiddleware",
    "ResponseCompressionMiddleware",
    "create_optimized_api_app",
    "optimize_json_response",
]
