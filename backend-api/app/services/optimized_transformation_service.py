"""Optimized Transformation Service with performance enhancements.

This module provides comprehensive optimizations for prompt transformations:
- Async transformation pipelines
- Memory-efficient processing
- Parallel technique execution
- Advanced caching with compression
- Streaming transformations
- Resource monitoring
"""

import asyncio
import contextlib
import gc
import gzip
import logging
import pickle
import time
import weakref
from collections import deque
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import psutil

from app.core.config import config
from app.core.unified_errors import TransformationError
from app.domain.models import StreamChunk
from app.services.transformation_service import TransformationEngine as BaseEngine
from app.services.transformation_service import TransformationMetadata, TransformationResult

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for transformations."""

    total_transformations: int = 0
    avg_execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    parallel_efficiency: float = 0.0

    # Technique-specific metrics
    technique_performance: dict[str, float] = field(default_factory=dict)

    # Resource utilization
    cpu_usage_percent: float = 0.0
    memory_peak_mb: float = 0.0


@dataclass
class TransformationJob:
    """Represents a transformation job for async processing."""

    job_id: str
    prompt: str
    potency_level: int
    technique_suite: str
    priority: int = 0
    created_at: float = field(default_factory=time.time)
    future: asyncio.Future | None = None
    dependencies: set[str] = field(default_factory=set)


class CompressedCache:
    """Memory-efficient cache with compression and smart eviction."""

    def __init__(
        self,
        max_memory_mb: int = 100,
        compression_threshold: int = 1000,  # Compress items > 1KB
        ttl_seconds: int = 3600,
    ) -> None:
        self.max_memory_mb = max_memory_mb
        self.compression_threshold = compression_threshold
        self.ttl_seconds = ttl_seconds

        self._cache: dict[str, tuple[bytes, float, bool]] = {}  # value, timestamp, is_compressed
        self._access_order: deque = deque()
        self._current_memory_mb = 0.0
        self._lock = asyncio.Lock()

        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "compressions": 0,
            "decompressions": 0,
        }

    def _estimate_size_mb(self, data: bytes) -> float:
        """Estimate memory usage in MB."""
        return len(data) / (1024 * 1024)

    def _compress_data(self, data: Any) -> tuple[bytes, bool]:
        """Compress data if it exceeds threshold."""
        serialized = pickle.dumps(data)

        if len(serialized) > self.compression_threshold:
            compressed = gzip.compress(serialized)
            self._stats["compressions"] += 1
            return compressed, True

        return serialized, False

    def _decompress_data(self, data: bytes, is_compressed: bool) -> Any:
        """Decompress data if needed."""
        if is_compressed:
            decompressed = gzip.decompress(data)
            self._stats["decompressions"] += 1
            return pickle.loads(decompressed)

        return pickle.loads(data)

    async def get(self, key: str) -> Any | None:
        """Get item from cache."""
        async with self._lock:
            if key not in self._cache:
                self._stats["misses"] += 1
                return None

            data, timestamp, is_compressed = self._cache[key]

            # Check expiration
            if time.time() - timestamp > self.ttl_seconds:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # Update access order
            with contextlib.suppress(ValueError):
                self._access_order.remove(key)
            self._access_order.append(key)

            self._stats["hits"] += 1

        # Decompress outside lock to reduce contention
        return self._decompress_data(data, is_compressed)

    async def set(self, key: str, value: Any) -> bool:
        """Set item in cache with memory management."""
        compressed_data, is_compressed = self._compress_data(value)
        size_mb = self._estimate_size_mb(compressed_data)

        async with self._lock:
            # Check if we need to free memory
            while self._current_memory_mb + size_mb > self.max_memory_mb and self._cache:
                await self._evict_lru()

            # Store the item
            self._cache[key] = (compressed_data, time.time(), is_compressed)
            self._current_memory_mb += size_mb

            # Update access order
            with contextlib.suppress(ValueError):
                self._access_order.remove(key)
            self._access_order.append(key)

        return True

    async def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_order:
            return

        lru_key = self._access_order.popleft()
        if lru_key in self._cache:
            data, _, _ = self._cache[lru_key]
            size_mb = self._estimate_size_mb(data)
            del self._cache[lru_key]
            self._current_memory_mb -= size_mb
            self._stats["evictions"] += 1

    async def clear(self) -> None:
        """Clear cache."""
        async with self._lock:
            self._cache.clear()
            self._access_order.clear()
            self._current_memory_mb = 0.0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._stats["hits"] + self._stats["misses"]
        return {
            **self._stats,
            "hit_rate": self._stats["hits"] / total_requests if total_requests > 0 else 0,
            "memory_usage_mb": self._current_memory_mb,
            "max_memory_mb": self.max_memory_mb,
            "memory_utilization": self._current_memory_mb / self.max_memory_mb,
            "items_count": len(self._cache),
        }


class AsyncTransformationPipeline:
    """Async pipeline for parallel transformation processing."""

    def __init__(self, max_workers: int = 10, max_queue_size: int = 100) -> None:
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size

        self._job_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self._active_jobs: dict[str, TransformationJob] = {}
        self._completed_jobs: dict[str, TransformationResult] = weakref.WeakValueDictionary()

        self._workers: list[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        self._stats = {
            "jobs_processed": 0,
            "jobs_failed": 0,
            "avg_processing_time_ms": 0.0,
            "queue_length": 0,
            "active_workers": 0,
        }

    async def start(self) -> None:
        """Start the pipeline workers."""
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)

        logger.info(f"Async transformation pipeline started with {self.max_workers} workers")

    async def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._shutdown_event.set()

        # Cancel all workers
        for worker in self._workers:
            worker.cancel()

        # Wait for workers to finish
        await asyncio.gather(*self._workers, return_exceptions=True)

        logger.info("Async transformation pipeline stopped")

    async def submit_job(
        self,
        job: TransformationJob,
        transform_func: callable,
    ) -> str:
        """Submit a transformation job."""
        # Check dependencies
        for dep_id in job.dependencies:
            if dep_id not in self._completed_jobs and dep_id not in self._active_jobs:
                msg = f"Dependency {dep_id} not found"
                raise TransformationError(msg)

        # Add to queue
        try:
            await self._job_queue.put((job, transform_func), timeout=1.0)
            self._active_jobs[job.job_id] = job
            return job.job_id
        except TimeoutError:
            msg = "Pipeline queue is full"
            raise TransformationError(msg)

    async def get_result(self, job_id: str, timeout: float = 30.0) -> TransformationResult:
        """Get job result."""
        if job_id not in self._active_jobs:
            msg = f"Job {job_id} not found"
            raise TransformationError(msg)

        job = self._active_jobs[job_id]
        if not job.future:
            msg = f"Job {job_id} not started"
            raise TransformationError(msg)

        try:
            return await asyncio.wait_for(job.future, timeout=timeout)
        except TimeoutError:
            msg = f"Job {job_id} timed out"
            raise TransformationError(msg)

    async def _worker_loop(self, worker_id: str) -> None:
        """Worker loop for processing jobs."""
        logger.debug(f"Worker {worker_id} started")

        while not self._shutdown_event.is_set():
            try:
                # Get job from queue
                job, transform_func = await asyncio.wait_for(self._job_queue.get(), timeout=1.0)

                self._stats["active_workers"] += 1
                start_time = time.time()

                try:
                    # Check if dependencies are ready
                    await self._wait_for_dependencies(job)

                    # Execute transformation
                    result = await transform_func(
                        job.prompt,
                        job.potency_level,
                        job.technique_suite,
                    )

                    # Store result and signal completion
                    self._completed_jobs[job.job_id] = result
                    if job.future:
                        job.future.set_result(result)

                    # Update stats
                    processing_time = (time.time() - start_time) * 1000
                    self._update_processing_stats(processing_time, success=True)

                    logger.debug(f"Job {job.job_id} completed in {processing_time:.2f}ms")

                except Exception as e:
                    logger.exception(f"Job {job.job_id} failed: {e}")
                    if job.future:
                        job.future.set_exception(e)
                    self._update_processing_stats(0, success=False)

                finally:
                    # Cleanup
                    if job.job_id in self._active_jobs:
                        del self._active_jobs[job.job_id]
                    self._stats["active_workers"] -= 1

            except TimeoutError:
                continue  # Check shutdown event
            except Exception as e:
                logger.exception(f"Worker {worker_id} error: {e}")

    async def _wait_for_dependencies(self, job: TransformationJob) -> None:
        """Wait for job dependencies to complete."""
        for dep_id in job.dependencies:
            if dep_id in self._completed_jobs:
                continue  # Already completed

            if dep_id in self._active_jobs:
                dep_job = self._active_jobs[dep_id]
                if dep_job.future:
                    await dep_job.future
            else:
                msg = f"Dependency {dep_id} not found"
                raise TransformationError(msg)

    def _update_processing_stats(self, processing_time_ms: float, success: bool) -> None:
        """Update processing statistics."""
        if success:
            self._stats["jobs_processed"] += 1

            # Update rolling average
            total_jobs = self._stats["jobs_processed"]
            current_avg = self._stats["avg_processing_time_ms"]
            self._stats["avg_processing_time_ms"] = (
                current_avg * (total_jobs - 1) + processing_time_ms
            ) / total_jobs
        else:
            self._stats["jobs_failed"] += 1

        self._stats["queue_length"] = self._job_queue.qsize()

    def get_stats(self) -> dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._stats,
            "queue_utilization": self._job_queue.qsize() / self.max_queue_size,
            "worker_utilization": self._stats["active_workers"] / self.max_workers,
            "success_rate": (
                self._stats["jobs_processed"]
                / (self._stats["jobs_processed"] + self._stats["jobs_failed"])
                if (self._stats["jobs_processed"] + self._stats["jobs_failed"]) > 0
                else 0
            ),
        }


class MemoryOptimizer:
    """Memory optimization utilities for transformation operations."""

    def __init__(self) -> None:
        self._memory_threshold_mb = 500  # Trigger optimization at 500MB
        self._gc_frequency = 100  # Trigger GC every 100 operations
        self._operation_count = 0

    def monitor_memory(self) -> dict[str, float]:
        """Monitor current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
            "percent": process.memory_percent(),
            "available_mb": psutil.virtual_memory().available / 1024 / 1024,
        }

    async def optimize_if_needed(self) -> bool:
        """Optimize memory if usage is high."""
        self._operation_count += 1

        memory_stats = self.monitor_memory()
        needs_optimization = (
            memory_stats["rss_mb"] > self._memory_threshold_mb
            or self._operation_count % self._gc_frequency == 0
        )

        if needs_optimization:
            return await self._perform_optimization(memory_stats)

        return False

    async def _perform_optimization(self, memory_stats: dict[str, float]) -> bool:
        """Perform memory optimization."""
        logger.debug(
            f"Performing memory optimization. Current usage: {memory_stats['rss_mb']:.2f}MB",
        )

        # Run garbage collection
        collected = gc.collect()

        # Clear weak references
        gc.collect()

        # Check if optimization helped
        new_stats = self.monitor_memory()
        freed_mb = memory_stats["rss_mb"] - new_stats["rss_mb"]

        logger.debug(
            f"Memory optimization completed. Freed: {freed_mb:.2f}MB, GC collected: {collected} objects",
        )

        return freed_mb > 0


class OptimizedTransformationEngine(BaseEngine):
    """Optimized transformation engine with performance enhancements.

    Features:
    - Async transformation pipelines
    - Memory-efficient caching with compression
    - Parallel technique execution
    - Resource monitoring and optimization
    - Streaming transformations
    - Advanced performance metrics
    """

    def __init__(
        self,
        enable_cache: bool = True,
        cache_max_memory_mb: int = 200,
        pipeline_workers: int = 8,
        enable_memory_optimization: bool = True,
    ) -> None:
        # Initialize base engine without cache to use our optimized version
        super().__init__(enable_cache=False)

        # Configuration
        self.enable_cache = enable_cache
        self.enable_memory_optimization = enable_memory_optimization

        # Optimized components
        self._compressed_cache = (
            CompressedCache(
                max_memory_mb=cache_max_memory_mb,
                compression_threshold=2000,  # 2KB threshold
                ttl_seconds=7200,  # 2 hours
            )
            if enable_cache
            else None
        )

        self._async_pipeline = AsyncTransformationPipeline(
            max_workers=pipeline_workers,
            max_queue_size=pipeline_workers * 10,
        )

        self._memory_optimizer = MemoryOptimizer() if enable_memory_optimization else None

        # Thread pool for CPU-intensive tasks
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="transform")

        # Performance tracking
        self._metrics = PerformanceMetrics()
        self._request_times: deque = deque(maxlen=1000)

        # Background tasks
        self._monitoring_task: asyncio.Task | None = None
        self._is_running = False

    async def initialize(self) -> None:
        """Initialize the optimized engine."""
        await self._async_pipeline.start()

        # Start performance monitoring
        self._monitoring_task = asyncio.create_task(self._performance_monitor())
        self._is_running = True

        logger.info("OptimizedTransformationEngine initialized")

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._is_running = False

        if self._monitoring_task:
            self._monitoring_task.cancel()

        await self._async_pipeline.stop()
        self._thread_pool.shutdown(wait=True)

        logger.info("OptimizedTransformationEngine shutdown complete")

    async def transform_parallel(
        self,
        prompts: list[str],
        potency_level: int,
        technique_suite: str,
        **kwargs,
    ) -> list[TransformationResult]:
        """Transform multiple prompts in parallel."""
        if not prompts:
            return []

        start_time = time.time()

        # Submit all jobs to pipeline
        job_ids = []
        for i, prompt in enumerate(prompts):
            job = TransformationJob(
                job_id=f"batch_{int(time.time())}_{i}",
                prompt=prompt,
                potency_level=potency_level,
                technique_suite=technique_suite,
                priority=kwargs.get("priority", 0),
            )

            job_id = await self._async_pipeline.submit_job(job, self._transform_single_optimized)
            job_ids.append(job_id)

        # Wait for all results
        results = []
        for job_id in job_ids:
            try:
                result = await self._async_pipeline.get_result(job_id, timeout=60.0)
                results.append(result)
            except Exception as e:
                logger.exception(f"Parallel transformation failed for job {job_id}: {e}")
                # Create error result
                error_result = TransformationResult(
                    original_prompt=prompts[job_ids.index(job_id)],
                    transformed_prompt=prompts[job_ids.index(job_id)],
                    metadata=TransformationMetadata(
                        strategy="error",
                        potency_level=potency_level,
                        technique_suite=technique_suite,
                    ),
                    success=False,
                    error=str(e),
                )
                results.append(error_result)

        # Update metrics
        execution_time_ms = (time.time() - start_time) * 1000
        self._metrics.parallel_efficiency = len(prompts) / (execution_time_ms / 1000)

        logger.info(
            f"Parallel transformation completed: {len(prompts)} prompts in {execution_time_ms:.2f}ms",
        )

        return results

    async def transform(
        self,
        prompt: str,
        potency_level: int,
        technique_suite: str,
        use_cache: bool = True,
        **kwargs,
    ) -> TransformationResult:
        """Optimized single prompt transformation."""
        start_time = time.time()

        try:
            # Memory optimization check
            if self._memory_optimizer:
                await self._memory_optimizer.optimize_if_needed()

            # Check cache first
            cache_key = None
            if self._compressed_cache and use_cache and self.enable_cache:
                cache_key = self._generate_cache_key(prompt, potency_level, technique_suite)
                cached_result = await self._compressed_cache.get(cache_key)
                if cached_result:
                    cached_result.metadata.cached = True
                    self._update_metrics(start_time, cached=True)
                    return cached_result

            # Perform transformation
            result = await self._transform_single_optimized(
                prompt,
                potency_level,
                technique_suite,
                **kwargs,
            )

            # Cache result
            if self._compressed_cache and cache_key and self.enable_cache:
                await self._compressed_cache.set(cache_key, result)

            # Update metrics
            self._update_metrics(start_time, cached=False)
            self._update_technique_metrics(technique_suite, start_time)

            return result

        except Exception as e:
            logger.exception(f"Optimized transformation failed: {e}")
            self._metrics.total_transformations += 1  # Count failed transformations
            raise

    async def _transform_single_optimized(
        self,
        prompt: str,
        potency_level: int,
        technique_suite: str,
        **kwargs,
    ) -> TransformationResult:
        """Optimized single transformation with CPU-intensive work in thread pool."""
        # For CPU-intensive transformations, use thread pool
        cpu_intensive_techniques = {"cipher", "advanced_obfuscation", "quantum_exploit"}

        if technique_suite in cpu_intensive_techniques:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._thread_pool,
                lambda: asyncio.run(
                    super(OptimizedTransformationEngine, self).transform(
                        prompt,
                        potency_level,
                        technique_suite,
                        use_cache=False,
                        **kwargs,
                    ),
                ),
            )
        # Use base implementation for other techniques
        return await super().transform(
            prompt,
            potency_level,
            technique_suite,
            use_cache=False,
            **kwargs,
        )

    async def transform_stream_optimized(
        self,
        prompt: str,
        potency_level: int,
        technique_suite: str,
        buffer_size: int = 8192,
    ) -> AsyncIterator[StreamChunk]:
        """Optimized streaming transformation with buffering."""
        buffer = ""

        async for chunk in self.transform_stream(prompt, potency_level, technique_suite):
            buffer += chunk.text

            # Yield buffered chunks for better network efficiency
            while len(buffer) >= buffer_size:
                yield_text = buffer[:buffer_size]
                buffer = buffer[buffer_size:]

                yield StreamChunk(
                    text=yield_text,
                    is_final=False,
                    finish_reason=None,
                )

            # Yield final chunk if complete
            if chunk.is_final and buffer:
                yield StreamChunk(
                    text=buffer,
                    is_final=True,
                    finish_reason=chunk.finish_reason,
                )
                break

    def _generate_cache_key(self, prompt: str, potency: int, technique: str) -> str:
        """Generate cache key for compressed cache."""
        import hashlib

        key_data = f"{prompt[:200]}:{potency}:{technique}"  # Limit prompt size in key
        return hashlib.sha256(key_data.encode()).hexdigest()

    def _update_metrics(self, start_time: float, cached: bool = False) -> None:
        """Update performance metrics."""
        execution_time_ms = (time.time() - start_time) * 1000
        self._request_times.append(execution_time_ms)

        self._metrics.total_transformations += 1

        if not cached:
            # Update rolling average execution time
            total = self._metrics.total_transformations
            current_avg = self._metrics.avg_execution_time_ms
            self._metrics.avg_execution_time_ms = (
                current_avg * (total - 1) + execution_time_ms
            ) / total

    def _update_technique_metrics(self, technique: str, start_time: float) -> None:
        """Update technique-specific performance metrics."""
        execution_time_ms = (time.time() - start_time) * 1000

        if technique not in self._metrics.technique_performance:
            self._metrics.technique_performance[technique] = execution_time_ms
        else:
            # Rolling average
            current = self._metrics.technique_performance[technique]
            self._metrics.technique_performance[technique] = (current + execution_time_ms) / 2

    async def _performance_monitor(self) -> None:
        """Background performance monitoring."""
        while self._is_running:
            try:
                await asyncio.sleep(30)  # Monitor every 30 seconds

                # Update memory metrics
                if self._memory_optimizer:
                    memory_stats = self._memory_optimizer.monitor_memory()
                    self._metrics.memory_usage_mb = memory_stats["rss_mb"]
                    self._metrics.memory_peak_mb = max(
                        self._metrics.memory_peak_mb,
                        memory_stats["rss_mb"],
                    )

                # Update cache hit rate
                if self._compressed_cache:
                    cache_stats = self._compressed_cache.get_stats()
                    self._metrics.cache_hit_rate = cache_stats["hit_rate"]

                # Log performance summary
                logger.info(
                    f"Transform Engine Metrics - "
                    f"Transformations: {self._metrics.total_transformations}, "
                    f"Avg time: {self._metrics.avg_execution_time_ms:.2f}ms, "
                    f"Memory: {self._metrics.memory_usage_mb:.2f}MB, "
                    f"Cache hit rate: {self._metrics.cache_hit_rate:.2%}",
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Performance monitoring error: {e}")

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get comprehensive performance metrics."""
        cache_stats = self._compressed_cache.get_stats() if self._compressed_cache else {}
        pipeline_stats = self._async_pipeline.get_stats()
        memory_stats = self._memory_optimizer.monitor_memory() if self._memory_optimizer else {}

        return {
            "engine": {
                "total_transformations": self._metrics.total_transformations,
                "avg_execution_time_ms": self._metrics.avg_execution_time_ms,
                "memory_usage_mb": self._metrics.memory_usage_mb,
                "memory_peak_mb": self._metrics.memory_peak_mb,
                "cache_hit_rate": self._metrics.cache_hit_rate,
                "parallel_efficiency": self._metrics.parallel_efficiency,
                "technique_performance": self._metrics.technique_performance,
            },
            "cache": cache_stats,
            "pipeline": pipeline_stats,
            "memory": memory_stats,
            "recent_execution_times_ms": list(self._request_times)[-10:],
        }

    async def clear_all_caches(self) -> None:
        """Clear all caches and free memory."""
        if self._compressed_cache:
            await self._compressed_cache.clear()

        if self._memory_optimizer:
            await self._memory_optimizer._perform_optimization(
                self._memory_optimizer.monitor_memory(),
            )

        logger.info("All caches cleared and memory optimized")

    async def optimize_for_memory_constrained(self) -> None:
        """Optimize for memory-constrained environments."""
        if self._compressed_cache:
            # Reduce cache size
            self._compressed_cache.max_memory_mb = 50
            self._compressed_cache.compression_threshold = 500  # More aggressive compression

        # Reduce pipeline workers
        await self._async_pipeline.stop()
        self._async_pipeline = AsyncTransformationPipeline(
            max_workers=4,
            max_queue_size=20,
        )
        await self._async_pipeline.start()

        # Reduce thread pool size
        self._thread_pool.shutdown(wait=True)
        self._thread_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="transform")

        logger.info("Engine optimized for memory-constrained environment")


# Global optimized transformation engine
optimized_transformation_engine = OptimizedTransformationEngine(
    enable_cache=getattr(getattr(config, "transformation", None), "enable_cache", True),
    cache_max_memory_mb=200,
    pipeline_workers=8,
    enable_memory_optimization=True,
)
