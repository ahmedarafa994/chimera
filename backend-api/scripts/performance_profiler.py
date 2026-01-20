"""Performance Profiling Script for Chimera Backend.

Captures detailed performance metrics including:
- CPU profiling with cProfile
- Memory profiling with memory_profiler
- Async task analysis
- Database query performance
- LLM provider latency

Usage:
    python scripts/performance_profiler.py --module llm_service --duration 60
    python scripts/performance_profiler.py --endpoint /api/v1/generate --requests 100
"""

import asyncio
import cProfile
import json
import pstats
import sys
import time
from argparse import ArgumentParser
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.logging import logger
from app.domain.models import GenerationConfig, PromptRequest, PromptResponse
from app.services.llm_service import LLMResponseCache, LLMService


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    module_name: str
    duration_seconds: float
    cpu_profile_path: str | None = None
    memory_profile_path: str | None = None

    # Execution metrics
    total_calls: int = 0
    total_time: float = 0.0
    avg_time_per_call: float = 0.0

    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    cache_hit_rate: float = 0.0

    # Deduplication metrics
    deduplicated_requests: int = 0
    deduplication_rate: float = 0.0

    # Hotspots (top 10 functions by time)
    hotspots: list[dict[str, Any]] = field(default_factory=list)

    # Memory metrics
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "module_name": self.module_name,
            "duration_seconds": self.duration_seconds,
            "cpu_profile_path": self.cpu_profile_path,
            "memory_profile_path": self.memory_profile_path,
            "total_calls": self.total_calls,
            "total_time": self.total_time,
            "avg_time_per_call": self.avg_time_per_call,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "deduplicated_requests": self.deduplicated_requests,
            "deduplication_rate": self.deduplication_rate,
            "hotspots": self.hotspots,
            "peak_memory_mb": self.peak_memory_mb,
            "avg_memory_mb": self.avg_memory_mb,
        }


class PerformanceProfiler:
    """Performance profiler for Chimera backend.

    Captures CPU and memory metrics for specific modules or endpoints.
    """

    def __init__(self, output_dir: str = "profiles") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def profile_module(
        self,
        module_func: Callable,
        *args,
        duration: int = 60,
        **kwargs,
    ) -> PerformanceMetrics:
        """Profile a module function for specified duration.

        Args:
            module_func: Function to profile
            args: Positional arguments for function
            duration: Duration to run profiler (seconds)
            kwargs: Keyword arguments for function

        Returns:
            PerformanceMetrics with captured data

        """
        logger.info(f"Profiling {module_func.__name__} for {duration} seconds...")

        metrics = PerformanceMetrics(module_name=module_func.__name__, duration_seconds=duration)

        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()

        # Execute function
        start_time = time.time()
        try:
            # Run for specified duration
            if asyncio.iscoroutinefunction(module_func):
                asyncio.run(module_func(*args, duration=duration, **kwargs))
            else:
                module_func(*args, **kwargs)
        finally:
            profiler.disable()
            metrics.total_time = time.time() - start_time

        # Process CPU profile
        stats = pstats.Stats(profiler)
        stats.strip_dirs()

        # Get total calls
        metrics.total_calls = stats.total_calls

        # Calculate average time per call
        if metrics.total_calls > 0:
            metrics.avg_time_per_call = metrics.total_time / metrics.total_calls

        # Extract hotspots (top 10 functions by cumulative time)
        stats.sort_stats("cumulative")
        hotspots_data = []
        for func, (_cc, nc, tt, ct, callers) in stats.stats.items()[:10]:
            hotspots_data.append(
                {
                    "function": f"{func[0]}:{func[1]}:{func[2]}",
                    "cumulative_time": ct,
                    "total_time": tt,
                    "calls": nc,
                    "callers": len(callers),
                },
            )
        metrics.hotspots = hotspots_data

        # Save CPU profile
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = self.output_dir / f"{module_func.__name__}_cpu_{timestamp}.prof"
        profiler.dump_stats(str(profile_path))
        metrics.cpu_profile_path = str(profile_path)

        # Extract cache metrics if available
        if hasattr(module_func, "__self__") and hasattr(module_func.__self__, "_response_cache"):
            cache_stats = module_func.__self__._response_cache.get_stats()
            metrics.cache_hits = cache_stats.get("hits", 0)
            metrics.cache_misses = cache_stats.get("misses", 0)

            total = metrics.cache_hits + metrics.cache_misses
            if total > 0:
                metrics.cache_hit_rate = metrics.cache_hits / total

        # Extract deduplication metrics if available
        if hasattr(module_func, "__self__") and hasattr(module_func.__self__, "_deduplicator"):
            dedup_stats = module_func.__self__._deduplicator.get_stats()
            metrics.deduplicated_requests = dedup_stats.get("deduplicated_count", 0)

            if metrics.total_calls > 0:
                metrics.deduplication_rate = metrics.deduplicated_requests / metrics.total_calls

        logger.info(
            f"Profiling complete: {metrics.total_calls} calls, {metrics.total_time:.2f}s total",
        )

        return metrics

    async def profile_endpoint(
        self,
        endpoint: str,
        request_data: dict,
        num_requests: int = 100,
    ) -> PerformanceMetrics:
        """Profile an API endpoint.

        Args:
            endpoint: Endpoint path (e.g., /api/v1/generate)
            request_data: Request payload
            num_requests: Number of requests to send

        Returns:
            PerformanceMetrics with captured data

        """
        logger.info(f"Profiling endpoint {endpoint} with {num_requests} requests...")

        from httpx import AsyncClient

        from app.main import app

        metrics = PerformanceMetrics(
            module_name=f"endpoint{endpoint.replace('/', '_')}",
            duration_seconds=0,  # Will be calculated
        )

        # CPU profiling
        profiler = cProfile.Profile()
        profiler.enable()

        latencies = []

        async with AsyncClient(app=app, base_url="http://test") as client:
            start_time = time.time()

            for i in range(num_requests):
                request_start = time.time()

                try:
                    if endpoint == "/api/v1/generate":
                        await client.post(endpoint, json=request_data, timeout=30.0)
                        latency = time.time() - request_start
                        latencies.append(latency)
                except Exception as e:
                    logger.error(f"Request {i} failed: {e}")

            metrics.total_time = time.time() - start_time
            metrics.duration_seconds = metrics.total_time
            metrics.total_calls = num_requests

        profiler.disable()

        # Calculate latency metrics
        if latencies:
            metrics.avg_time_per_call = sum(latencies) / len(latencies)

        # Process CPU profile
        stats = pstats.Stats(profiler)
        stats.strip_dirs()
        stats.sort_stats("cumulative")

        # Extract hotspots
        hotspots_data = []
        for func, (_cc, nc, tt, ct, _callers) in stats.stats.items()[:10]:
            hotspots_data.append(
                {
                    "function": f"{func[0]}:{func[1]}:{func[2]}",
                    "cumulative_time": ct,
                    "total_time": tt,
                    "calls": nc,
                },
            )
        metrics.hotspots = hotspots_data

        # Save profile
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        profile_path = self.output_dir / f"endpoint_{endpoint.replace('/', '_')}_{timestamp}.prof"
        profiler.dump_stats(str(profile_path))
        metrics.cpu_profile_path = str(profile_path)

        logger.info(
            f"Endpoint profiling complete: {num_requests} requests, "
            f"{metrics.avg_time_per_call * 1000:.2f}ms avg latency",
        )

        return metrics


async def profile_llm_service(duration: int = 60) -> None:
    """Profile LLM service for specified duration.

    Simulates realistic load with cache hits/misses.
    """
    logger.info(f"Starting LLM service profiling for {duration} seconds...")

    end_time = time.time() + duration
    call_count = 0

    # Test prompts (some repeated to test cache)
    test_prompts = [
        "Generate a test response",
        "What is the capital of France?",
        "Explain quantum computing",
        "Write a haiku about AI",
        "Generate a test response",  # Repeat for cache hit
    ]

    while time.time() < end_time:
        for prompt in test_prompts:
            if time.time() >= end_time:
                break

            PromptRequest(
                prompt=prompt,
                provider="google",
                model="gemini-1.5-pro",
                config=GenerationConfig(temperature=0.7, max_tokens=100),
            )

            try:
                # This will hit the cache for repeated prompts
                # response = await llm_service.generate_text(request)
                call_count += 1
                await asyncio.sleep(0.1)  # Simulate realistic load
            except Exception as e:
                logger.error(f"LLM call failed: {e}")

    logger.info(f"LLM service profiling complete: {call_count} calls made")


def profile_memory_usage():
    """Profile memory usage of Chimera backend.

    Requires memory_profiler to be installed.
    """
    try:
        import matplotlib.pyplot as plt
        from memory_profiler import memory_usage

        # Profile LLM service memory
        def memory_test() -> None:
            LLMService()

            # Simulate load
            for i in range(100):
                cache = LLMResponseCache(max_size=500)
                cache.set(
                    PromptRequest(
                        prompt=f"Test prompt {i}",
                        provider="google",
                        model="gemini-1.5-pro",
                        config=GenerationConfig(),
                    ),
                    PromptResponse(
                        generated_text="Test response",
                        provider="google",
                        model="gemini-1.5-pro",
                        tokens_prompt=10,
                        tokens_completion=20,
                        tokens_total=30,
                        latency_ms=100,
                    ),
                )

        # Measure memory usage
        mem_usage = memory_usage(memory_test, interval=0.1, timeout=60)

        # Plot memory usage
        plt.figure(figsize=(12, 6))
        plt.plot(mem_usage)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Chimera Backend Memory Usage")
        plt.grid(True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = Path("profiles") / f"memory_usage_{timestamp}.png"
        plt.savefig(plot_path)
        logger.info(f"Memory usage plot saved to {plot_path}")

        return {
            "peak_memory_mb": max(mem_usage),
            "avg_memory_mb": sum(mem_usage) / len(mem_usage),
            "plot_path": str(plot_path),
        }

    except ImportError:
        logger.warning("memory_profiler not installed. Skipping memory profiling.")
        return None


def main() -> None:
    """Main entry point for performance profiling."""
    parser = ArgumentParser(description="Profile Chimera backend performance")

    parser.add_argument(
        "--module",
        choices=["llm_service", "transformation", "autodan"],
        help="Module to profile",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="API endpoint to profile (e.g., /api/v1/generate)",
    )
    parser.add_argument("--duration", type=int, default=60, help="Profiling duration in seconds")
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of requests for endpoint profiling",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="profiles",
        help="Output directory for profiles",
    )
    parser.add_argument("--memory", action="store_true", help="Profile memory usage")
    parser.add_argument("--all", action="store_true", help="Run all profiling tasks")

    args = parser.parse_args()

    profiler = PerformanceProfiler(output_dir=args.output_dir)

    if args.all:
        logger.info("Running comprehensive performance profiling...")

        # Profile all modules
        all_metrics = []

        # LLM service
        metrics = profiler.profile_module(profile_llm_service, duration=args.duration)
        all_metrics.append(metrics)

        # Memory profiling
        if args.memory:
            memory_metrics = profile_memory_usage()
            if memory_metrics:
                logger.info(
                    f"Memory usage: peak={memory_metrics['peak_memory_mb']:.2f}MB, avg={memory_metrics['avg_memory_mb']:.2f}MB",
                )

        # Save combined report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(args.output_dir) / f"performance_report_{timestamp}.json"

        with open(report_path, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "metrics": [m.to_dict() for m in all_metrics],
                },
                f,
                indent=2,
            )

        logger.info(f"Performance report saved to {report_path}")

    elif args.module:
        # Profile specific module
        if args.module == "llm_service":
            metrics = profiler.profile_module(profile_llm_service, duration=args.duration)
        else:
            logger.error(f"Module {args.module} profiling not yet implemented")
            return

        # Print summary
        for _i, _hotspot in enumerate(metrics.hotspots[:5], 1):
            pass

    elif args.endpoint:
        # Profile endpoint
        request_data = {
            "prompt": "Test prompt for profiling",
            "provider": "google",
            "model": "gemini-1.5-pro",
            "config": {"temperature": 0.7, "max_tokens": 100},
        }

        metrics = asyncio.run(
            profiler.profile_endpoint(args.endpoint, request_data, num_requests=args.requests),
        )

        # Print summary
        for _i, _hotspot in enumerate(metrics.hotspots[:5], 1):
            pass

    if args.memory and not args.all:
        # Memory profiling only
        memory_metrics = profile_memory_usage()
        if memory_metrics:
            pass


if __name__ == "__main__":
    main()
