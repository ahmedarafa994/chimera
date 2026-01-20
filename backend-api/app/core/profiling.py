"""Advanced profiling utilities for Chimera backend
Provides CPU profiling, memory profiling, and flame graph generation.
"""

import asyncio
import cProfile
import json
import os
import pstats
import subprocess
import time
import tracemalloc
from collections.abc import Callable
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from functools import wraps

import psutil


class CPUProfiler:
    """CPU profiling with flame graph generation."""

    def __init__(self, output_dir: str = "performance/profiles") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.profiler = None
        self.active_profiles = {}

    @contextmanager
    def profile_context(self, name: str):
        """Context manager for CPU profiling."""
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()

        try:
            yield profiler
        finally:
            profiler.disable()
            duration = time.time() - start_time

            # Save profile data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/cpu_profile_{name}_{timestamp}.prof"
            profiler.dump_stats(filename)

            # Generate text report
            self._generate_text_report(filename, name, duration)

            # Generate flame graph
            self._generate_flame_graph(filename, name)

    def _generate_text_report(self, profile_file: str, name: str, duration: float) -> None:
        """Generate human-readable profile report."""
        stats = pstats.Stats(profile_file)
        stats.sort_stats("cumulative")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.output_dir}/cpu_report_{name}_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write(f"CPU Profile Report: {name}\n")
            f.write(f"Duration: {duration:.2f} seconds\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            # Top functions by cumulative time
            f.write("TOP 20 FUNCTIONS BY CUMULATIVE TIME:\n")
            f.write("-" * 40 + "\n")
            stats.print_stats(20, file=f)

            f.write("\n\nTOP 20 FUNCTIONS BY INTERNAL TIME:\n")
            f.write("-" * 40 + "\n")
            stats.sort_stats("time")
            stats.print_stats(20, file=f)

    def _generate_flame_graph(self, profile_file: str, name: str) -> None:
        """Generate flame graph from profile data."""
        try:
            # Convert profile to flame graph format
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            flamegraph_file = f"{self.output_dir}/flamegraph_{name}_{timestamp}.svg"

            # Use py-spy for flame graph generation
            cmd = ["py-spy", "flamegraph", "--file", profile_file, "--output", flamegraph_file]

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
            )  # - hardcoded safe command

        except subprocess.CalledProcessError:
            pass
        except FileNotFoundError:
            pass

    def profile_async_function(self, func: Callable, name: str | None = None):
        """Decorator for profiling async functions."""
        if name is None:
            name = func.__name__

        @wraps(func)
        async def wrapper(*args, **kwargs):
            with self.profile_context(f"async_{name}"):
                return await func(*args, **kwargs)

        return wrapper

    def profile_sync_function(self, func: Callable, name: str | None = None):
        """Decorator for profiling sync functions."""
        if name is None:
            name = func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_context(f"sync_{name}"):
                return func(*args, **kwargs)

        return wrapper


class MemoryProfiler:
    """Memory profiling and leak detection."""

    def __init__(self, output_dir: str = "performance/profiles") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.snapshots = {}
        self.monitoring = False

    def start_monitoring(self) -> None:
        """Start memory monitoring."""
        if not self.monitoring:
            tracemalloc.start()
            self.monitoring = True
            self.baseline_snapshot = tracemalloc.take_snapshot()

    def stop_monitoring(self) -> None:
        """Stop memory monitoring."""
        if self.monitoring:
            tracemalloc.stop()
            self.monitoring = False

    def take_snapshot(self, name: str) -> None:
        """Take a memory snapshot."""
        if not self.monitoring:
            self.start_monitoring()

        snapshot = tracemalloc.take_snapshot()
        self.snapshots[name] = {
            "snapshot": snapshot,
            "timestamp": datetime.now(),
            "process_memory": psutil.Process().memory_info(),
        }

        # Generate memory report
        self._generate_memory_report(name)

    def _generate_memory_report(self, name: str) -> None:
        """Generate detailed memory report."""
        if name not in self.snapshots:
            return

        snapshot_data = self.snapshots[name]
        snapshot = snapshot_data["snapshot"]
        timestamp = snapshot_data["timestamp"].strftime("%Y%m%d_%H%M%S")

        report_file = f"{self.output_dir}/memory_report_{name}_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write(f"Memory Profile Report: {name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n\n")

            # Process memory info
            mem_info = snapshot_data["process_memory"]
            f.write("Process Memory Usage:\n")
            f.write(f"  RSS: {mem_info.rss / 1024 / 1024:.2f} MB\n")
            f.write(f"  VMS: {mem_info.vms / 1024 / 1024:.2f} MB\n")
            f.write(f"  Percent: {psutil.Process().memory_percent():.2f}%\n\n")

            # Top memory consumers
            f.write("TOP 20 MEMORY CONSUMERS:\n")
            f.write("-" * 40 + "\n")
            top_stats = snapshot.statistics("lineno")
            f.writelines(f"{index:2d}. {stat}\n" for index, stat in enumerate(top_stats[:20], 1))

            # Traceback for top consumers
            f.write("\n\nTOP 5 TRACEBACKS:\n")
            f.write("-" * 40 + "\n")
            for stat in top_stats[:5]:
                f.write(f"\n{stat}:\n")
                f.writelines(f"  {line}" for line in stat.traceback.format())

    def compare_snapshots(self, name1: str, name2: str) -> None:
        """Compare two memory snapshots."""
        if name1 not in self.snapshots or name2 not in self.snapshots:
            return

        snapshot1 = self.snapshots[name1]["snapshot"]
        snapshot2 = self.snapshots[name2]["snapshot"]

        top_stats = snapshot2.compare_to(snapshot1, "lineno")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.output_dir}/memory_diff_{name1}_to_{name2}_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write(f"Memory Comparison: {name1} → {name2}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("=" * 80 + "\n\n")

            f.write("TOP 20 MEMORY CHANGES:\n")
            f.write("-" * 40 + "\n")
            f.writelines(f"{stat}\n" for stat in top_stats[:20])

    @contextmanager
    def profile_memory_context(self, name: str):
        """Context manager for memory profiling."""
        self.start_monitoring()
        self.take_snapshot(f"{name}_start")

        try:
            yield
        finally:
            self.take_snapshot(f"{name}_end")
            self.compare_snapshots(f"{name}_start", f"{name}_end")


class IOProfiler:
    """I/O operations profiling."""

    def __init__(self, output_dir: str = "performance/profiles") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.io_stats = []

    @asynccontextmanager
    async def profile_io_context(self, name: str):
        """Context manager for I/O profiling."""
        start_time = time.time()
        process = psutil.Process()
        start_io = process.io_counters()

        try:
            yield
        finally:
            end_time = time.time()
            end_io = process.io_counters()

            stats = {
                "name": name,
                "duration": end_time - start_time,
                "read_bytes": end_io.read_bytes - start_io.read_bytes,
                "write_bytes": end_io.write_bytes - start_io.write_bytes,
                "read_count": end_io.read_count - start_io.read_count,
                "write_count": end_io.write_count - start_io.write_count,
                "timestamp": datetime.now().isoformat(),
            }

            self.io_stats.append(stats)
            self._save_io_stats()

    def _save_io_stats(self) -> None:
        """Save I/O statistics to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_file = f"{self.output_dir}/io_stats_{timestamp}.json"

        with open(stats_file, "w") as f:
            json.dump(self.io_stats, f, indent=2)


class HotPathAnalyzer:
    """Analyze hot paths and performance bottlenecks."""

    def __init__(self, output_dir: str = "performance/analysis") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.call_stats = {}
        self.execution_times = {}

    def record_call(self, function_name: str, duration: float, args_info: str = "") -> None:
        """Record function call for analysis."""
        if function_name not in self.call_stats:
            self.call_stats[function_name] = {
                "count": 0,
                "total_time": 0,
                "min_time": float("inf"),
                "max_time": 0,
                "avg_time": 0,
                "recent_calls": [],
            }

        stats = self.call_stats[function_name]
        stats["count"] += 1
        stats["total_time"] += duration
        stats["min_time"] = min(stats["min_time"], duration)
        stats["max_time"] = max(stats["max_time"], duration)
        stats["avg_time"] = stats["total_time"] / stats["count"]

        # Keep recent calls for trend analysis
        stats["recent_calls"].append(
            {"duration": duration, "timestamp": time.time(), "args_info": args_info},
        )

        # Keep only last 100 calls
        if len(stats["recent_calls"]) > 100:
            stats["recent_calls"].pop(0)

    def generate_hotpath_report(self) -> None:
        """Generate hot path analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.output_dir}/hotpath_analysis_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write("HOT PATH ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            # Sort by total time
            sorted_functions = sorted(
                self.call_stats.items(),
                key=lambda x: x[1]["total_time"],
                reverse=True,
            )

            f.write("TOP FUNCTIONS BY TOTAL TIME:\n")
            f.write("-" * 50 + "\n")
            for func_name, stats in sorted_functions[:20]:
                f.write(f"{func_name}:\n")
                f.write(f"  Calls: {stats['count']}\n")
                f.write(f"  Total Time: {stats['total_time']:.3f}s\n")
                f.write(f"  Avg Time: {stats['avg_time']:.3f}s\n")
                f.write(f"  Min Time: {stats['min_time']:.3f}s\n")
                f.write(f"  Max Time: {stats['max_time']:.3f}s\n\n")

            # Sort by average time
            sorted_by_avg = sorted(
                self.call_stats.items(),
                key=lambda x: x[1]["avg_time"],
                reverse=True,
            )

            f.write("\nTOP FUNCTIONS BY AVERAGE TIME:\n")
            f.write("-" * 50 + "\n")
            for func_name, stats in sorted_by_avg[:20]:
                if stats["count"] >= 5:  # Only include functions called at least 5 times
                    f.write(f"{func_name}: {stats['avg_time']:.3f}s avg ({stats['count']} calls)\n")

    def profile_function(self, func: Callable, name: str | None = None):
        """Decorator for hot path analysis."""
        if name is None:
            name = f"{func.__module__}.{func.__name__}"

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    duration = time.time() - start_time
                    args_info = f"args={len(args)}, kwargs={len(kwargs)}"
                    self.record_call(name, duration, args_info)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                args_info = f"args={len(args)}, kwargs={len(kwargs)}"
                self.record_call(name, duration, args_info)

        return sync_wrapper


class PerformanceProfiler:
    """Main profiler orchestrator."""

    def __init__(self, output_dir: str = "performance") -> None:
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.cpu_profiler = CPUProfiler(f"{output_dir}/cpu")
        self.memory_profiler = MemoryProfiler(f"{output_dir}/memory")
        self.io_profiler = IOProfiler(f"{output_dir}/io")
        self.hotpath_analyzer = HotPathAnalyzer(f"{output_dir}/hotpaths")

    @asynccontextmanager
    async def profile_all_context(self, name: str):
        """Profile CPU, memory, and I/O for a code block."""
        with (
            self.cpu_profiler.profile_context(name),
            self.memory_profiler.profile_memory_context(name),
        ):
            async with self.io_profiler.profile_io_context(name):
                yield

    def generate_comprehensive_report(self) -> None:
        """Generate comprehensive performance report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.output_dir}/comprehensive_report_{timestamp}.txt"

        with open(report_file, "w") as f:
            f.write("COMPREHENSIVE PERFORMANCE REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")

            # System information
            process = psutil.Process()
            f.write("SYSTEM INFORMATION:\n")
            f.write(f"  CPU Count: {psutil.cpu_count()}\n")
            f.write(f"  Memory Total: {psutil.virtual_memory().total / 1024**3:.2f} GB\n")
            f.write(f"  Memory Available: {psutil.virtual_memory().available / 1024**3:.2f} GB\n")
            f.write(f"  Process Memory: {process.memory_info().rss / 1024**2:.2f} MB\n")
            f.write(f"  Process CPU: {process.cpu_percent():.2f}%\n\n")

        # Generate individual reports
        self.hotpath_analyzer.generate_hotpath_report()


# Global profiler instance
global_profiler = PerformanceProfiler()

# Decorators for easy use
profile_cpu = global_profiler.cpu_profiler.profile_sync_function
profile_cpu_async = global_profiler.cpu_profiler.profile_async_function
profile_hotpath = global_profiler.hotpath_analyzer.profile_function
