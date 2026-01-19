"""
Comprehensive CPU Profiling with Flame Graph Generation
Profiles FastAPI backend, LLM services, and transformation engines
"""

import cProfile
import json
import os
import pstats
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import psutil
from profiling_config import MetricType, config


@dataclass
class CPUProfileResult:
    """CPU profiling result data"""

    profile_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    total_calls: int
    total_time: float
    flame_graph_path: str | None
    hotspots: list[dict[str, Any]]
    cpu_usage_avg: float
    cpu_usage_max: float
    process_stats: dict[str, Any]


class CPUProfiler:
    """Comprehensive CPU profiling for Chimera system"""

    def __init__(self):
        self.active_profiles: dict[str, cProfile.Profile] = {}
        self.process_monitors: dict[str, threading.Thread] = {}
        self.cpu_usage_data: dict[str, list[float]] = {}

    @contextmanager
    def profile_context(self, profile_name: str):
        """Context manager for CPU profiling"""
        profile_id = f"{profile_name}_{int(time.time())}"

        try:
            self.start_profiling(profile_id)
            yield profile_id
        finally:
            self.stop_profiling(profile_id)

    def start_profiling(self, profile_id: str) -> None:
        """Start CPU profiling for a specific profile"""
        if not config.is_metric_enabled(MetricType.CPU):
            return

        # Start cProfile
        profiler = cProfile.Profile()
        profiler.enable()
        self.active_profiles[profile_id] = profiler

        # Start CPU usage monitoring
        self.cpu_usage_data[profile_id] = []
        monitor_thread = threading.Thread(
            target=self._monitor_cpu_usage, args=(profile_id,), daemon=True
        )
        monitor_thread.start()
        self.process_monitors[profile_id] = monitor_thread

        print(f"Started CPU profiling: {profile_id}")

    def stop_profiling(self, profile_id: str) -> CPUProfileResult | None:
        """Stop CPU profiling and generate results"""
        if profile_id not in self.active_profiles:
            return None

        # Stop profiler
        profiler = self.active_profiles.pop(profile_id)
        profiler.disable()

        # Stop CPU monitoring
        if profile_id in self.process_monitors:
            # Signal to stop monitoring (we'll use a simple approach)
            pass

        # Generate profile stats
        stats_output = config.get_output_path(MetricType.CPU, f"{profile_id}_stats.txt")
        with open(stats_output, "w") as f:
            stats = pstats.Stats(profiler, stream=f)
            stats.sort_stats("cumulative")
            stats.print_stats(50)  # Top 50 functions

        # Generate flame graph
        flame_graph_path = self._generate_flame_graph(profiler, profile_id)

        # Analyze hotspots
        hotspots = self._analyze_hotspots(profiler)

        # Get CPU usage statistics
        cpu_data = self.cpu_usage_data.get(profile_id, [])
        cpu_usage_avg = sum(cpu_data) / len(cpu_data) if cpu_data else 0
        cpu_usage_max = max(cpu_data) if cpu_data else 0

        # Get process statistics
        process_stats = self._get_process_stats()

        result = CPUProfileResult(
            profile_id=profile_id,
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC),
            duration_seconds=0.0,  # Will be calculated
            total_calls=stats.total_calls if "stats" in locals() else 0,
            total_time=stats.total_tt if "stats" in locals() else 0.0,
            flame_graph_path=flame_graph_path,
            hotspots=hotspots,
            cpu_usage_avg=cpu_usage_avg,
            cpu_usage_max=cpu_usage_max,
            process_stats=process_stats,
        )

        # Save detailed results
        self._save_profile_results(result)

        print(f"Completed CPU profiling: {profile_id}")
        return result

    def _monitor_cpu_usage(self, profile_id: str) -> None:
        """Monitor CPU usage during profiling"""
        process = psutil.Process()

        while profile_id in self.active_profiles:
            try:
                cpu_percent = process.cpu_percent(interval=1.0)
                self.cpu_usage_data[profile_id].append(cpu_percent)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
            except Exception as e:
                print(f"Error monitoring CPU: {e}")
                break

    def _generate_flame_graph(self, profiler: cProfile.Profile, profile_id: str) -> str | None:
        """Generate flame graph from cProfile data"""
        try:
            # Save profile data in a format suitable for flame graph generation
            profile_data_path = config.get_output_path(MetricType.CPU, f"{profile_id}.prof")
            profiler.dump_stats(profile_data_path)

            # Convert to flame graph format using py-spy or similar tool
            flame_graph_path = config.get_output_path(MetricType.CPU, f"{profile_id}_flame.svg")

            # Use flameprof to convert cProfile to flame graph
            try:
                subprocess.run(
                    [
                        "flameprof",
                        profile_data_path,
                        "--format",
                        "svg",
                        "--output",
                        flame_graph_path,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )

                return flame_graph_path
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Fallback to py-spy if available
                return self._generate_pyspy_flame_graph(profile_id)

        except Exception as e:
            print(f"Error generating flame graph: {e}")
            return None

    def _generate_pyspy_flame_graph(self, profile_id: str) -> str | None:
        """Generate flame graph using py-spy"""
        try:
            flame_graph_path = config.get_output_path(
                MetricType.CPU, f"{profile_id}_pyspy_flame.svg"
            )

            # Get current process PID
            current_pid = os.getpid()

            # Run py-spy for a short duration to capture current state
            subprocess.run(
                [
                    "py-spy",
                    "record",
                    "--pid",
                    str(current_pid),
                    "--duration",
                    "10",
                    "--format",
                    "flamegraph",
                    "--output",
                    flame_graph_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            return flame_graph_path
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("py-spy not available for flame graph generation")
            return None

    def _analyze_hotspots(self, profiler: cProfile.Profile) -> list[dict[str, Any]]:
        """Analyze performance hotspots from profile data"""
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")

        hotspots = []

        # Get top 20 functions by cumulative time
        items = list(stats.stats.items())[:20]
        for func_info, (cc, _nc, tt, ct, _callers) in items:
            filename, lineno, func_name = func_info

            hotspot = {
                "function": func_name,
                "filename": filename,
                "line_number": lineno,
                "call_count": cc,
                "total_time": tt,
                "cumulative_time": ct,
                "per_call_time": ct / cc if cc > 0 else 0,
                "percentage_of_total": ((ct / stats.total_tt * 100) if stats.total_tt > 0 else 0),
            }

            hotspots.append(hotspot)

        return hotspots

    def _get_process_stats(self) -> dict[str, Any]:
        """Get current process statistics"""
        try:
            process = psutil.Process()

            return {
                "pid": process.pid,
                "cpu_percent": process.cpu_percent(),
                "memory_info": process.memory_info()._asdict(),
                "memory_percent": process.memory_percent(),
                "num_threads": process.num_threads(),
                "num_fds": (process.num_fds() if hasattr(process, "num_fds") else None),
                "create_time": process.create_time(),
                "cpu_times": process.cpu_times()._asdict(),
            }
        except Exception as e:
            print(f"Error getting process stats: {e}")
            return {}

    def _save_profile_results(self, result: CPUProfileResult) -> None:
        """Save profiling results to JSON file"""
        output_path = config.get_output_path(MetricType.CPU, f"{result.profile_id}_results.json")

        result_dict = {
            "profile_id": result.profile_id,
            "start_time": result.start_time.isoformat(),
            "end_time": result.end_time.isoformat(),
            "duration_seconds": result.duration_seconds,
            "total_calls": result.total_calls,
            "total_time": result.total_time,
            "flame_graph_path": result.flame_graph_path,
            "hotspots": result.hotspots,
            "cpu_usage_avg": result.cpu_usage_avg,
            "cpu_usage_max": result.cpu_usage_max,
            "process_stats": result.process_stats,
        }

        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2)

    async def profile_async_function(self, func: Callable, *args, **kwargs) -> tuple:
        """Profile an async function"""
        profile_id = f"async_{func.__name__}_{int(time.time())}"

        with self.profile_context(profile_id):
            result = await func(*args, **kwargs)

        return result, profile_id

    def profile_endpoint_performance(
        self, endpoint_url: str, method: str = "GET", duration: int = 60
    ) -> CPUProfileResult:
        """Profile performance during endpoint load testing"""
        import requests

        profile_id = f"endpoint_{method}_{endpoint_url.replace('/', '_')}_" f"{int(time.time())}"

        def make_requests():
            """Make continuous requests to endpoint"""
            session = requests.Session()
            start_time = time.time()

            while time.time() - start_time < duration:
                try:
                    if method.upper() == "GET":
                        session.get(f"{config.backend_url}{endpoint_url}")
                    elif method.upper() == "POST":
                        session.post(f"{config.backend_url}{endpoint_url}", json={})

                    time.sleep(0.1)  # 100ms between requests
                except Exception as e:
                    print(f"Request error: {e}")
                    continue

        with self.profile_context(profile_id):
            # Start making requests in a separate thread
            request_thread = threading.Thread(target=make_requests, daemon=True)
            request_thread.start()

            # Let it run for the specified duration
            time.sleep(duration)

        return self.active_profiles.get(profile_id)


# Global CPU profiler instance
cpu_profiler = CPUProfiler()


# Decorator for profiling functions
def profile_cpu(profile_name: str | None = None):
    """Decorator to profile CPU usage of functions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            name = profile_name or f"{func.__module__}.{func.__name__}"
            with cpu_profiler.profile_context(name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


# Async decorator for profiling async functions
def profile_cpu_async(profile_name: str | None = None):
    """Decorator to profile CPU usage of async functions"""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            name = profile_name or f"{func.__module__}.{func.__name__}"
            with cpu_profiler.profile_context(name):
                return await func(*args, **kwargs)

        return wrapper

    return decorator
