"""
Comprehensive Memory Profiling and Heap Analysis
Monitors memory usage, detects leaks, generates heap dumps
"""

import gc
import json
import threading
import time
import tracemalloc
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import psutil

try:
    import pympler.muppy
    import pympler.summary
    import pympler.tracker

    PYMPLER_AVAILABLE = True
except ImportError:
    PYMPLER_AVAILABLE = False

try:
    import memory_profiler

    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

from profiling_config import MetricType, config


@dataclass
class MemorySnapshot:
    """Memory snapshot data"""

    timestamp: datetime
    process_memory: dict[str, Any]
    gc_stats: dict[str, Any]
    tracemalloc_stats: dict[str, Any] | None
    top_objects: list[dict[str, Any]]
    memory_growth: float | None


@dataclass
class MemoryLeakAnalysis:
    """Memory leak analysis results"""

    suspected_leaks: list[dict[str, Any]]
    growth_rate_mb_per_hour: float
    confidence_score: float
    recommendations: list[str]


@dataclass
class HeapDumpAnalysis:
    """Heap dump analysis results"""

    dump_id: str
    timestamp: datetime
    total_objects: int
    total_memory_mb: float
    largest_objects: list[dict[str, Any]]
    object_type_distribution: dict[str, int]
    growth_since_last_dump: float | None


class MemoryProfiler:
    """Comprehensive memory profiling for Chimera system"""

    def __init__(self):
        self.monitoring_active = False
        self.memory_snapshots: list[MemorySnapshot] = []
        self.monitoring_thread: threading.Thread | None = None
        self.pympler_tracker = None

        # Initialize tracemalloc if available
        if not tracemalloc.is_tracing():
            tracemalloc.start(25)  # Keep 25 frames in traceback

        # Initialize pympler tracker if available
        if PYMPLER_AVAILABLE:
            self.pympler_tracker = pympler.tracker.SummaryTracker()

    def start_monitoring(self, interval: int = 30) -> None:
        """Start continuous memory monitoring"""
        if not config.is_metric_enabled(MetricType.MEMORY):
            return

        if self.monitoring_active:
            print("Memory monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,), daemon=True
        )
        self.monitoring_thread.start()
        print(f"Started memory monitoring with {interval}s interval")

    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print("Stopped memory monitoring")

    def _monitoring_loop(self, interval: int) -> None:
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                snapshot = self._take_memory_snapshot()
                self.memory_snapshots.append(snapshot)

                # Keep only last 1000 snapshots to prevent memory bloat
                if len(self.memory_snapshots) > 1000:
                    self.memory_snapshots = self.memory_snapshots[-1000:]

                # Check for memory issues
                self._check_memory_thresholds(snapshot)

                time.sleep(interval)
            except Exception as e:
                print(f"Error in memory monitoring loop: {e}")
                time.sleep(interval)

    def _take_memory_snapshot(self) -> MemorySnapshot:
        """Take a comprehensive memory snapshot"""
        timestamp = datetime.now(UTC)

        # Get process memory information
        process = psutil.Process()
        process_memory = {
            "rss": process.memory_info().rss,
            "vms": process.memory_info().vms,
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available,
            "total": psutil.virtual_memory().total,
        }

        # Get garbage collector stats
        gc_stats = {
            "collected": gc.get_stats(),
            "counts": gc.get_count(),
            "threshold": gc.get_threshold(),
        }

        # Get tracemalloc stats if available
        tracemalloc_stats = None
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc_stats = {"current_mb": current / 1024 / 1024, "peak_mb": peak / 1024 / 1024}

        # Get top memory-consuming objects
        top_objects = self._get_top_objects()

        # Calculate memory growth
        memory_growth = None
        if len(self.memory_snapshots) > 0:
            last_snapshot = self.memory_snapshots[-1]
            memory_growth = (
                (process_memory["rss"] - last_snapshot.process_memory["rss"]) / 1024 / 1024
            )

        return MemorySnapshot(
            timestamp=timestamp,
            process_memory=process_memory,
            gc_stats=gc_stats,
            tracemalloc_stats=tracemalloc_stats,
            top_objects=top_objects,
            memory_growth=memory_growth,
        )

    def _get_top_objects(self) -> list[dict[str, Any]]:
        """Get top memory-consuming objects"""
        top_objects = []

        if PYMPLER_AVAILABLE:
            try:
                all_objects = pympler.muppy.get_objects()
                summary = pympler.summary.summarize(all_objects)

                for item in summary[:20]:  # Top 20 object types
                    top_objects.append(
                        {"type": str(item[0]), "count": item[1], "total_size": item[2]}
                    )
            except Exception as e:
                print(f"Error getting top objects with pympler: {e}")

        return top_objects

    def _check_memory_thresholds(self, snapshot: MemorySnapshot) -> None:
        """Check if memory usage exceeds thresholds"""
        memory_mb = snapshot.process_memory["rss"] / 1024 / 1024
        snapshot.process_memory["percent"]

        thresholds = config.performance_thresholds["memory"]

        if memory_mb > thresholds["critical"]:
            print(
                f"CRITICAL: Memory usage {memory_mb:.1f}MB exceeds critical threshold {thresholds['critical']}MB"
            )
        elif memory_mb > thresholds["warning"]:
            print(
                f"WARNING: Memory usage {memory_mb:.1f}MB exceeds warning threshold {thresholds['warning']}MB"
            )

        # Check for rapid growth
        if snapshot.memory_growth and snapshot.memory_growth > 100:  # 100MB growth
            print(f"WARNING: Large memory growth detected: {snapshot.memory_growth:.1f}MB")

    def generate_heap_dump(self, dump_id: str | None = None) -> HeapDumpAnalysis:
        """Generate comprehensive heap dump analysis"""
        if not dump_id:
            dump_id = f"heap_dump_{int(time.time())}"

        timestamp = datetime.now(UTC)

        # Force garbage collection before analysis
        gc.collect()

        # Get all objects in memory
        total_objects = len(gc.get_objects())

        # Get memory usage
        process = psutil.Process()
        total_memory_mb = process.memory_info().rss / 1024 / 1024

        # Analyze largest objects
        largest_objects = []
        object_type_distribution = defaultdict(int)

        if PYMPLER_AVAILABLE:
            try:
                all_objects = pympler.muppy.get_objects()
                summary = pympler.summary.summarize(all_objects)

                for item in summary[:50]:  # Top 50 object types
                    obj_info = {
                        "type": str(item[0]),
                        "count": item[1],
                        "total_size_mb": item[2] / 1024 / 1024,
                        "avg_size_bytes": item[2] / item[1] if item[1] > 0 else 0,
                    }
                    largest_objects.append(obj_info)

                # Build type distribution
                for item in summary:
                    object_type_distribution[str(item[0])] = item[1]

            except Exception as e:
                print(f"Error analyzing objects with pympler: {e}")

        # Calculate growth since last dump
        growth_since_last_dump = None
        # This would be calculated by comparing with previous dump

        analysis = HeapDumpAnalysis(
            dump_id=dump_id,
            timestamp=timestamp,
            total_objects=total_objects,
            total_memory_mb=total_memory_mb,
            largest_objects=largest_objects,
            object_type_distribution=dict(object_type_distribution),
            growth_since_last_dump=growth_since_last_dump,
        )

        # Save heap dump to file
        self._save_heap_dump(analysis)

        return analysis

    def _save_heap_dump(self, analysis: HeapDumpAnalysis) -> None:
        """Save heap dump analysis to file"""
        output_path = config.get_output_path(MetricType.MEMORY, f"{analysis.dump_id}.json")

        # Convert to serializable format
        dump_data = asdict(analysis)
        dump_data["timestamp"] = analysis.timestamp.isoformat()

        with open(output_path, "w") as f:
            json.dump(dump_data, f, indent=2)

        print(f"Heap dump saved: {output_path}")

    def analyze_memory_leaks(self, min_snapshots: int = 10) -> MemoryLeakAnalysis:
        """Analyze memory snapshots for potential leaks"""
        if len(self.memory_snapshots) < min_snapshots:
            return MemoryLeakAnalysis(
                suspected_leaks=[],
                growth_rate_mb_per_hour=0.0,
                confidence_score=0.0,
                recommendations=["Need more snapshots for leak analysis"],
            )

        # Calculate memory growth trend
        recent_snapshots = self.memory_snapshots[-min_snapshots:]
        memory_values = [s.process_memory["rss"] / 1024 / 1024 for s in recent_snapshots]
        time_values = [
            (s.timestamp - recent_snapshots[0].timestamp).total_seconds() / 3600
            for s in recent_snapshots
        ]

        # Simple linear regression for growth rate
        n = len(memory_values)
        sum_x = sum(time_values)
        sum_y = sum(memory_values)
        sum_xy = sum(x * y for x, y in zip(time_values, memory_values, strict=False))
        sum_x2 = sum(x * x for x in time_values)

        if n * sum_x2 - sum_x * sum_x != 0:
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            growth_rate_mb_per_hour = slope
        else:
            growth_rate_mb_per_hour = 0.0

        # Analyze suspected leaks
        suspected_leaks = []
        confidence_score = 0.0

        # Check for consistent growth patterns
        if growth_rate_mb_per_hour > 10:  # More than 10MB/hour growth
            confidence_score = min(growth_rate_mb_per_hour / 100, 1.0)

            # Analyze object growth patterns
            if PYMPLER_AVAILABLE and len(recent_snapshots) >= 2:
                first_objects = recent_snapshots[0].top_objects
                last_objects = recent_snapshots[-1].top_objects

                # Compare object counts
                for first_obj in first_objects:
                    for last_obj in last_objects:
                        if first_obj["type"] == last_obj["type"]:
                            growth = last_obj["count"] - first_obj["count"]
                            if growth > first_obj["count"] * 0.5:  # 50% growth
                                suspected_leaks.append(
                                    {
                                        "object_type": first_obj["type"],
                                        "initial_count": first_obj["count"],
                                        "final_count": last_obj["count"],
                                        "growth": growth,
                                        "growth_percentage": (growth / first_obj["count"]) * 100,
                                    }
                                )

        # Generate recommendations
        recommendations = []
        if growth_rate_mb_per_hour > 50:
            recommendations.append(
                "Critical memory leak detected - immediate investigation required"
            )
        elif growth_rate_mb_per_hour > 10:
            recommendations.append("Potential memory leak - monitor and investigate object growth")

        if suspected_leaks:
            recommendations.append("Focus on objects with highest growth rates")
            recommendations.append("Review code that creates instances of growing object types")

        recommendations.extend(
            [
                "Enable detailed memory tracking in production",
                "Consider implementing memory pooling for frequently allocated objects",
                "Review garbage collection settings and tuning",
            ]
        )

        return MemoryLeakAnalysis(
            suspected_leaks=suspected_leaks,
            growth_rate_mb_per_hour=growth_rate_mb_per_hour,
            confidence_score=confidence_score,
            recommendations=recommendations,
        )

    def get_memory_report(self) -> dict[str, Any]:
        """Generate comprehensive memory usage report"""
        if not self.memory_snapshots:
            return {"error": "No memory snapshots available"}

        latest_snapshot = self.memory_snapshots[-1]

        # Calculate statistics
        memory_values = [s.process_memory["rss"] / 1024 / 1024 for s in self.memory_snapshots]

        report = {
            "timestamp": latest_snapshot.timestamp.isoformat(),
            "current_memory_mb": memory_values[-1],
            "average_memory_mb": sum(memory_values) / len(memory_values),
            "peak_memory_mb": max(memory_values),
            "min_memory_mb": min(memory_values),
            "snapshots_count": len(self.memory_snapshots),
            "gc_collections": latest_snapshot.gc_stats["collected"],
            "top_objects": latest_snapshot.top_objects[:10],
            "memory_thresholds": config.performance_thresholds["memory"],
        }

        if latest_snapshot.tracemalloc_stats:
            report["tracemalloc"] = latest_snapshot.tracemalloc_stats

        # Add leak analysis if enough data
        if len(self.memory_snapshots) >= 10:
            leak_analysis = self.analyze_memory_leaks()
            report["leak_analysis"] = asdict(leak_analysis)

        return report

    def save_memory_report(self) -> str:
        """Save memory report to file"""
        report = self.get_memory_report()
        timestamp = int(time.time())
        output_path = config.get_output_path(MetricType.MEMORY, f"memory_report_{timestamp}.json")

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Memory report saved: {output_path}")
        return output_path


# Global memory profiler instance
memory_profiler = MemoryProfiler()


# Decorator for monitoring memory usage of functions
def monitor_memory(func_name: str | None = None):
    """Decorator to monitor memory usage of functions"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            name = func_name or f"{func.__module__}.{func.__name__}"

            # Take snapshot before
            snapshot_before = memory_profiler._take_memory_snapshot()

            try:
                result = func(*args, **kwargs)
            finally:
                # Take snapshot after
                snapshot_after = memory_profiler._take_memory_snapshot()

                # Log memory usage
                memory_diff = (
                    (snapshot_after.process_memory["rss"] - snapshot_before.process_memory["rss"])
                    / 1024
                    / 1024
                )

                if abs(memory_diff) > 10:  # Log if more than 10MB difference
                    print(f"Function {name} memory change: {memory_diff:+.1f}MB")

            return result

        return wrapper

    return decorator
