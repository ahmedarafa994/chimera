#!/usr/bin/env python3
"""
Advanced Flame Graph Generator and Memory Analysis for Chimera
Generates CPU flame graphs, memory profiles, and hot path analysis
"""

import asyncio
import cProfile
import gc
import json
import pstats
import subprocess
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


class FlameGraphGenerator:
    """Generate CPU flame graphs using multiple profiling backends"""

    def __init__(self, output_dir: str = "performance/flame-graphs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.profilers = {}
        self.active_profiles = set()

    def profile_with_py_spy(
        self, pid: int, duration: int = 60, rate: int = 100
    ) -> str:
        """Profile using py-spy for production environments"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = (
            self.output_dir / f"py_spy_flamegraph_{pid}_{timestamp}.svg"
        )

        try:
            cmd = [
                "py-spy",
                "record",
                "--pid",
                str(pid),
                "--duration",
                str(duration),
                "--rate",
                str(rate),
                "--format",
                "flamegraph",
                "--output",
                str(output_file),
                "--subprocesses",
            ]

            print(f"Starting py-spy profiling for PID {pid} for {duration}s...")
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=duration + 10, check=False
            )

            if result.returncode == 0:
                print(f"Flame graph generated: {output_file}")
                return str(output_file)
            else:
                print(f"py-spy failed: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("py-spy profiling timed out")
            return None
        except FileNotFoundError:
            print("py-spy not found. Install with: pip install py-spy")
            return None

    def profile_with_cprofile(self, target_function, *args, **kwargs) -> str:
        """Profile using cProfile for detailed function analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prof_file = self.output_dir / f"cprofile_{timestamp}.prof"
        flamegraph_file = (
            self.output_dir / f"cprofile_flamegraph_{timestamp}.svg"
        )

        try:
            # Run profiling
            profiler = cProfile.Profile()
            profiler.enable()

            if asyncio.iscoroutinefunction(target_function):
                asyncio.run(target_function(*args, **kwargs))
            else:
                target_function(*args, **kwargs)

            profiler.disable()

            # Save profile data
            profiler.dump_stats(str(prof_file))

            # Generate flame graph using flameprof or custom converter
            self._convert_cprofile_to_flamegraph(
                str(prof_file), str(flamegraph_file)
            )

            print(f"cProfile flame graph generated: {flamegraph_file}")
            return str(flamegraph_file)

        except Exception as e:
            print(f"cProfile profiling failed: {e}")
            return None

    def _convert_cprofile_to_flamegraph(
        self, prof_file: str, output_file: str
    ):
        """Convert cProfile output to flame graph format"""
        try:
            # Try using flameprof if available
            cmd = [
                "flameprof",
                prof_file,
                "--format",
                "svg",
                "-o",
                output_file,
            ]
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=False
            )

            if result.returncode != 0:
                # Fallback to manual conversion
                self._manual_cprofile_conversion(prof_file, output_file)

        except FileNotFoundError:
            # flameprof not available, use manual conversion
            self._manual_cprofile_conversion(prof_file, output_file)

    def _manual_cprofile_conversion(self, prof_file: str, output_file: str):
        """Manually convert cProfile to flame graph format"""
        stats = pstats.Stats(prof_file)
        stats.sort_stats('cumulative')

        # Extract call stack information
        flame_data = []
        for func, (_cc, _nc, tt, ct, _callers) in stats.stats.items():
            filename, _line, funcname = func

            # Create flame graph entry
            stack = f"{filename}:{funcname}"
            flame_data.append({
                'stack': stack,
                'samples': int(ct * 1000),  # Convert to samples
                'cumulative_time': ct,
                'total_time': tt
            })

        # Generate simplified SVG (basic flame graph)
        self._generate_simple_flamegraph(flame_data, output_file)

    def _generate_simple_flamegraph(
        self, flame_data: list[dict], output_file: str
    ):
        """Generate a simple SVG flame graph"""
        # Sort by cumulative time
        flame_data.sort(key=lambda x: x["cumulative_time"], reverse=True)

        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="1200" height="600" xmlns="http://www.w3.org/2000/svg">
<style>
    .func { font-family: Verdana, sans-serif; font-size: 12px; }
    .label { font-size: 12px; text-anchor: middle; }
</style>
<rect width="1200" height="600" fill="#eeeeee"/>
"""

        y_pos = 20
        max_width = 1180

        for i, func_data in enumerate(flame_data[:20]):  # Top 20 functions
            base_time = flame_data[0]["cumulative_time"]
            width = min(
                max_width,
                (func_data["cumulative_time"] / base_time) * max_width,
            )
            color = f"hsl({(i * 15) % 360}, 60%, 70%)"

            svg_content += f"""
<rect x="10" y="{y_pos}" width="{width}" height="20" fill="{color}" stroke="black"/>
<text x="{10 + width/2}" y="{y_pos + 15}" class="label">{func_data['stack'][:50]}... ({func_data['cumulative_time']:.3f}s)</text>
"""
            y_pos += 25

        svg_content += '</svg>'

        with open(output_file, 'w') as f:
            f.write(svg_content)

    def profile_asyncio_application(
        self, app_factory, duration: int = 60
    ) -> str:
        """Profile an asyncio application"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        async def run_profiling():
            app_factory()

            # Start profiling
            profiler = cProfile.Profile()
            profiler.enable()

            # Run application for specified duration
            time.time()

            flamegraph_file = None
            try:
                # This would typically involve starting your FastAPI server
                # For demonstration, we'll simulate workload
                await asyncio.sleep(duration)
            finally:
                profiler.disable()

                # Save and convert profile
                prof_file = (
                    self.output_dir / f"asyncio_profile_{timestamp}.prof"
                )
                profiler.dump_stats(str(prof_file))

                flamegraph_file = (
                    self.output_dir / f"asyncio_flamegraph_{timestamp}.svg"
                )
                self._convert_cprofile_to_flamegraph(
                    str(prof_file), str(flamegraph_file)
                )

            return str(flamegraph_file)

        return asyncio.run(run_profiling())

class MemoryAnalyzer:
    """Advanced memory analysis and leak detection"""

    def __init__(self, output_dir: str = "performance/memory-analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots = {}
        self.tracking_active = False

    def start_tracking(self):
        """Start memory tracking"""
        if not self.tracking_active:
            tracemalloc.start(25)  # Track up to 25 stack frames
            self.tracking_active = True
            self.baseline_snapshot = tracemalloc.take_snapshot()
            print("Memory tracking started")

    def stop_tracking(self):
        """Stop memory tracking"""
        if self.tracking_active:
            tracemalloc.stop()
            self.tracking_active = False
            print("Memory tracking stopped")

    def take_snapshot(self, name: str) -> str:
        """Take a memory snapshot"""
        if not self.tracking_active:
            self.start_tracking()

        snapshot = tracemalloc.take_snapshot()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        snapshot_data = {
            "snapshot": snapshot,
            "timestamp": datetime.now(),
            "process_info": self._get_process_info(),
            "gc_stats": self._get_gc_stats(),
        }

        self.snapshots[name] = snapshot_data

        # Save detailed analysis
        report_file = (
            self.output_dir / f"memory_snapshot_{name}_{timestamp}.json"
        )
        self._generate_snapshot_report(snapshot_data, str(report_file))

        return str(report_file)

    def _get_process_info(self) -> dict[str, Any]:
        """Get current process memory information"""
        process = psutil.Process()
        memory_info = process.memory_info()

        return {
            'rss': memory_info.rss,
            'vms': memory_info.vms,
            'percent': process.memory_percent(),
            'available': psutil.virtual_memory().available,
            'total': psutil.virtual_memory().total
        }

    def _get_gc_stats(self) -> dict[str, Any]:
        """Get garbage collection statistics"""
        return {
            'counts': gc.get_count(),
            'stats': gc.get_stats(),
            'threshold': gc.get_threshold()
        }

    def _generate_snapshot_report(self, snapshot_data: dict, output_file: str):
        """Generate detailed memory snapshot report"""
        snapshot = snapshot_data['snapshot']
        stats = snapshot.statistics('lineno')

        report = {
            'timestamp': snapshot_data['timestamp'].isoformat(),
            'process_info': snapshot_data['process_info'],
            'gc_stats': snapshot_data['gc_stats'],
            'top_allocations': []
        }

        # Top memory allocations
        for i, stat in enumerate(stats[:50]):
            report['top_allocations'].append({
                'rank': i + 1,
                'size': stat.size,
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count,
                'traceback': stat.traceback.format()
            })

        # Save as JSON
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Generate human-readable report
        text_file = output_file.replace('.json', '.txt')
        with open(text_file, 'w') as f:
            f.write(f"Memory Snapshot Report - {snapshot_data['timestamp']}\n")
            f.write("=" * 60 + "\n\n")

            # Process information
            proc_info = snapshot_data['process_info']
            f.write("Process Memory Usage:\n")
            f.write(f"  RSS: {proc_info['rss'] / 1024 / 1024:.2f} MB\n")
            f.write(f"  VMS: {proc_info['vms'] / 1024 / 1024:.2f} MB\n")
            f.write(f"  Percent: {proc_info['percent']:.2f}%\n\n")

            # Top allocations
            f.write("Top 20 Memory Allocations:\n")
            f.write("-" * 40 + "\n")
            for allocation in report["top_allocations"][:20]:
                f.write(
                    f"{allocation['rank']:2d}. {allocation['size_mb']:.2f} MB "
                    f"({allocation['count']} objects)\n"
                )
                f.write(f"    {allocation['traceback'][0]}\n")

    def compare_snapshots(
        self, snapshot1_name: str, snapshot2_name: str
    ) -> str:
        """Compare two memory snapshots to detect leaks"""
        if (
            snapshot1_name not in self.snapshots
            or snapshot2_name not in self.snapshots
        ):
            raise ValueError("Invalid snapshot names")

        snapshot1 = self.snapshots[snapshot1_name]["snapshot"]
        snapshot2 = self.snapshots[snapshot2_name]["snapshot"]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = (
            self.output_dir
            / f"memory_diff_{snapshot1_name}_to_{snapshot2_name}_{timestamp}.txt"
        )

        # Compare snapshots
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        with open(report_file, 'w') as f:
            f.write(f"Memory Comparison: {snapshot1_name} → {snapshot2_name}\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("=" * 60 + "\n\n")

            # Memory growth/shrinkage
            growth_stats = [stat for stat in top_stats if stat.size_diff > 0]
            shrink_stats = [stat for stat in top_stats if stat.size_diff < 0]

            f.write(f"Memory Growth ({len(growth_stats)} locations):\n")
            f.write("-" * 40 + "\n")
            for stat in growth_stats[:20]:
                f.write(f"+{stat.size_diff / 1024 / 1024:.2f} MB: {stat}\n")

            f.write(f"\nMemory Reduction ({len(shrink_stats)} locations):\n")
            f.write("-" * 40 + "\n")
            for stat in shrink_stats[:10]:
                f.write(f"{stat.size_diff / 1024 / 1024:.2f} MB: {stat}\n")

        return str(report_file)

    def detect_memory_leaks(self, baseline_name: str, current_name: str) -> dict[str, Any]:
        """Detect potential memory leaks"""
        comparison = self.compare_snapshots(baseline_name, current_name)

        baseline_proc = self.snapshots[baseline_name]['process_info']
        current_proc = self.snapshots[current_name]['process_info']

        memory_growth = current_proc['rss'] - baseline_proc['rss']
        growth_percent = (memory_growth / baseline_proc['rss']) * 100

        return {
            'memory_growth_bytes': memory_growth,
            'memory_growth_mb': memory_growth / 1024 / 1024,
            'growth_percent': growth_percent,
            'is_potential_leak': growth_percent > 20,  # 20% growth threshold
            'comparison_report': comparison
        }

    def profile_memory_usage_over_time(self, duration: int = 300, interval: int = 10) -> str:
        """Profile memory usage over time"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = self.output_dir / f"memory_timeline_{timestamp}.json"

        timeline_data = []
        start_time = time.time()

        print(f"Starting memory timeline profiling for {duration}s...")

        while time.time() - start_time < duration:
            memory_info = self._get_process_info()
            gc_stats = self._get_gc_stats()

            timeline_data.append({
                'timestamp': time.time(),
                'elapsed': time.time() - start_time,
                'memory': memory_info,
                'gc': gc_stats
            })

            time.sleep(interval)

        # Save timeline data
        with open(data_file, 'w') as f:
            json.dump(timeline_data, f, indent=2)

        # Generate visualization script
        self._generate_memory_timeline_chart(timeline_data, str(data_file).replace('.json', '.png'))

        return str(data_file)

    def _generate_memory_timeline_chart(self, timeline_data: list[dict], output_file: str):
        """Generate memory usage timeline chart"""
        try:
            from datetime import datetime

            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt

            _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            times = [datetime.fromtimestamp(d['timestamp']) for d in timeline_data]
            rss_values = [d['memory']['rss'] / 1024 / 1024 for d in timeline_data]  # MB
            percent_values = [d['memory']['percent'] for d in timeline_data]

            # RSS usage
            ax1.plot(times, rss_values, 'b-', linewidth=2)
            ax1.set_ylabel('RSS Memory (MB)')
            ax1.set_title('Memory Usage Over Time')
            ax1.grid(True, alpha=0.3)

            # Memory percentage
            ax2.plot(times, percent_values, 'r-', linewidth=2)
            ax2.set_ylabel('Memory Percentage (%)')
            ax2.set_xlabel('Time')
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Memory timeline chart saved: {output_file}")

        except ImportError:
            print("Matplotlib not available for chart generation")

class HotPathProfiler:
    """Profile hot paths and performance bottlenecks in real-time"""

    def __init__(self, output_dir: str = "performance/hotpaths"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.call_graph = {}
        self.execution_times = {}
        self.profiling_active = False

    def start_profiling(self):
        """Start hot path profiling"""
        if not self.profiling_active:
            sys.settrace(self._trace_calls)
            self.profiling_active = True
            self.start_time = time.time()
            print("Hot path profiling started")

    def stop_profiling(self) -> str:
        """Stop profiling and generate report"""
        if self.profiling_active:
            sys.settrace(None)
            self.profiling_active = False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"hotpath_analysis_{timestamp}.json"

            self._generate_hotpath_report(str(report_file))
            print(f"Hot path profiling stopped. Report: {report_file}")

            return str(report_file)

    def _trace_calls(self, frame, event, _arg):
        """Trace function calls for hot path analysis"""
        if event == 'call':
            func_name = f"{frame.f_code.co_filename}:{frame.f_code.co_name}"

            if func_name not in self.call_graph:
                self.call_graph[func_name] = {
                    'count': 0,
                    'total_time': 0,
                    'callers': set(),
                    'callees': set()
                }

            self.call_graph[func_name]['count'] += 1

            # Track caller-callee relationship
            if frame.f_back:
                caller = f"{frame.f_back.f_code.co_filename}:{frame.f_back.f_code.co_name}"
                self.call_graph[func_name]['callers'].add(caller)

                if caller in self.call_graph:
                    self.call_graph[caller]['callees'].add(func_name)

        return self._trace_calls

    def _generate_hotpath_report(self, output_file: str):
        """Generate hot path analysis report"""
        # Convert sets to lists for JSON serialization
        serializable_data = {}
        for func_name, data in self.call_graph.items():
            serializable_data[func_name] = {
                'count': data['count'],
                'total_time': data['total_time'],
                'callers': list(data['callers']),
                'callees': list(data['callees'])
            }

        report = {
            'timestamp': datetime.now().isoformat(),
            'profiling_duration': time.time() - self.start_time,
            'total_functions': len(self.call_graph),
            'call_graph': serializable_data
        }

        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate human-readable summary
        summary_file = output_file.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("Hot Path Analysis Summary\n")
            f.write("=" * 30 + "\n\n")

            # Top functions by call count
            sorted_functions = sorted(
                self.call_graph.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )

            f.write("Top 20 Functions by Call Count:\n")
            f.write("-" * 40 + "\n")
            for func_name, data in sorted_functions[:20]:
                f.write(f"{data['count']:8d} calls - {func_name}\n")

def run_comprehensive_profiling():
    """Run comprehensive performance profiling"""

    # Find Python processes running Chimera
    chimera_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower() and proc.info['cmdline']:
                cmdline = ' '.join(proc.info['cmdline'])
                if 'chimera' in cmdline.lower() or 'uvicorn' in cmdline.lower():
                    chimera_processes.append(proc.info['pid'])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    if not chimera_processes:
        print("No Chimera processes found. Starting sample profiling...")
        # Run sample profiling
        run_sample_profiling()
        return

    print(f"Found Chimera processes: {chimera_processes}")

    # Initialize profilers
    flame_gen = FlameGraphGenerator()
    memory_analyzer = MemoryAnalyzer()
    hotpath_profiler = HotPathProfiler()

    # Start memory tracking
    memory_analyzer.start_tracking()
    memory_analyzer.take_snapshot("baseline")

    # Start hot path profiling
    hotpath_profiler.start_profiling()

    # Profile main process with py-spy
    main_pid = chimera_processes[0]
    flame_graph = flame_gen.profile_with_py_spy(main_pid, duration=60)

    # Memory profiling over time
    memory_timeline = memory_analyzer.profile_memory_usage_over_time(duration=120)

    # Take final memory snapshot
    memory_analyzer.take_snapshot("final")

    # Stop hot path profiling
    hotpath_report = hotpath_profiler.stop_profiling()

    # Detect memory leaks
    leak_analysis = memory_analyzer.detect_memory_leaks("baseline", "final")

    print("\nProfiling Summary:")
    print(f"Flame Graph: {flame_graph}")
    print(f"Memory Timeline: {memory_timeline}")
    print(f"Hot Path Report: {hotpath_report}")
    print(f"Memory Growth: {leak_analysis['memory_growth_mb']:.2f} MB ({leak_analysis['growth_percent']:.1f}%)")

    if leak_analysis['is_potential_leak']:
        print("⚠️  Potential memory leak detected!")

def run_sample_profiling():
    """Run sample profiling for demonstration"""

    def sample_workload():
        """Sample workload for profiling"""
        import math

        data = []
        for i in range(10000):
            # Simulate some computation
            # Use secrets for occasional randomness to satisfy S311
            import secrets
            value = math.sin((secrets.randbelow(1000) / 1000.0) * math.pi)
            import secrets
            data.append(value * (secrets.randbelow(100) + 1))

            # Simulate string operations
            text = f"Sample text {i} with value {value}"
            processed = text.upper().replace(" ", "_")
            data.append(len(processed))

        return data

    # Initialize profilers
    flame_gen = FlameGraphGenerator()
    memory_analyzer = MemoryAnalyzer()

    print("Running sample profiling...")

    # Profile sample function
    flame_graph = flame_gen.profile_with_cprofile(sample_workload)

    # Memory profiling
    memory_analyzer.start_tracking()
    memory_analyzer.take_snapshot("before_workload")

    sample_workload()

    memory_analyzer.take_snapshot("after_workload")

    # Analyze memory usage
    leak_analysis = memory_analyzer.detect_memory_leaks("before_workload", "after_workload")

    print("\nSample Profiling Complete:")
    print(f"Flame Graph: {flame_graph}")
    print(f"Memory Analysis: {leak_analysis}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        run_sample_profiling()
    else:
        run_comprehensive_profiling()
