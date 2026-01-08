"""
Performance Testing Utilities and Helpers.

This module provides utility functions and classes for performance testing:
- Test data generators
- Performance assertion helpers
- Benchmark result analyzers
- CI/CD integration utilities
"""

import asyncio
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from faker import Faker

from app.core.logging import logger
from app.domain.models import GenerationConfig, PromptRequest

# =====================================================
# Test Data Generators
# =====================================================

class PerformanceTestDataGenerator:
    """Generate realistic test data for performance testing."""

    def __init__(self, seed: int = 42):
        self.fake = Faker()
        Faker.seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def generate_prompt_requests(self, count: int, complexity: str = "mixed") -> list[PromptRequest]:
        """Generate realistic prompt requests for testing."""
        requests = []

        for _i in range(count):
            if complexity == "simple":
                prompt = self.fake.sentence(nb_words=random.randint(5, 15))
            elif complexity == "complex":
                prompt = " ".join([self.fake.paragraph() for _ in range(random.randint(2, 5))])
            else:  # mixed
                if random.random() < 0.3:
                    prompt = self.fake.sentence(nb_words=random.randint(5, 15))
                elif random.random() < 0.7:
                    prompt = self.fake.paragraph()
                else:
                    prompt = " ".join([self.fake.paragraph() for _ in range(random.randint(2, 3))])

            config = GenerationConfig(
                max_tokens=random.randint(50, 500),
                temperature=random.uniform(0.0, 1.0),
                top_p=random.uniform(0.7, 1.0),
            )

            request = PromptRequest(
                prompt=prompt,
                provider=random.choice(["mock", "google", "openai"]),
                config=config
            )

            requests.append(request)

        return requests

    def generate_transformation_requests(self, count: int) -> list[dict[str, Any]]:
        """Generate transformation test requests."""
        transformation_types = [
            "simple", "advanced", "expert", "cognitive_hacking",
            "hierarchical_persona", "advanced_obfuscation", "contextual_inception"
        ]

        requests = []
        for _i in range(count):
            prompt = self.fake.paragraph()
            transformation_type = random.choice(transformation_types)

            request = {
                "prompt": prompt,
                "transformation_type": transformation_type,
                "parameters": {
                    "intensity": random.uniform(0.3, 1.0),
                    "preserve_meaning": random.choice([True, False]),
                }
            }

            requests.append(request)

        return requests

    def generate_cache_test_data(self, key_count: int, value_size_kb: float = 1.0) -> list[tuple[str, dict[str, Any]]]:
        """Generate cache test data with realistic keys and values."""
        test_data = []

        for i in range(key_count):
            # Generate cache key
            key = f"test:{self.fake.uuid4()}:{i}"

            # Generate value of specified size
            text_size = int(value_size_kb * 1024 / 4)  # Rough estimate for JSON
            value = {
                "prompt": self.fake.text(max_nb_chars=text_size // 2),
                "response": self.fake.text(max_nb_chars=text_size // 2),
                "metadata": {
                    "timestamp": time.time(),
                    "provider": random.choice(["google", "openai", "anthropic"]),
                    "tokens": random.randint(50, 500),
                }
            }

            test_data.append((key, value))

        return test_data

    def generate_websocket_messages(self, count: int) -> list[dict[str, Any]]:
        """Generate WebSocket test messages."""
        message_types = ["text", "stream_chunk", "heartbeat", "status"]
        messages = []

        for _i in range(count):
            msg_type = random.choice(message_types)

            if msg_type == "text":
                message = {
                    "type": "text",
                    "data": self.fake.sentence(),
                    "timestamp": time.time(),
                }
            elif msg_type == "stream_chunk":
                message = {
                    "type": "stream_chunk",
                    "data": {
                        "text": self.fake.words(nb=random.randint(1, 10)),
                        "is_final": random.choice([True, False]),
                    },
                    "timestamp": time.time(),
                }
            elif msg_type == "heartbeat":
                message = {
                    "type": "heartbeat",
                    "timestamp": time.time(),
                }
            else:  # status
                message = {
                    "type": "status",
                    "data": {"status": random.choice(["processing", "complete", "error"])},
                    "timestamp": time.time(),
                }

            messages.append(message)

        return messages


# =====================================================
# Performance Assertion Helpers
# =====================================================

class PerformanceAssertions:
    """Helper class for performance-related assertions."""

    @staticmethod
    def assert_response_time(response_time_ms: float, max_time_ms: float, context: str = "") -> None:
        """Assert that response time is within acceptable limits."""
        if response_time_ms > max_time_ms:
            raise AssertionError(
                f"Response time {response_time_ms:.2f}ms exceeds limit {max_time_ms}ms"
                f"{f' for {context}' if context else ''}"
            )

    @staticmethod
    def assert_throughput(operations: int, duration_seconds: float, min_ops_per_second: float, context: str = "") -> None:
        """Assert that throughput meets minimum requirements."""
        actual_ops_per_second = operations / duration_seconds
        if actual_ops_per_second < min_ops_per_second:
            raise AssertionError(
                f"Throughput {actual_ops_per_second:.2f} ops/sec below minimum {min_ops_per_second} ops/sec"
                f"{f' for {context}' if context else ''}"
            )

    @staticmethod
    def assert_memory_usage(memory_mb: float, max_memory_mb: float, context: str = "") -> None:
        """Assert that memory usage is within limits."""
        if memory_mb > max_memory_mb:
            raise AssertionError(
                f"Memory usage {memory_mb:.2f}MB exceeds limit {max_memory_mb}MB"
                f"{f' for {context}' if context else ''}"
            )

    @staticmethod
    def assert_cache_hit_ratio(hits: int, total: int, min_ratio: float, context: str = "") -> None:
        """Assert that cache hit ratio meets minimum requirements."""
        actual_ratio = hits / total if total > 0 else 0
        if actual_ratio < min_ratio:
            raise AssertionError(
                f"Cache hit ratio {actual_ratio:.2%} below minimum {min_ratio:.2%}"
                f"{f' for {context}' if context else ''}"
            )

    @staticmethod
    def assert_error_rate(errors: int, total: int, max_error_rate: float, context: str = "") -> None:
        """Assert that error rate is within acceptable limits."""
        actual_error_rate = errors / total if total > 0 else 0
        if actual_error_rate > max_error_rate:
            raise AssertionError(
                f"Error rate {actual_error_rate:.2%} exceeds maximum {max_error_rate:.2%}"
                f"{f' for {context}' if context else ''}"
            )


# =====================================================
# Benchmark Result Analyzer
# =====================================================

@dataclass
class PerformanceRegression:
    """Detected performance regression."""

    metric_name: str
    baseline_value: float
    current_value: float
    regression_percent: float
    threshold_percent: float
    severity: str  # "minor", "major", "critical"


class BenchmarkResultAnalyzer:
    """Analyze benchmark results and detect performance regressions."""

    def __init__(self, baseline_file: str | None = None, regression_threshold: float = 10.0):
        self.regression_threshold = regression_threshold
        self.baseline_data: dict[str, Any] | None = None

        if baseline_file and Path(baseline_file).exists():
            self.load_baseline(baseline_file)

    def load_baseline(self, baseline_file: str) -> None:
        """Load baseline performance data."""
        import json

        try:
            with open(baseline_file) as f:
                self.baseline_data = json.load(f)
            logger.info(f"Loaded baseline data from {baseline_file}")
        except Exception as e:
            logger.warning(f"Failed to load baseline data: {e}")
            self.baseline_data = None

    def save_baseline(self, results: dict[str, Any], baseline_file: str) -> None:
        """Save current results as new baseline."""
        import json

        try:
            with open(baseline_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Saved baseline data to {baseline_file}")
        except Exception as e:
            logger.error(f"Failed to save baseline data: {e}")

    def detect_regressions(self, current_results: dict[str, Any]) -> list[PerformanceRegression]:
        """Detect performance regressions compared to baseline."""
        if not self.baseline_data:
            logger.warning("No baseline data available for regression detection")
            return []

        regressions = []

        # Check key performance metrics
        metrics_to_check = [
            ("api_performance.p95_response_time", "ms", False),  # Lower is better
            ("llm_performance.p95_response_time", "ms", False),
            ("transformation_performance.p95_response_time", "ms", False),
            ("memory_usage.peak_mb", "MB", False),
            ("cache_performance.hit_ratio_percent", "%", True),  # Higher is better
            ("concurrent_performance.requests_per_second", "rps", True),
        ]

        for metric_path, unit, higher_is_better in metrics_to_check:
            regression = self._check_metric_regression(
                current_results, metric_path, higher_is_better, unit
            )
            if regression:
                regressions.append(regression)

        return regressions

    def _check_metric_regression(
        self,
        current_results: dict[str, Any],
        metric_path: str,
        higher_is_better: bool,
        unit: str
    ) -> PerformanceRegression | None:
        """Check a specific metric for regression."""
        # Extract values from nested dict
        baseline_value = self._get_nested_value(self.baseline_data, metric_path)
        current_value = self._get_nested_value(current_results, metric_path)

        if baseline_value is None or current_value is None:
            return None

        # Calculate percentage change
        if baseline_value == 0:
            return None

        change_percent = ((current_value - baseline_value) / baseline_value) * 100

        # Determine if this is a regression
        is_regression = False
        if higher_is_better:
            is_regression = change_percent < -self.regression_threshold
        else:
            is_regression = change_percent > self.regression_threshold

        if not is_regression:
            return None

        # Determine severity
        abs_change = abs(change_percent)
        if abs_change >= 50:
            severity = "critical"
        elif abs_change >= 25:
            severity = "major"
        else:
            severity = "minor"

        return PerformanceRegression(
            metric_name=metric_path,
            baseline_value=baseline_value,
            current_value=current_value,
            regression_percent=change_percent,
            threshold_percent=self.regression_threshold,
            severity=severity
        )

    def _get_nested_value(self, data: dict[str, Any], path: str) -> float | None:
        """Extract nested value from dictionary using dot notation."""
        try:
            keys = path.split('.')
            value = data
            for key in keys:
                value = value[key]
            return float(value)
        except (KeyError, TypeError, ValueError):
            return None

    def generate_regression_report(self, regressions: list[PerformanceRegression]) -> dict[str, Any]:
        """Generate comprehensive regression analysis report."""
        if not regressions:
            return {
                "status": "no_regressions",
                "message": "No performance regressions detected",
                "regressions": []
            }

        # Categorize by severity
        critical = [r for r in regressions if r.severity == "critical"]
        major = [r for r in regressions if r.severity == "major"]
        minor = [r for r in regressions if r.severity == "minor"]

        return {
            "status": "regressions_detected",
            "total_regressions": len(regressions),
            "by_severity": {
                "critical": len(critical),
                "major": len(major),
                "minor": len(minor)
            },
            "regressions": [
                {
                    "metric": r.metric_name,
                    "baseline": r.baseline_value,
                    "current": r.current_value,
                    "change_percent": r.regression_percent,
                    "severity": r.severity,
                }
                for r in regressions
            ],
            "recommendation": self._get_regression_recommendation(regressions)
        }

    def _get_regression_recommendation(self, regressions: list[PerformanceRegression]) -> str:
        """Get recommendation based on detected regressions."""
        critical = any(r.severity == "critical" for r in regressions)
        major = any(r.severity == "major" for r in regressions)

        if critical:
            return "BLOCK: Critical performance regressions detected. Do not deploy."
        elif major:
            return "INVESTIGATE: Major performance regressions detected. Review before deployment."
        else:
            return "MONITOR: Minor performance regressions detected. Monitor in production."


# =====================================================
# CI/CD Integration Utilities
# =====================================================

class CICDPerformanceGate:
    """Performance gate for CI/CD pipelines."""

    def __init__(
        self,
        config_file: str | None = None,
        baseline_dir: str = "./performance_baselines"
    ):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

        # Default performance gates
        self.gates = {
            "max_response_time_ms": 2000,
            "min_throughput_rps": 150,
            "max_memory_mb": 1024,
            "min_cache_hit_ratio": 0.85,
            "max_error_rate": 0.01,
            "max_regression_percent": 10.0,
        }

        # Load custom configuration if provided
        if config_file and Path(config_file).exists():
            self._load_config(config_file)

    def _load_config(self, config_file: str) -> None:
        """Load performance gate configuration."""
        import json

        try:
            with open(config_file) as f:
                config = json.load(f)
                self.gates.update(config.get("performance_gates", {}))
            logger.info(f"Loaded performance gate config from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load performance gate config: {e}")

    def evaluate_performance(self, benchmark_results: dict[str, Any]) -> dict[str, Any]:
        """Evaluate benchmark results against performance gates."""
        gate_results = {}

        # Check individual gates
        gate_results["response_time"] = self._check_response_time_gate(benchmark_results)
        gate_results["throughput"] = self._check_throughput_gate(benchmark_results)
        gate_results["memory"] = self._check_memory_gate(benchmark_results)
        gate_results["cache"] = self._check_cache_gate(benchmark_results)
        gate_results["error_rate"] = self._check_error_rate_gate(benchmark_results)
        gate_results["regressions"] = self._check_regression_gate(benchmark_results)

        # Overall pass/fail
        all_passed = all(result["passed"] for result in gate_results.values())

        return {
            "overall_result": "PASS" if all_passed else "FAIL",
            "gates": gate_results,
            "summary": {
                "total_gates": len(gate_results),
                "passed": sum(1 for r in gate_results.values() if r["passed"]),
                "failed": sum(1 for r in gate_results.values() if not r["passed"]),
            }
        }

    def _check_response_time_gate(self, results: dict[str, Any]) -> dict[str, Any]:
        """Check response time performance gate."""
        max_time = self.gates["max_response_time_ms"]

        # Extract P95 response times
        api_p95 = self._extract_metric(results, "api_performance.percentiles.p95", float('inf'))
        llm_p95 = self._extract_metric(results, "llm_performance.percentiles.p95", float('inf'))

        worst_p95 = max(api_p95, llm_p95)
        passed = worst_p95 <= max_time

        return {
            "passed": passed,
            "metric": "response_time_p95_ms",
            "actual": worst_p95,
            "threshold": max_time,
            "details": f"P95 response time: {worst_p95:.2f}ms (limit: {max_time}ms)"
        }

    def _check_throughput_gate(self, results: dict[str, Any]) -> dict[str, Any]:
        """Check throughput performance gate."""
        min_rps = self.gates["min_throughput_rps"]

        actual_rps = self._extract_metric(results, "concurrent_performance.requests_per_second", 0)
        passed = actual_rps >= min_rps

        return {
            "passed": passed,
            "metric": "throughput_rps",
            "actual": actual_rps,
            "threshold": min_rps,
            "details": f"Throughput: {actual_rps:.1f} rps (minimum: {min_rps} rps)"
        }

    def _check_memory_gate(self, results: dict[str, Any]) -> dict[str, Any]:
        """Check memory usage performance gate."""
        max_memory = self.gates["max_memory_mb"]

        peak_memory = self._extract_metric(results, "memory_report.peak_memory_mb", float('inf'))
        passed = peak_memory <= max_memory

        return {
            "passed": passed,
            "metric": "peak_memory_mb",
            "actual": peak_memory,
            "threshold": max_memory,
            "details": f"Peak memory: {peak_memory:.1f}MB (limit: {max_memory}MB)"
        }

    def _check_cache_gate(self, results: dict[str, Any]) -> dict[str, Any]:
        """Check cache hit ratio performance gate."""
        min_ratio = self.gates["min_cache_hit_ratio"]

        hit_ratio = self._extract_metric(results, "cache_analytics.hit_ratio_percent", 0) / 100
        passed = hit_ratio >= min_ratio

        return {
            "passed": passed,
            "metric": "cache_hit_ratio",
            "actual": hit_ratio,
            "threshold": min_ratio,
            "details": f"Cache hit ratio: {hit_ratio:.2%} (minimum: {min_ratio:.2%})"
        }

    def _check_error_rate_gate(self, results: dict[str, Any]) -> dict[str, Any]:
        """Check error rate performance gate."""
        max_error_rate = self.gates["max_error_rate"]

        # Calculate overall error rate
        total_errors = sum(
            self._extract_metric(results, f"{test}.error_count", 0)
            for test in ["api_performance", "llm_performance", "concurrent_performance"]
        )
        total_requests = sum(
            self._extract_metric(results, f"{test}.total_requests", 0)
            for test in ["api_performance", "llm_performance", "concurrent_performance"]
        )

        error_rate = total_errors / total_requests if total_requests > 0 else 0
        passed = error_rate <= max_error_rate

        return {
            "passed": passed,
            "metric": "error_rate",
            "actual": error_rate,
            "threshold": max_error_rate,
            "details": f"Error rate: {error_rate:.2%} (maximum: {max_error_rate:.2%})"
        }

    def _check_regression_gate(self, results: dict[str, Any]) -> dict[str, Any]:
        """Check for performance regressions."""
        max_regression = self.gates["max_regression_percent"]

        # Use the latest baseline file
        baseline_files = list(self.baseline_dir.glob("performance_report_*.json"))
        if not baseline_files:
            # No baseline to compare against
            return {
                "passed": True,
                "metric": "regression_check",
                "actual": 0,
                "threshold": max_regression,
                "details": "No baseline available for regression comparison"
            }

        # Use most recent baseline
        latest_baseline = max(baseline_files, key=lambda f: f.stat().st_mtime)

        analyzer = BenchmarkResultAnalyzer(str(latest_baseline), max_regression)
        regressions = analyzer.detect_regressions(results)

        # Check if any regressions exceed threshold
        significant_regressions = [
            r for r in regressions
            if abs(r.regression_percent) > max_regression
        ]

        passed = len(significant_regressions) == 0

        return {
            "passed": passed,
            "metric": "performance_regression",
            "actual": len(significant_regressions),
            "threshold": 0,
            "details": f"Significant regressions: {len(significant_regressions)}"
        }

    def _extract_metric(self, data: dict[str, Any], path: str, default: Any) -> Any:
        """Extract metric value from nested dictionary."""
        try:
            keys = path.split('.')
            value = data
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default


# =====================================================
# Integration Functions
# =====================================================

async def run_performance_gate_check(
    config_file: str | None = None,
    baseline_dir: str = "./performance_baselines",
    save_baseline: bool = False
) -> tuple[bool, dict[str, Any]]:
    """Run performance gate check for CI/CD."""
    from app.testing.performance_benchmarks import run_performance_benchmark

    # Run benchmark
    benchmark_results = await run_performance_benchmark()

    # Evaluate against gates
    gate = CICDPerformanceGate(config_file, baseline_dir)
    gate_results = gate.evaluate_performance(benchmark_results)

    # Save as baseline if requested
    if save_baseline:
        timestamp = int(time.time())
        baseline_file = Path(baseline_dir) / f"performance_report_{timestamp}.json"
        gate.baseline_dir.mkdir(parents=True, exist_ok=True)

        import json
        with open(baseline_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)

    passed = gate_results["overall_result"] == "PASS"
    return passed, gate_results


if __name__ == "__main__":
    # CLI for performance gate check
    import sys

    async def main():
        passed, results = await run_performance_gate_check()

        print(f"Performance Gate: {results['overall_result']}")
        print(f"Gates Passed: {results['summary']['passed']}/{results['summary']['total_gates']}")

        if not passed:
            print("Failed gates:")
            for name, result in results["gates"].items():
                if not result["passed"]:
                    print(f"  - {name}: {result['details']}")

        sys.exit(0 if passed else 1)

    asyncio.run(main())
