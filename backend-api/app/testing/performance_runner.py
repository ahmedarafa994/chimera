"""
Performance Testing Configuration and Runner.

This module provides configuration management and test runner
for the comprehensive performance testing suite.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# =====================================================
# Default Performance Configuration
# =====================================================

DEFAULT_PERFORMANCE_CONFIG = {
    "performance_targets": {
        "api_response_p95_ms": 2000.0,
        "llm_provider_p95_ms": 10000.0,
        "transformation_p95_ms": 1000.0,
        "concurrent_requests_per_second": 150.0,
        "memory_optimization_percent": 30.0,
        "cache_hit_ratio_percent": 85.0,
    },
    "load_test_config": {
        "duration_seconds": 60,
        "ramp_up_seconds": 10,
        "concurrent_users": 50,
        "request_rate": 10.0,
        "test_scenarios": ["api", "llm", "transformation", "websocket"],
    },
    "performance_gates": {
        "max_response_time_ms": 2000,
        "min_throughput_rps": 150,
        "max_memory_mb": 1024,
        "min_cache_hit_ratio": 0.85,
        "max_error_rate": 0.01,
        "max_regression_percent": 10.0,
    },
    "test_data_config": {
        "prompt_complexity": "mixed",
        "cache_test_size": 100,
        "concurrent_test_users": 20,
        "websocket_message_count": 50,
    },
    "output_config": {
        "results_directory": "./performance_results",
        "baseline_directory": "./performance_baselines",
        "report_formats": ["json", "csv", "html"],
        "save_baseline_on_pass": True,
        "retention_days": 30,
    },
    "ci_config": {
        "fail_on_regression": True,
        "fail_on_gate_failure": True,
        "skip_slow_tests": False,
        "parallel_execution": True,
        "timeout_minutes": 30,
    },
}


def load_performance_config(config_file: str | None = None) -> dict[str, Any]:
    """Load performance testing configuration."""
    config = DEFAULT_PERFORMANCE_CONFIG.copy()

    # Load from file if provided
    if config_file and Path(config_file).exists():
        try:
            with open(config_file) as f:
                file_config = json.load(f)
                # Deep merge configurations
                for section, values in file_config.items():
                    if section in config and isinstance(config[section], dict):
                        config[section].update(values)
                    else:
                        config[section] = values
        except Exception as e:
            print(f"Warning: Failed to load config file {config_file}: {e}")

    # Override with environment variables
    env_overrides = {
        "PERF_API_RESPONSE_TARGET": ("performance_targets", "api_response_p95_ms"),
        "PERF_LLM_RESPONSE_TARGET": ("performance_targets", "llm_provider_p95_ms"),
        "PERF_TRANSFORMATION_TARGET": ("performance_targets", "transformation_p95_ms"),
        "PERF_THROUGHPUT_TARGET": ("performance_targets", "concurrent_requests_per_second"),
        "PERF_MEMORY_TARGET": ("performance_targets", "memory_optimization_percent"),
        "PERF_CACHE_TARGET": ("performance_targets", "cache_hit_ratio_percent"),
        "PERF_TEST_DURATION": ("load_test_config", "duration_seconds"),
        "PERF_CONCURRENT_USERS": ("load_test_config", "concurrent_users"),
        "PERF_REQUEST_RATE": ("load_test_config", "request_rate"),
        "PERF_FAIL_ON_REGRESSION": ("ci_config", "fail_on_regression"),
        "PERF_SKIP_SLOW_TESTS": ("ci_config", "skip_slow_tests"),
    }

    for env_var, (section, key) in env_overrides.items():
        value = os.getenv(env_var)
        if value is not None:
            try:
                # Convert to appropriate type
                if key.endswith(("_ms", "_seconds", "_percent", "_rps")):
                    config[section][key] = float(value)
                elif key in ["concurrent_users", "duration_seconds"]:
                    config[section][key] = int(value)
                elif key in ["fail_on_regression", "skip_slow_tests"]:
                    config[section][key] = value.lower() in ("true", "1", "yes")
                else:
                    config[section][key] = value
            except ValueError:
                print(f"Warning: Invalid value for {env_var}: {value}")

    return config


def save_performance_config(config: dict[str, Any], config_file: str) -> None:
    """Save performance testing configuration."""
    try:
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Performance configuration saved to {config_file}")
    except Exception as e:
        print(f"Error: Failed to save config to {config_file}: {e}")


# =====================================================
# Performance Test Runner
# =====================================================


class PerformanceTestRunner:
    """Main runner for performance tests and benchmarks."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or load_performance_config()

    async def run_full_suite(self) -> dict[str, Any]:
        """Run the complete performance test suite."""
        from app.testing.performance_benchmarks import PerformanceBenchmarkSuite, PerformanceTargets
        from app.testing.performance_utils import CICDPerformanceGate

        print("🚀 Starting comprehensive performance test suite...")

        # Create targets from config
        targets = PerformanceTargets(**self.config["performance_targets"])

        # Create benchmark suite
        output_dir = self.config["output_config"]["results_directory"]
        suite = PerformanceBenchmarkSuite(targets, output_dir)

        try:
            # Run benchmarks
            results = await suite.run_comprehensive_benchmark()

            # Evaluate performance gates
            gate = CICDPerformanceGate(
                baseline_dir=self.config["output_config"]["baseline_directory"]
            )
            gate_results = gate.evaluate_performance(results)

            # Combine results
            final_results = {
                "benchmark_results": results,
                "gate_results": gate_results,
                "config": self.config,
                "status": "PASS" if gate_results["overall_result"] == "PASS" else "FAIL",
            }

            # Save baseline if configured and tests pass
            if (
                self.config["output_config"]["save_baseline_on_pass"]
                and gate_results["overall_result"] == "PASS"
            ):
                self._save_baseline(results)

            return final_results

        except Exception as e:
            print(f"❌ Performance test suite failed: {e}")
            raise

    async def run_quick_check(self) -> bool:
        """Run quick performance validation check."""
        from app.testing.performance_utils import run_performance_gate_check

        print("⚡ Running quick performance check...")

        try:
            passed, gate_results = await run_performance_gate_check(
                baseline_dir=self.config["output_config"]["baseline_directory"]
            )

            if passed:
                print("✅ Quick performance check: PASS")
            else:
                print("❌ Quick performance check: FAIL")
                self._print_gate_failures(gate_results)

            return passed

        except Exception as e:
            print(f"❌ Quick performance check failed: {e}")
            return False

    async def run_load_test(self, scenario: str = "api") -> dict[str, Any]:
        """Run load test for specific scenario."""
        from app.testing.performance_benchmarks import LoadTestConfig, LoadTestEngine

        print(f"🔄 Running load test for scenario: {scenario}")

        # Create load test config
        config = LoadTestConfig(**self.config["load_test_config"])
        engine = LoadTestEngine(config)

        try:
            result = await engine.run_load_test(scenario)

            print("📊 Load test completed:")
            print(f"  - Total Requests: {result.total_requests}")
            print(f"  - Success Rate: {result.success_rate:.2%}")
            print(f"  - Duration: {result.end_time - result.start_time:.2f}s")

            return {"scenario": scenario, "result": result, "success": result.success_rate >= 0.95}

        except Exception as e:
            print(f"❌ Load test failed: {e}")
            raise

    def _save_baseline(self, results: dict[str, Any]) -> None:
        """Save performance results as new baseline."""
        import time

        baseline_dir = Path(self.config["output_config"]["baseline_directory"])
        baseline_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        baseline_file = baseline_dir / f"performance_baseline_{timestamp}.json"

        try:
            with open(baseline_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📁 Saved performance baseline: {baseline_file}")
        except Exception as e:
            print(f"⚠️  Failed to save baseline: {e}")

    def _print_gate_failures(self, gate_results: dict[str, Any]) -> None:
        """Print performance gate failures."""
        print("📋 Performance Gate Results:")
        for gate_name, result in gate_results["gates"].items():
            status = "✅" if result["passed"] else "❌"
            print(f"  {status} {gate_name}: {result['details']}")


# =====================================================
# CLI Interface
# =====================================================


async def main():
    """Main CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Chimera Performance Testing Suite")
    parser.add_argument(
        "command", choices=["full", "quick", "load", "config"], help="Test command to run"
    )
    parser.add_argument("--config", "-c", help="Path to performance configuration file")
    parser.add_argument(
        "--scenario", "-s", default="api", help="Load test scenario (for 'load' command)"
    )
    parser.add_argument(
        "--output", "-o", help="Output file for configuration (for 'config' command)"
    )

    args = parser.parse_args()

    # Load configuration
    config = load_performance_config(args.config)
    runner = PerformanceTestRunner(config)

    try:
        if args.command == "full":
            print("🎯 Running full performance test suite...")
            results = await runner.run_full_suite()

            if results["status"] == "PASS":
                print("🎉 All performance tests PASSED!")
                score = results["benchmark_results"]["summary"]["overall_score"]
                print(f"📈 Overall Performance Score: {score:.1f}%")
                sys.exit(0)
            else:
                print("💥 Performance tests FAILED!")
                sys.exit(1)

        elif args.command == "quick":
            success = await runner.run_quick_check()
            sys.exit(0 if success else 1)

        elif args.command == "load":
            result = await runner.run_load_test(args.scenario)
            sys.exit(0 if result["success"] else 1)

        elif args.command == "config":
            if args.output:
                save_performance_config(config, args.output)
            else:
                print(json.dumps(config, indent=2))

    except KeyboardInterrupt:
        print("\n⏹️  Performance tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"💥 Performance tests failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
