#!/usr/bin/env python3
"""
Chimera Load Test Runner

Convenient script to run various load test scenarios.

Usage:
    # Run smoke test (quick validation)
    python tests/load/run_load_test.py smoke

    # Run load test (normal capacity)
    python tests/load/run_load_test.py load

    # Run stress test (breakpoint testing)
    python tests/load/run_load_test.py stress

    # Run soak test (memory leak detection)
    python tests/load/run_load_test.py soak

    # Custom configuration
    python tests/load/run_load_test.py --users 500 --spawn-rate 25 --run-time 15m
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.load.locust_config import (
    get_all_configs,
    get_config,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Chimera load tests with Locust",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s smoke              # Run quick smoke test
  %(prog)s load               # Run standard load test
  %(prog)s stress             # Run stress test
  %(prog)s soak               # Run soak test (1 hour)
  %(prog)s --users 200        # Custom: 200 users
  %(prog)s --users 500 --spawn-rate 50 --run-time 10m
        """,
    )

    # Predefined test types
    test_type_choices = ["smoke", "load", "stress", "soak"]
    parser.add_argument(
        "test_type",
        nargs="?",
        choices=test_type_choices,
        default="smoke",
        help="Type of load test to run (default: smoke)",
    )

    # Custom configuration options
    parser.add_argument(
        "--users",
        type=int,
        help="Number of users to simulate (overrides preset)",
    )
    parser.add_argument(
        "--spawn-rate",
        type=int,
        help="Rate at which users spawn (users/sec) (overrides preset)",
    )
    parser.add_argument(
        "--run-time",
        type=str,
        help="How long the test runs (e.g., 1m, 5m, 1h) (overrides preset)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("BACKEND_URL", "http://localhost:8001"),
        help="Target host URL (default: from BACKEND_URL env or localhost:8001)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no web UI)",
    )
    parser.add_argument(
        "--html-report",
        type=str,
        help="Generate HTML report at specified path",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available test configurations",
    )

    return parser.parse_args()


def list_configs() -> None:
    """Print all available test configurations."""
    print("\nAvailable Load Test Configurations:")
    print("=" * 70)
    for config in get_all_configs():
        print(f"\n{config.name} ({config.test_type})")
        print(f"  Users:        {config.users}")
        print(f"  Spawn Rate:   {config.spawn_rate}/sec")
        print(f"  Duration:     {config.run_time}")
    print("\n" + "=" * 70 + "\n")


def build_locust_command(args: argparse.Namespace) -> list[str]:
    """Build Locust command line arguments."""
    # Get base config if using preset
    if args.test_type:
        base_config = get_config(args.test_type)  # type: ignore[arg-type]
        users = args.users or base_config.users
        spawn_rate = args.spawn_rate or base_config.spawn_rate
        run_time = args.run_time or base_config.run_time
        headless = args.headless or base_config.headless
        html_report = args.html_report or base_config.html_report
        host = args.host or base_config.host
    else:
        # Custom configuration
        users = args.users or 10
        spawn_rate = args.spawn_rate or 1
        run_time = args.run_time or "1m"
        headless = args.headless
        html_report = args.html_report
        host = args.host

    # Build command
    cmd = [
        "locust",
        "-f", "tests/load/locustfile.py",
        "--host", host,
        "--users", str(users),
        "--spawn-rate", str(spawn_rate),
        "--run-time", run_time,
    ]

    if headless:
        cmd.append("--headless")

    if html_report:
        cmd.extend(["--html", html_report])

    return cmd


def run_load_test(cmd: list[str]) -> int:
    """Execute the load test command."""
    print("\n" + "=" * 70)
    print("CHIMERA LOAD TEST")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print("=" * 70 + "\n")

    result = subprocess.run(cmd)

    print("\n" + "=" * 70)
    if result.returncode == 0:
        print("Load test completed successfully!")
    else:
        print(f"Load test exited with code: {result.returncode}")
    print("=" * 70 + "\n")

    return result.returncode


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.list:
        list_configs()
        return 0

    cmd = build_locust_command(args)
    return run_load_test(cmd)


if __name__ == "__main__":
    sys.exit(main())
