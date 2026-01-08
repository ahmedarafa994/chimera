"""
Chimera Load Testing Configuration

Defines test scenarios with varying intensity levels for different purposes:
- Smoke tests: Quick validation (10 users, 1 minute)
- Load tests: Normal capacity (100 users, 5 minutes)
- Stress tests: Breakpoint identification (1000 users, 10 minutes)
- Soak tests: Memory leak detection (50 users, 1 hour)
"""

from dataclasses import dataclass
from typing import Literal

TestType = Literal["smoke", "load", "stress", "soak"]


@dataclass
class LoadTestConfig:
    """Configuration for a load test scenario."""

    name: str
    test_type: TestType
    users: int
    spawn_rate: int  # Users per second
    run_time: str  # Duration in format: 1m, 5m, 1h, etc.
    host: str = "http://localhost:8001"
    headless: bool = False
    html_report: str | None = None
    csv_prefix: str | None = None

    def to_cli_args(self) -> list[str]:
        """Convert config to Locust CLI arguments."""
        args = [
            "--host", self.host,
            "--users", str(self.users),
            "--spawn-rate", str(self.spawn_rate),
            "--run-time", self.run_time,
        ]

        if self.headless:
            args.append("--headless")

        if self.html_report:
            args.extend(["--html", self.html_report])

        if self.csv_prefix:
            args.extend(["--csv", self.csv_prefix])

        return args


# Predefined test configurations
LOAD_TEST_CONFIGS: dict[TestType, LoadTestConfig] = {
    "smoke": LoadTestConfig(
        name="Smoke Test",
        test_type="smoke",
        users=10,
        spawn_rate=2,
        run_time="1m",
        headless=True,
    ),
    "load": LoadTestConfig(
        name="Load Test",
        test_type="load",
        users=100,
        spawn_rate=10,
        run_time="5m",
        html_report="load_test_report.html",
        csv_prefix="load_test",
    ),
    "stress": LoadTestConfig(
        name="Stress Test",
        test_type="stress",
        users=1000,
        spawn_rate=50,
        run_time="10m",
        html_report="stress_test_report.html",
        csv_prefix="stress_test",
    ),
    "soak": LoadTestConfig(
        name="Soak Test",
        test_type="soak",
        users=50,
        spawn_rate=5,
        run_time="1h",
        html_report="soak_test_report.html",
        csv_prefix="soak_test",
    ),
}


def get_config(test_type: TestType) -> LoadTestConfig:
    """Get load test configuration by type."""
    return LOAD_TEST_CONFIGS[test_type]


def get_all_configs() -> list[LoadTestConfig]:
    """Get all predefined load test configurations."""
    return list(LOAD_TEST_CONFIGS.values())
