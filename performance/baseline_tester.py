"""
Performance Baseline Testing for Critical User Journeys
Comprehensive testing framework for establishing performance baselines
"""

import asyncio
import importlib.util as _importlib
import json
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

# HTTP clients for API testing
try:
    import aiohttp

    HTTP_CLIENTS_AVAILABLE = True
except ImportError:
    HTTP_CLIENTS_AVAILABLE = False

from frontend_profiler import CHIMERA_USER_JOURNEYS, frontend_profiler
from profiling_config import MetricType, config

# Load testing framework availability
LOCUST_AVAILABLE = _importlib.find_spec("locust") is not None


@dataclass
class BaselineTestResult:
    """Individual baseline test result"""

    journey_name: str
    test_type: str  # api, frontend, integration
    start_time: datetime
    end_time: datetime
    duration_ms: float
    success: bool
    response_time_ms: float
    status_code: int | None
    error_message: str | None
    metrics: dict[str, Any]


@dataclass
class BaselineReport:
    """Comprehensive baseline test report"""

    report_id: str
    timestamp: datetime
    test_results: list[BaselineTestResult]
    performance_baselines: dict[str, dict[str, float]]
    regression_thresholds: dict[str, float]
    system_metrics: dict[str, Any]
    recommendations: list[str]
    baseline_status: str  # pass, fail, warning


class BaselineTester:
    """Performance baseline testing framework"""

    def __init__(self):
        self.test_results: list[BaselineTestResult] = []
        self.baseline_cache: dict[str, dict[str, float]] = {}
        self.session: aiohttp.ClientSession | None = None

    async def run_comprehensive_baseline(
        self, _test_config: dict[str, Any] | None = None
    ) -> BaselineReport:
        """Run comprehensive baseline testing for all critical user journeys"""
        report_id = f"baseline_{int(time.time())}"
        start_time = datetime.now(UTC)

        print(f"Starting comprehensive baseline testing: {report_id}")

        # Initialize HTTP session
        if HTTP_CLIENTS_AVAILABLE:
            connector = aiohttp.TCPConnector(limit=100, keepalive_timeout=30)
            self.session = aiohttp.ClientSession(
                connector=connector, timeout=aiohttp.ClientTimeout(total=30)
            )

        test_results = []
        system_metrics = {}

        try:
            # Test 1: API Endpoint Baselines
            print("Testing API endpoint baselines...")
            api_results = await self._test_api_baselines()
            test_results.extend(api_results)

            # Test 2: Frontend User Journey Baselines
            print("Testing frontend user journey baselines...")
            frontend_results = await self._test_frontend_baselines()
            test_results.extend(frontend_results)

            # Test 3: Integration Test Baselines
            print("Testing integration baselines...")
            integration_results = await self._test_integration_baselines()
            test_results.extend(integration_results)

            # Test 4: Load Test Baselines
            print("Testing load baselines...")
            load_results = await self._test_load_baselines()
            test_results.extend(load_results)

            # Collect system metrics during testing
            system_metrics = self._collect_system_metrics()

        finally:
            if self.session:
                await self.session.close()

        # Generate performance baselines
        performance_baselines = self._calculate_performance_baselines(test_results)

        # Set regression thresholds
        regression_thresholds = self._calculate_regression_thresholds(performance_baselines)

        # Generate recommendations
        recommendations = self._generate_baseline_recommendations(test_results, system_metrics)

        # Determine overall status
        baseline_status = self._determine_baseline_status(test_results, performance_baselines)

        report = BaselineReport(
            report_id=report_id,
            timestamp=start_time,
            test_results=test_results,
            performance_baselines=performance_baselines,
            regression_thresholds=regression_thresholds,
            system_metrics=system_metrics,
            recommendations=recommendations,
            baseline_status=baseline_status,
        )

        # Save report
        self._save_baseline_report(report)

        print(f"Baseline testing completed: {baseline_status.upper()}")
        return report

    async def _test_api_baselines(self) -> list[BaselineTestResult]:
        """Test API endpoint performance baselines"""
        results = []

        # Critical API endpoints from config
        api_endpoints = [
            {"path": "/health", "method": "GET", "expected_time_ms": 100},
            {"path": "/health/ready", "method": "GET", "expected_time_ms": 500},
            {"path": "/api/v1/providers", "method": "GET", "expected_time_ms": 200},
            {"path": "/api/v1/session/models", "method": "GET", "expected_time_ms": 300},
            {
                "path": "/api/v1/generate",
                "method": "POST",
                "expected_time_ms": 2000,
                "payload": {"prompt": "Test prompt", "provider": "mock"},
            },
            {
                "path": "/api/v1/transform",
                "method": "POST",
                "expected_time_ms": 1000,
                "payload": {"prompt": "Test transformation", "techniques": ["simple"]},
            },
            {
                "path": "/api/v1/generation/jailbreak/generate",
                "method": "POST",
                "expected_time_ms": 5000,
                "payload": {"prompt": "Test jailbreak", "technique": "dan"},
            },
        ]

        for endpoint in api_endpoints:
            for _attempt in range(3):  # 3 attempts per endpoint
                result = await self._test_api_endpoint(
                    endpoint["path"],
                    endpoint["method"],
                    endpoint.get("payload"),
                    endpoint["expected_time_ms"],
                )
                results.append(result)

                # Small delay between attempts
                await asyncio.sleep(0.5)

        return results

    async def _test_api_endpoint(
        self,
        path: str,
        method: str,
        payload: dict[str, Any] | None = None,
        expected_time_ms: float = 1000,
    ) -> BaselineTestResult:
        """Test individual API endpoint"""
        if not self.session:
            return self._create_failed_result("api_endpoint", path, "HTTP client not available")

        start_time = datetime.now(UTC)
        url = f"{config.backend_url}{path}"

        try:
            request_start = time.time()

            if method.upper() == "GET":
                async with self.session.get(url) as response:
                    await response.text()
                    status_code = response.status
                    success = 200 <= status_code < 300

            elif method.upper() == "POST":
                async with self.session.post(url, json=payload or {}) as response:
                    await response.text()
                    status_code = response.status
                    success = 200 <= status_code < 300

            else:
                return self._create_failed_result(
                    "api_endpoint", path, f"Unsupported method: {method}"
                )

            response_time_ms = (time.time() - request_start) * 1000

            return BaselineTestResult(
                journey_name=f"api_{method.lower()}_{path.replace('/', '_')}",
                test_type="api",
                start_time=start_time,
                end_time=datetime.now(UTC),
                duration_ms=response_time_ms,
                success=success,
                response_time_ms=response_time_ms,
                status_code=status_code,
                error_message=None if success else f"HTTP {status_code}",
                metrics={
                    "expected_time_ms": expected_time_ms,
                    "baseline_met": response_time_ms <= expected_time_ms * 1.5,  # 50% tolerance
                },
            )

        except Exception as e:
            return self._create_failed_result("api_endpoint", path, str(e))

    async def _test_frontend_baselines(self) -> list[BaselineTestResult]:
        """Test frontend user journey baselines"""
        results = []

        # Test Core Web Vitals for main pages
        frontend_pages = [
            {"url": f"{config.frontend_url}", "name": "homepage", "expected_lcp_ms": 2500},
            {
                "url": f"{config.frontend_url}/dashboard",
                "name": "dashboard",
                "expected_lcp_ms": 3000,
            },
            {
                "url": f"{config.frontend_url}/dashboard/providers",
                "name": "providers",
                "expected_lcp_ms": 2000,
            },
            {
                "url": f"{config.frontend_url}/dashboard/jailbreak",
                "name": "jailbreak",
                "expected_lcp_ms": 3500,
            },
        ]

        for page in frontend_pages:
            try:
                # Measure Core Web Vitals
                core_vitals = frontend_profiler.measure_core_web_vitals(page["url"])

                if core_vitals:
                    success = core_vitals.lcp <= page["expected_lcp_ms"]
                    result = BaselineTestResult(
                        journey_name=f"frontend_{page['name']}",
                        test_type="frontend",
                        start_time=datetime.now(UTC),
                        end_time=datetime.now(UTC),
                        duration_ms=core_vitals.lcp,
                        success=success,
                        response_time_ms=core_vitals.lcp,
                        status_code=None,
                        error_message=(
                            None
                            if success
                            else f"LCP {core_vitals.lcp}ms exceeds {page['expected_lcp_ms']}ms"
                        ),
                        metrics={
                            "lcp": core_vitals.lcp,
                            "fcp": core_vitals.fcp,
                            "cls": core_vitals.cls,
                            "ttfb": core_vitals.ttfb,
                            "expected_lcp_ms": page["expected_lcp_ms"],
                        },
                    )
                else:
                    result = self._create_failed_result(
                        "frontend", page["name"], "Could not measure Core Web Vitals"
                    )

                results.append(result)

            except Exception as e:
                results.append(self._create_failed_result("frontend", page["name"], str(e)))

        # Test user journeys
        for journey_config in CHIMERA_USER_JOURNEYS:
            try:
                journey_result = frontend_profiler.test_user_journey(journey_config)

                success = journey_result.success and len(journey_result.budget_violations) == 0

                result = BaselineTestResult(
                    journey_name=f"user_journey_{journey_result.journey_name}",
                    test_type="frontend",
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                    duration_ms=journey_result.total_duration,
                    success=success,
                    response_time_ms=journey_result.total_duration,
                    status_code=None,
                    error_message=(
                        None
                        if success
                        else "; ".join(
                            journey_result.error_messages + journey_result.budget_violations
                        )
                    ),
                    metrics={
                        "steps_completed": len([s for s in journey_result.steps if s["success"]]),
                        "total_steps": len(journey_result.steps),
                        "budget_violations": len(journey_result.budget_violations),
                    },
                )

                results.append(result)

            except Exception as e:
                results.append(
                    self._create_failed_result("frontend", journey_config["name"], str(e))
                )

        return results

    async def _test_integration_baselines(self) -> list[BaselineTestResult]:
        """Test end-to-end integration baselines"""
        results = []

        # Test complete workflows
        integration_tests = [
            {
                "name": "prompt_generation_e2e",
                "description": "Complete prompt generation workflow",
                "steps": [
                    {"type": "api", "path": "/api/v1/providers", "method": "GET"},
                    {
                        "type": "api",
                        "path": "/api/v1/generate",
                        "method": "POST",
                        "payload": {"prompt": "Create a marketing email", "provider": "mock"},
                    },
                    {"type": "delay", "duration": 0.1},
                ],
                "expected_total_ms": 3000,
            },
            {
                "name": "jailbreak_workflow_e2e",
                "description": "Complete jailbreak testing workflow",
                "steps": [
                    {
                        "type": "api",
                        "path": "/api/v1/transform",
                        "method": "POST",
                        "payload": {"prompt": "Test prompt", "techniques": ["dan_persona"]},
                    },
                    {
                        "type": "api",
                        "path": "/api/v1/generation/jailbreak/generate",
                        "method": "POST",
                        "payload": {"prompt": "Test jailbreak", "technique": "dan"},
                    },
                    {"type": "delay", "duration": 0.1},
                ],
                "expected_total_ms": 8000,
            },
        ]

        for test in integration_tests:
            result = await self._run_integration_test(test)
            results.append(result)

        return results

    async def _run_integration_test(self, test_config: dict[str, Any]) -> BaselineTestResult:
        """Run individual integration test"""
        start_time = datetime.now(UTC)
        test_start = time.time()

        try:
            step_results = []

            for step in test_config["steps"]:
                step_start = time.time()

                if step["type"] == "api":
                    # Make API call
                    url = f"{config.backend_url}{step['path']}"

                    if step["method"].upper() == "GET":
                        async with self.session.get(url) as response:
                            await response.text()
                            step_success = 200 <= response.status < 300

                    elif step["method"].upper() == "POST":
                        async with self.session.post(url, json=step.get("payload", {})) as response:
                            await response.text()
                            step_success = 200 <= response.status < 300

                elif step["type"] == "delay":
                    await asyncio.sleep(step["duration"])
                    step_success = True

                step_duration = (time.time() - step_start) * 1000
                step_results.append({"success": step_success, "duration_ms": step_duration})

            total_duration_ms = (time.time() - test_start) * 1000
            overall_success = all(s["success"] for s in step_results)
            baseline_met = total_duration_ms <= test_config["expected_total_ms"]

            return BaselineTestResult(
                journey_name=test_config["name"],
                test_type="integration",
                start_time=start_time,
                end_time=datetime.now(UTC),
                duration_ms=total_duration_ms,
                success=overall_success and baseline_met,
                response_time_ms=total_duration_ms,
                status_code=None,
                error_message=(
                    None
                    if (overall_success and baseline_met)
                    else f"Integration test failed or exceeded baseline {test_config['expected_total_ms']}ms"
                ),
                metrics={
                    "steps_executed": len(step_results),
                    "steps_successful": len([s for s in step_results if s["success"]]),
                    "expected_total_ms": test_config["expected_total_ms"],
                    "baseline_met": baseline_met,
                    "step_durations": [s["duration_ms"] for s in step_results],
                },
            )

        except Exception as e:
            return self._create_failed_result("integration", test_config["name"], str(e))

    async def _test_load_baselines(self) -> list[BaselineTestResult]:
        """Test load performance baselines"""
        results = []

        if not LOCUST_AVAILABLE:
            results.append(
                self._create_failed_result("load", "locust_test", "Locust not available")
            )
            return results

        # Simple load test scenarios
        load_scenarios = [
            {
                "name": "baseline_load_health",
                "endpoint": "/health",
                "users": 10,
                "duration": 30,  # seconds
                "expected_avg_response_ms": 50,
            },
            {
                "name": "baseline_load_providers",
                "endpoint": "/api/v1/providers",
                "users": 5,
                "duration": 30,
                "expected_avg_response_ms": 200,
            },
        ]

        for scenario in load_scenarios:
            result = await self._run_load_test(scenario)
            results.append(result)

        return results

    async def _run_load_test(self, scenario: dict[str, Any]) -> BaselineTestResult:
        """Run individual load test scenario"""
        start_time = datetime.now(UTC)

        try:
            # Create a simple load test using direct HTTP requests
            # (Since full Locust integration would be complex for this context)

            test_start = time.time()

            # Simulate concurrent users
            async def make_request():
                url = f"{config.backend_url}{scenario['endpoint']}"
                request_start = time.time()

                try:
                    async with self.session.get(url) as response:
                        await response.text()
                        request_duration = (time.time() - request_start) * 1000
                        return {
                            "success": 200 <= response.status < 300,
                            "duration_ms": request_duration,
                        }
                except Exception:
                    return {"success": False, "duration_ms": 0}

            # Run concurrent requests for the duration
            duration = scenario["duration"]
            users = scenario["users"]
            requests_per_second = 2  # Simple rate

            request_results = []

            for _ in range(duration * requests_per_second):
                batch_tasks = [make_request() for _ in range(min(users, 5))]  # Max 5 concurrent
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

                for result in batch_results:
                    if isinstance(result, dict):
                        request_results.append(result)

                await asyncio.sleep(1 / requests_per_second)

            # Analyze results
            successful_requests = [r for r in request_results if r["success"]]
            total_requests = len(request_results)
            success_rate = len(successful_requests) / total_requests if total_requests > 0 else 0

            if successful_requests:
                avg_response_ms = statistics.mean(r["duration_ms"] for r in successful_requests)
                p95_response_ms = (
                    statistics.quantiles([r["duration_ms"] for r in successful_requests], n=20)[-1]
                    if len(successful_requests) > 5
                    else avg_response_ms
                )
            else:
                avg_response_ms = 0
                p95_response_ms = 0

            baseline_met = (
                avg_response_ms <= scenario["expected_avg_response_ms"] and success_rate >= 0.95
            )

            total_duration_ms = (time.time() - test_start) * 1000

            return BaselineTestResult(
                journey_name=scenario["name"],
                test_type="load",
                start_time=start_time,
                end_time=datetime.now(UTC),
                duration_ms=total_duration_ms,
                success=baseline_met,
                response_time_ms=avg_response_ms,
                status_code=None,
                error_message=(
                    None
                    if baseline_met
                    else f"Load test failed baseline - Avg: {avg_response_ms:.0f}ms, Expected: {scenario['expected_avg_response_ms']}ms"
                ),
                metrics={
                    "total_requests": total_requests,
                    "successful_requests": len(successful_requests),
                    "success_rate": success_rate,
                    "avg_response_ms": avg_response_ms,
                    "p95_response_ms": p95_response_ms,
                    "expected_avg_response_ms": scenario["expected_avg_response_ms"],
                    "baseline_met": baseline_met,
                },
            )

        except Exception as e:
            return self._create_failed_result("load", scenario["name"], str(e))

    def _create_failed_result(
        self, test_type: str, journey_name: str, error_message: str
    ) -> BaselineTestResult:
        """Create a failed test result"""
        timestamp = datetime.now(UTC)
        return BaselineTestResult(
            journey_name=journey_name,
            test_type=test_type,
            start_time=timestamp,
            end_time=timestamp,
            duration_ms=0,
            success=False,
            response_time_ms=0,
            status_code=None,
            error_message=error_message,
            metrics={},
        )

    def _collect_system_metrics(self) -> dict[str, Any]:
        """Collect system metrics during baseline testing"""
        try:
            import psutil

            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": (
                    psutil.disk_usage("/").percent if psutil.disk_usage("/") else 0
                ),
                "network_connections": len(psutil.net_connections()),
                "timestamp": datetime.now(UTC).isoformat(),
            }
        except Exception as e:
            return {"error": f"Could not collect system metrics: {e}"}

    def _calculate_performance_baselines(
        self, test_results: list[BaselineTestResult]
    ) -> dict[str, dict[str, float]]:
        """Calculate performance baselines from test results"""
        baselines = {}

        # Group results by journey and test type
        grouped_results = {}
        for result in test_results:
            key = f"{result.test_type}_{result.journey_name}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)

        # Calculate statistics for each group
        for key, results in grouped_results.items():
            successful_results = [r for r in results if r.success]

            if successful_results:
                response_times = [r.response_time_ms for r in successful_results]

                baselines[key] = {
                    "avg_response_time_ms": statistics.mean(response_times),
                    "median_response_time_ms": statistics.median(response_times),
                    "p95_response_time_ms": (
                        statistics.quantiles(response_times, n=20)[-1]
                        if len(response_times) > 5
                        else max(response_times)
                    ),
                    "max_response_time_ms": max(response_times),
                    "success_rate": len(successful_results) / len(results),
                    "sample_size": len(results),
                }
            else:
                baselines[key] = {
                    "avg_response_time_ms": 0,
                    "median_response_time_ms": 0,
                    "p95_response_time_ms": 0,
                    "max_response_time_ms": 0,
                    "success_rate": 0,
                    "sample_size": len(results),
                }

        return baselines

    def _calculate_regression_thresholds(
        self, baselines: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Calculate regression detection thresholds"""
        thresholds = {}

        for key, baseline in baselines.items():
            # Set thresholds at 50% above baseline for warnings, 100% for critical
            avg_time = baseline["avg_response_time_ms"]
            p95_time = baseline["p95_response_time_ms"]

            thresholds[f"{key}_warning_ms"] = avg_time * 1.5
            thresholds[f"{key}_critical_ms"] = avg_time * 2.0
            thresholds[f"{key}_p95_warning_ms"] = p95_time * 1.3
            thresholds[f"{key}_p95_critical_ms"] = p95_time * 1.5

        return thresholds

    def _generate_baseline_recommendations(
        self, test_results: list[BaselineTestResult], system_metrics: dict[str, Any]
    ) -> list[str]:
        """Generate recommendations based on baseline test results"""
        recommendations = []

        # Analyze failed tests
        failed_tests = [r for r in test_results if not r.success]
        if failed_tests:
            recommendations.append(
                f"Fix {len(failed_tests)} failed baseline tests before establishing baselines"
            )

            # Categorize failures
            api_failures = [r for r in failed_tests if r.test_type == "api"]
            frontend_failures = [r for r in failed_tests if r.test_type == "frontend"]
            integration_failures = [r for r in failed_tests if r.test_type == "integration"]

            if api_failures:
                recommendations.append(
                    "API performance issues detected - review backend optimization"
                )
            if frontend_failures:
                recommendations.append(
                    "Frontend performance issues detected - review client-side optimization"
                )
            if integration_failures:
                recommendations.append("Integration issues detected - review end-to-end workflows")

        # Analyze slow tests
        slow_tests = [r for r in test_results if r.success and r.response_time_ms > 2000]
        if slow_tests:
            recommendations.append(
                f"{len(slow_tests)} tests exceeded 2-second response time - consider optimization"
            )

        # System resource recommendations
        if system_metrics.get("cpu_percent", 0) > 70:
            recommendations.append(
                "High CPU usage during testing - consider scaling or optimization"
            )

        if system_metrics.get("memory_percent", 0) > 80:
            recommendations.append("High memory usage during testing - review memory optimization")

        # General recommendations
        recommendations.extend(
            [
                "Establish automated baseline monitoring for continuous performance tracking",
                "Set up alerting for performance regressions above established thresholds",
                "Review and update baselines monthly or after major system changes",
                "Consider implementing performance budgets for new features",
            ]
        )

        return recommendations

    def _determine_baseline_status(
        self, test_results: list[BaselineTestResult], _baselines: dict[str, dict[str, float]]
    ) -> str:
        """Determine overall baseline testing status"""
        total_tests = len(test_results)
        successful_tests = len([r for r in test_results if r.success])
        success_rate = successful_tests / total_tests if total_tests > 0 else 0

        if success_rate >= 0.95:  # 95% success rate
            return "pass"
        elif success_rate >= 0.80:  # 80% success rate
            return "warning"
        else:
            return "fail"

    def _save_baseline_report(self, report: BaselineReport) -> None:
        """Save baseline report to file"""
        output_path = config.get_output_path(MetricType.FRONTEND, f"{report.report_id}.json")

        # Convert to serializable format
        report_dict = {
            "report_id": report.report_id,
            "timestamp": report.timestamp.isoformat(),
            "baseline_status": report.baseline_status,
            "test_summary": {
                "total_tests": len(report.test_results),
                "successful_tests": len([r for r in report.test_results if r.success]),
                "failed_tests": len([r for r in report.test_results if not r.success]),
                "success_rate": (
                    len([r for r in report.test_results if r.success]) / len(report.test_results)
                    if report.test_results
                    else 0
                ),
            },
            "performance_baselines": report.performance_baselines,
            "regression_thresholds": report.regression_thresholds,
            "system_metrics": report.system_metrics,
            "recommendations": report.recommendations,
            "failed_tests": [
                {
                    "journey_name": r.journey_name,
                    "test_type": r.test_type,
                    "error_message": r.error_message,
                    "response_time_ms": r.response_time_ms,
                }
                for r in report.test_results
                if not r.success
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

        print(f"Baseline report saved: {output_path}")


# Global baseline tester instance
baseline_tester = BaselineTester()


# Utility function for running baseline tests
async def run_performance_baselines(test_type: str = "comprehensive") -> BaselineReport:
    """Run performance baseline tests"""
    if test_type == "comprehensive":
        return await baseline_tester.run_comprehensive_baseline()
    else:
        # Could add specific test types here
        return await baseline_tester.run_comprehensive_baseline()
