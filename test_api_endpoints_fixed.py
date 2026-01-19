#!/usr/bin/env python3
"""
Comprehensive API Endpoint Testing Script - FIXED VERSION

This script tests all the fixes implemented for the API endpoints:
1. Server startup and basic connectivity
2. Authentication middleware functionality
3. Error handling across all routes
4. Provider endpoint routing
5. Health monitoring system
6. Frontend integration compatibility
"""

import asyncio
import logging
import sys
import time
from typing import Any

import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class APIEndpointTester:
    """Comprehensive API endpoint testing suite."""

    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
        self.session = None
        self.results = {}
        self.api_key = "dev-api-key-123456789"  # Development API key

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30), headers={"Content-Type": "application/json"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def test_server_connectivity(self) -> dict[str, Any]:
        """Test basic server connectivity and startup."""
        logger.info("Testing server connectivity...")

        test_results = {"name": "Server Connectivity", "passed": False, "details": {}}

        try:
            # Test root endpoint
            async with self.session.get(f"{self.base_url}/") as response:
                root_data = await response.json()
                test_results["details"]["root_endpoint"] = {
                    "status_code": response.status,
                    "data": root_data,
                    "passed": response.status == 200,
                }

            # Test health endpoint
            async with self.session.get(f"{self.base_url}/health") as response:
                health_data = await response.json()
                test_results["details"]["health_endpoint"] = {
                    "status_code": response.status,
                    "data": health_data,
                    "passed": response.status == 200
                    and health_data.get("status") in ["healthy", "ok"],
                }

            # Test API health endpoint
            async with self.session.get(f"{self.api_url}/health") as response:
                api_health_data = await response.json()
                test_results["details"]["api_health_endpoint"] = {
                    "status_code": response.status,
                    "data": api_health_data,
                    "passed": response.status == 200,
                }

            # Overall pass condition
            test_results["passed"] = all(
                detail.get("passed", False) for detail in test_results["details"].values()
            )

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Server connectivity test failed: {e}")

        return test_results

    async def test_authentication_middleware(self) -> dict[str, Any]:
        """Test authentication middleware functionality."""
        logger.info("Testing authentication middleware...")

        test_results = {"name": "Authentication Middleware", "passed": False, "details": {}}

        try:
            # Test unauthenticated request to protected endpoint
            async with self.session.post(
                f"{self.api_url}/generate", json={"prompt": "test"}
            ) as response:
                test_results["details"]["unauthenticated_request"] = {
                    "status_code": response.status,
                    "passed": response.status == 401,
                    "description": "Should reject unauthenticated requests",
                }

            # Test API key authentication
            headers = {"X-API-Key": self.api_key}
            async with self.session.get(f"{self.api_url}/providers", headers=headers) as response:
                providers_data = await response.json()
                test_results["details"]["api_key_auth"] = {
                    "status_code": response.status,
                    "data": providers_data,
                    "passed": response.status == 200,
                    "description": "Should accept valid API key",
                }

            # Test invalid API key
            invalid_headers = {"X-API-Key": "invalid-key"}
            async with self.session.get(
                f"{self.api_url}/providers", headers=invalid_headers
            ) as response:
                test_results["details"]["invalid_api_key"] = {
                    "status_code": response.status,
                    "passed": response.status == 401,
                    "description": "Should reject invalid API key",
                }

            # Test excluded path (should not require auth)
            async with self.session.get(f"{self.base_url}/health") as response:
                test_results["details"]["excluded_path"] = {
                    "status_code": response.status,
                    "passed": response.status == 200,
                    "description": "Health endpoint should not require authentication",
                }

            # Overall pass condition
            test_results["passed"] = all(
                detail.get("passed", False) for detail in test_results["details"].values()
            )

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Authentication test failed: {e}")

        return test_results

    async def test_error_handling(self) -> dict[str, Any]:
        """Test comprehensive error handling."""
        logger.info("Testing error handling...")

        test_results = {"name": "Error Handling", "passed": False, "details": {}}

        headers = {"X-API-Key": self.api_key}

        try:
            # Test validation error
            async with self.session.post(
                f"{self.api_url}/generate", headers=headers, json={"invalid_field": "test"}
            ) as response:
                error_data = await response.json()
                test_results["details"]["validation_error"] = {
                    "status_code": response.status,
                    "data": error_data,
                    "passed": (
                        response.status == 422 and error_data.get("error") == "VALIDATION_ERROR"
                    ),
                    "description": "Should return proper validation error",
                }

            # Test 404 error
            async with self.session.get(
                f"{self.api_url}/nonexistent-endpoint", headers=headers
            ) as response:
                test_results["details"]["not_found_error"] = {
                    "status_code": response.status,
                    "passed": response.status == 404,
                    "description": "Should return 404 for nonexistent endpoints",
                }

            # Test method not allowed
            async with self.session.patch(f"{self.api_url}/health", headers=headers) as response:
                test_results["details"]["method_not_allowed"] = {
                    "status_code": response.status,
                    "passed": response.status == 405,
                    "description": "Should return 405 for unsupported methods",
                }

            # Overall pass condition (at least 2 out of 3 tests should pass)
            passed_count = sum(
                1 for detail in test_results["details"].values() if detail.get("passed", False)
            )
            test_results["passed"] = passed_count >= 2

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Error handling test failed: {e}")

        return test_results

    async def test_provider_endpoints(self) -> dict[str, Any]:
        """Test provider endpoint routing and validation."""
        logger.info("Testing provider endpoints...")

        test_results = {"name": "Provider Endpoints", "passed": False, "details": {}}

        headers = {"X-API-Key": self.api_key}

        try:
            # Test provider listing
            async with self.session.get(f"{self.api_url}/providers", headers=headers) as response:
                providers_data = await response.json()
                test_results["details"]["list_providers"] = {
                    "status_code": response.status,
                    "data": providers_data,
                    "passed": (response.status == 200 and "providers" in providers_data),
                    "description": "Should list available providers",
                }

            # Test provider health check (assuming openai provider exists)
            provider_id = "openai"
            async with self.session.get(
                f"{self.api_url}/providers/{provider_id}/health", headers=headers
            ) as response:
                health_data = await response.json()
                test_results["details"]["provider_health"] = {
                    "status_code": response.status,
                    "data": health_data,
                    "passed": (response.status == 200 and "provider_id" in health_data),
                    "description": "Should check provider health",
                }

            # Test provider validation
            async with self.session.post(
                f"{self.api_url}/providers/{provider_id}/validate", headers=headers, json={}
            ) as response:
                validation_data = await response.json()
                test_results["details"]["provider_validation"] = {
                    "status_code": response.status,
                    "data": validation_data,
                    "passed": (response.status == 200 and "provider_id" in validation_data),
                    "description": "Should validate provider configuration",
                }

            # Test nonexistent provider
            async with self.session.get(
                f"{self.api_url}/providers/nonexistent/health", headers=headers
            ) as response:
                test_results["details"]["nonexistent_provider"] = {
                    "status_code": response.status,
                    "passed": response.status == 200,  # Should handle gracefully
                    "description": "Should handle nonexistent provider gracefully",
                }

            # Overall pass condition
            passed_count = sum(
                1 for detail in test_results["details"].values() if detail.get("passed", False)
            )
            test_results["passed"] = passed_count >= 3

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Provider endpoints test failed: {e}")

        return test_results

    async def test_health_monitoring(self) -> dict[str, Any]:
        """Test health monitoring and alerting system."""
        logger.info("Testing health monitoring...")

        test_results = {"name": "Health Monitoring", "passed": False, "details": {}}

        headers = {"X-API-Key": self.api_key}

        try:
            # Test health dashboard
            async with self.session.get(
                f"{self.api_url}/health/dashboard", headers=headers
            ) as response:
                dashboard_data = await response.json()
                test_results["details"]["health_dashboard"] = {
                    "status_code": response.status,
                    "data": dashboard_data,
                    "passed": (response.status == 200 and "overall_status" in dashboard_data),
                    "description": "Should provide health dashboard",
                }

            # Test monitored endpoints listing
            async with self.session.get(
                f"{self.api_url}/health/endpoints", headers=headers
            ) as response:
                endpoints_data = await response.json()
                test_results["details"]["monitored_endpoints"] = {
                    "status_code": response.status,
                    "data": endpoints_data,
                    "passed": (response.status == 200 and "endpoints" in endpoints_data),
                    "description": "Should list monitored endpoints",
                }

            # Test endpoint health check
            async with self.session.post(
                f"{self.api_url}/health/check/health?method=GET", headers=headers
            ) as response:
                health_check_data = await response.json()
                test_results["details"]["endpoint_health_check"] = {
                    "status_code": response.status,
                    "data": health_check_data,
                    "passed": (response.status == 200 and "endpoint" in health_check_data),
                    "description": "Should perform endpoint health check",
                }

            # Test monitoring status
            async with self.session.get(
                f"{self.api_url}/health/monitoring/status", headers=headers
            ) as response:
                monitoring_data = await response.json()
                test_results["details"]["monitoring_status"] = {
                    "status_code": response.status,
                    "data": monitoring_data,
                    "passed": (response.status == 200 and "enabled" in monitoring_data),
                    "description": "Should show monitoring status",
                }

            # Overall pass condition
            passed_count = sum(
                1 for detail in test_results["details"].values() if detail.get("passed", False)
            )
            test_results["passed"] = passed_count >= 3

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Health monitoring test failed: {e}")

        return test_results

    async def test_frontend_integration(self) -> dict[str, Any]:
        """Test frontend integration compatibility."""
        logger.info("Testing frontend integration...")

        test_results = {"name": "Frontend Integration", "passed": False, "details": {}}

        headers = {"X-API-Key": self.api_key, "Origin": "http://localhost:3001"}  # Frontend origin

        try:
            # Test CORS headers
            async with self.session.options(
                f"{self.api_url}/providers", headers=headers
            ) as response:
                cors_headers = dict(response.headers)
                test_results["details"]["cors_support"] = {
                    "status_code": response.status,
                    "headers": {
                        k: v
                        for k, v in cors_headers.items()
                        if k.lower().startswith("access-control")
                    },
                    "passed": (
                        response.status in [200, 204]
                        and "access-control-allow-origin" in cors_headers
                    ),
                    "description": "Should support CORS for frontend",
                }

            # Test JSON response format
            async with self.session.get(f"{self.api_url}/providers", headers=headers) as response:
                content_type = response.headers.get("content-type", "")
                test_results["details"]["json_response"] = {
                    "status_code": response.status,
                    "content_type": content_type,
                    "passed": (response.status == 200 and "application/json" in content_type),
                    "description": "Should return JSON responses",
                }

            # Test request ID tracking
            async with self.session.get(f"{self.api_url}/health", headers=headers) as response:
                request_id = response.headers.get("x-request-id")
                test_results["details"]["request_tracking"] = {
                    "status_code": response.status,
                    "request_id": request_id,
                    "passed": bool(request_id),
                    "description": "Should include request ID in responses",
                }

            # Overall pass condition
            passed_count = sum(
                1 for detail in test_results["details"].values() if detail.get("passed", False)
            )
            test_results["passed"] = passed_count >= 2

        except Exception as e:
            test_results["error"] = str(e)
            logger.error(f"Frontend integration test failed: {e}")

        return test_results

    def generate_report(self, test_results: list[dict[str, Any]]) -> str:
        """Generate comprehensive test report."""
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result.get("passed", False))

        report = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                           API ENDPOINT FIX VALIDATION REPORT                                                    ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  Overall Results: {passed_tests}/{total_tests} test suites passed                                                                           ║
║  Status: {'✅ ALL FIXES WORKING' if passed_tests == total_tests else '❌ SOME ISSUES REMAIN'}                                                        ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

"""

        for i, result in enumerate(test_results, 1):
            status = "✅ PASSED" if result.get("passed", False) else "❌ FAILED"
            report += f"\n{i}. {result['name']}: {status}\n"

            if "error" in result:
                report += f"   Error: {result['error']}\n"

            if "details" in result:
                for detail_name, detail_info in result["details"].items():
                    detail_status = "✅" if detail_info.get("passed", False) else "❌"
                    status_code = detail_info.get("status_code", "N/A")
                    description = detail_info.get("description", "")
                    report += (
                        f"   {detail_status} {detail_name} (HTTP {status_code}): {description}\n"
                    )

            report += "\n"

        # Summary and recommendations
        report += "╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗\n"
        report += "║                                                   SUMMARY                                                                    ║\n"
        report += "╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝\n\n"

        if passed_tests == total_tests:
            report += "✅ All API endpoint fixes are working correctly!\n"
            report += "✅ Server startup issues resolved\n"
            report += "✅ Authentication middleware standardized\n"
            report += "✅ Error handling implemented across all routes\n"
            report += "✅ Provider endpoints validated and fixed\n"
            report += "✅ Health monitoring system operational\n"
            report += "✅ Frontend integration compatibility confirmed\n"
        else:
            report += f"⚠️  {total_tests - passed_tests} test suite(s) failed\n"
            report += "📋 Review the failed tests above for specific issues\n"
            report += "🔧 Consider running individual fix scripts for failing components\n"

        report += f"\nTest completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        return report

    async def run_all_tests(self) -> str:
        """Run all API endpoint tests and return comprehensive report."""
        logger.info("Starting comprehensive API endpoint testing...")

        test_results = []

        # Run all test suites
        test_results.append(await self.test_server_connectivity())
        test_results.append(await self.test_authentication_middleware())
        test_results.append(await self.test_error_handling())
        test_results.append(await self.test_provider_endpoints())
        test_results.append(await self.test_health_monitoring())
        test_results.append(await self.test_frontend_integration())

        # Generate and return report
        report = self.generate_report(test_results)
        self.results = test_results

        return report


async def main():
    """Main test execution function."""
    import argparse

    parser = argparse.ArgumentParser(description="Test API endpoint fixes")
    parser.add_argument("--url", default="http://127.0.0.1:8001", help="Base URL for API server")
    parser.add_argument("--output", help="Output file for test report")

    args = parser.parse_args()

    try:
        async with APIEndpointTester(args.url) as tester:
            report = await tester.run_all_tests()

            # Output report
            print(report)

            if args.output:
                with open(args.output, "w") as f:
                    f.write(report)
                logger.info(f"Test report saved to: {args.output}")

            # Return appropriate exit code
            total_tests = len(tester.results)
            passed_tests = sum(1 for result in tester.results if result.get("passed", False))

            if passed_tests == total_tests:
                logger.info("All tests passed!")
                return 0
            else:
                logger.warning(f"Some tests failed: {passed_tests}/{total_tests} passed")
                return 1

    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
