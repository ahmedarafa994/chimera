#!/usr/bin/env python3
"""
Endpoint Synchronization Verification Script

This script tests all backend endpoints to ensure they are properly configured
and match the frontend expectations after synchronization fixes.

Usage:
    python verify_endpoint_synchronization.py [--verbose] [--port PORT]
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime

import aiohttp


class EndpointTester:
    def __init__(self, base_url: str = "http://localhost:8001", verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.verbose = verbose
        self.results: list[dict] = []

    async def test_endpoint(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        expected_status: int = 200,
        payload: dict | None = None,
        headers: dict | None = None,
        description: str = "",
    ) -> dict:
        """Test a single endpoint and return results."""
        url = f"{self.base_url}{path}"

        test_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if headers:
            test_headers.update(headers)

        result = {
            "method": method,
            "path": path,
            "url": url,
            "description": description,
            "status": "UNKNOWN",
            "response_status": None,
            "response_time_ms": None,
            "error": None,
            "timestamp": datetime.utcnow().isoformat(),
        }

        start_time = asyncio.get_event_loop().time()

        try:
            async with session.request(
                method,
                url,
                json=payload,
                headers=test_headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                end_time = asyncio.get_event_loop().time()
                result["response_time_ms"] = round((end_time - start_time) * 1000, 2)
                result["response_status"] = response.status

                if response.status == expected_status:
                    result["status"] = "PASS"
                    if self.verbose:
                        print(
                            f"✅ {method} {path} - {response.status} ({result['response_time_ms']}ms)"
                        )
                else:
                    result["status"] = "FAIL"
                    result["error"] = f"Expected {expected_status}, got {response.status}"
                    print(f"❌ {method} {path} - Expected {expected_status}, got {response.status}")

                # Try to get response body for additional context
                try:
                    if response.content_type and "json" in response.content_type:
                        response_data = await response.json()
                        result["response_body"] = response_data
                except Exception:
                    pass

        except TimeoutError:
            result["status"] = "TIMEOUT"
            result["error"] = "Request timed out after 30 seconds"
            print(f"⏰ {method} {path} - Timeout")
        except aiohttp.ClientConnectorError:
            result["status"] = "CONNECTION_ERROR"
            result["error"] = "Could not connect to server"
            print(f"🔌 {method} {path} - Connection error")
        except Exception as e:
            result["status"] = "ERROR"
            result["error"] = str(e)
            print(f"💥 {method} {path} - {e!s}")

        self.results.append(result)
        return result

    async def run_tests(self) -> dict:
        """Run all endpoint tests."""
        print(f"🚀 Testing endpoints at {self.base_url}")
        print("=" * 80)

        async with aiohttp.ClientSession() as session:
            # Test core endpoints
            await self.test_core_endpoints(session)

            # Test provider endpoints
            await self.test_provider_endpoints(session)

            # Test jailbreak endpoints
            await self.test_jailbreak_endpoints(session)

            # Test utility endpoints
            await self.test_utility_endpoints(session)

            # Test autodan endpoints
            await self.test_autodan_endpoints(session)

        return self.generate_report()

    async def test_core_endpoints(self, session: aiohttp.ClientSession):
        """Test core generation and transformation endpoints."""
        print("\n📡 Testing Core Endpoints")
        print("-" * 40)

        # Health check
        await self.test_endpoint(session, "GET", "/health", 200, description="Basic health check")

        # Metrics
        await self.test_endpoint(
            session, "GET", "/api/v1/metrics", 200, description="System metrics"
        )

        # Generation endpoint
        await self.test_endpoint(
            session,
            "POST",
            "/api/v1/generate",
            401,  # Expect auth required
            payload={"prompt": "test"},
            description="Text generation (auth required)",
        )

        # Transform endpoint
        await self.test_endpoint(
            session,
            "POST",
            "/api/v1/transform",
            200,
            payload={"core_request": "test prompt", "technique_suite": "basic", "potency_level": 5},
            description="Prompt transformation",
        )

    async def test_provider_endpoints(self, session: aiohttp.ClientSession):
        """Test provider management endpoints."""
        print("\n🔌 Testing Provider Endpoints")
        print("-" * 40)

        # List providers
        await self.test_endpoint(
            session, "GET", "/api/v1/providers", 200, description="List available providers"
        )

        # Models endpoint
        await self.test_endpoint(
            session, "GET", "/api/v1/models", 200, description="List available models"
        )

        # Provider config endpoints
        await self.test_endpoint(
            session,
            "GET",
            "/api/v1/provider-config/providers",
            200,
            description="Provider configuration list",
        )

        # Provider sync status
        await self.test_endpoint(
            session, "GET", "/api/v1/provider-sync/status", 200, description="Provider sync status"
        )

    async def test_jailbreak_endpoints(self, session: aiohttp.ClientSession):
        """Test jailbreak generation endpoints."""
        print("\n🔓 Testing Jailbreak Endpoints")
        print("-" * 40)

        # Main jailbreak endpoint (alias we added)
        await self.test_endpoint(
            session,
            "POST",
            "/api/v1/jailbreak",
            401,  # Expect auth required
            payload={
                "core_request": "test",
                "technique_suite": "quantum_exploit",
                "potency_level": 7,
            },
            description="Jailbreak generation alias (auth required)",
        )

        # Original jailbreak endpoint
        await self.test_endpoint(
            session,
            "POST",
            "/api/v1/generation/jailbreak/generate",
            401,  # Expect auth required
            payload={
                "core_request": "test",
                "technique_suite": "quantum_exploit",
                "potency_level": 7,
            },
            description="Original jailbreak generation (auth required)",
        )

    async def test_utility_endpoints(self, session: aiohttp.ClientSession):
        """Test utility endpoints."""
        print("\n🛠️  Testing Utility Endpoints")
        print("-" * 40)

        # Techniques list (alias we added)
        await self.test_endpoint(
            session,
            "GET",
            "/api/v1/techniques",
            200,
            description="List available techniques (alias)",
        )

        # Technique detail
        await self.test_endpoint(
            session, "GET", "/api/v1/techniques/basic", 200, description="Get technique details"
        )

        # Session endpoints
        await self.test_endpoint(
            session, "POST", "/api/v1/session", 200, description="Create session"
        )

    async def test_autodan_endpoints(self, session: aiohttp.ClientSession):
        """Test AutoDAN endpoints."""
        print("\n🤖 Testing AutoDAN Endpoints")
        print("-" * 40)

        # AutoDAN-Turbo health
        await self.test_endpoint(
            session,
            "GET",
            "/api/v1/autodan-turbo/health",
            200,
            description="AutoDAN-Turbo health check",
        )

        # AutoDAN-Turbo strategies
        await self.test_endpoint(
            session,
            "GET",
            "/api/v1/autodan-turbo/strategies",
            200,
            description="List AutoDAN strategies",
        )

        # AutoDAN-Turbo library stats
        await self.test_endpoint(
            session,
            "GET",
            "/api/v1/autodan-turbo/library/stats",
            200,
            description="AutoDAN library statistics",
        )

    def generate_report(self) -> dict:
        """Generate test report."""
        print("\n" + "=" * 80)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 80)

        total_tests = len(self.results)
        passed = len([r for r in self.results if r["status"] == "PASS"])
        failed = len([r for r in self.results if r["status"] == "FAIL"])
        errors = len(
            [r for r in self.results if r["status"] in ["ERROR", "TIMEOUT", "CONNECTION_ERROR"]]
        )

        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed}")
        print(f"❌ Failed: {failed}")
        print(f"💥 Errors: {errors}")
        print(f"Success Rate: {(passed/total_tests)*100:.1f}%")

        if failed > 0 or errors > 0:
            print("\n🚨 ISSUES FOUND:")
            for result in self.results:
                if result["status"] in ["FAIL", "ERROR", "TIMEOUT", "CONNECTION_ERROR"]:
                    print(f"  {result['method']} {result['path']}: {result['error']}")

        # Average response time for successful requests
        successful_times = [
            r["response_time_ms"] for r in self.results if r["response_time_ms"] is not None
        ]
        if successful_times:
            avg_time = sum(successful_times) / len(successful_times)
            print(f"\n⚡ Average Response Time: {avg_time:.2f}ms")

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "base_url": self.base_url,
            "summary": {
                "total_tests": total_tests,
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "success_rate": round((passed / total_tests) * 100, 1),
                "avg_response_time_ms": round(avg_time, 2) if successful_times else None,
            },
            "results": self.results,
        }

        return report


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test backend endpoint synchronization")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--port", "-p", type=int, default=8001, help="Backend port")
    parser.add_argument("--host", default="localhost", help="Backend host")
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    tester = EndpointTester(base_url, args.verbose)

    try:
        report = await tester.run_tests()

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n💾 Results saved to {args.output}")

        # Exit with error code if tests failed
        if report["summary"]["failed"] > 0 or report["summary"]["errors"] > 0:
            sys.exit(1)
        else:
            print("\n🎉 All tests passed!")
            sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n⏹️  Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
